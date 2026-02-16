"""
EnteAI - Personal AI Knowledge Assistant
Core RAG pipeline implementation with error handling and logging
"""
import os
import pickle
import re
from typing import List, Tuple, Optional
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import ollama

from config import (
    DOCUMENTS_DIR, EMBEDDING_MODEL, EMBEDDING_DEVICE, CHUNK_SIZE,
    TOP_K_RESULTS, OLLAMA_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS,
    VECTOR_INDEX_PATH, CHUNKS_PATH, METADATA_PATH, STORAGE_DIR,
    SIMILARITY_THRESHOLD, CHUNK_OVERLAP
)
from logger import setup_logger

logger = setup_logger()

# Initialize embedding model globally
try:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL, device=EMBEDDING_DEVICE)
    logger.info(f"Loaded embedding model: {EMBEDDING_MODEL} on {EMBEDDING_DEVICE}")
except Exception as e:
    logger.error(f"Failed to load embedding model: {e}")
    raise


def read_txt(file_path: str) -> str:
    """
    Read text from a .txt file
    
    Args:
        file_path: Path to the text file
        
    Returns:
        Extracted text content
        
    Raises:
        FileNotFoundError: If file doesn't exist
        UnicodeDecodeError: If file encoding is not UTF-8
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
            logger.info(f"Successfully read {len(content)} characters from {file_path}")
            return content
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except UnicodeDecodeError as e:
        logger.error(f"Encoding error reading {file_path}: {e}")
        # Try with different encoding
        try:
            with open(file_path, "r", encoding="latin-1") as file:
                content = file.read()
                logger.warning(f"Read {file_path} with latin-1 encoding")
                return content
        except Exception as e2:
            logger.error(f"Failed to read file with alternative encoding: {e2}")
            raise


def read_pdf(file_path: str) -> str:
    """
    Extract text from a PDF file
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Extracted text content
        
    Raises:
        Exception: If PDF reading fails
    """
    try:
        reader = PdfReader(file_path)
        text = ""
        
        for i, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text() or ""
                page_text = page_text.replace("\n", " ")
                text += page_text
            except Exception as e:
                logger.warning(f"Failed to extract text from page {i} of {file_path}: {e}")
                continue
        
        if not text.strip():
            logger.warning(f"No text extracted from PDF: {file_path}")
            return ""
        
        logger.info(f"Extracted {len(text)} characters from PDF: {file_path}")
        logger.debug(f"First 500 chars: {text[:500]}")
        
        return text
    except Exception as e:
        logger.error(f"Failed to read PDF {file_path}: {e}")
        raise


def load_documents() -> List[Tuple[str, str]]:
    """
    Load all documents from the documents directory
    
    Returns:
        List of tuples (filename, text_content)
    """
    documents = []
    
    if not os.path.exists(DOCUMENTS_DIR):
        logger.warning(f"Documents directory not found: {DOCUMENTS_DIR}")
        return documents
    
    files = os.listdir(DOCUMENTS_DIR)
    logger.info(f"Found {len(files)} files in {DOCUMENTS_DIR}")
    
    for filename in files:
        file_path = os.path.join(DOCUMENTS_DIR, filename)
        
        try:
            if filename.endswith(".txt"):
                text = read_txt(file_path)
                documents.append((filename, text))
            elif filename.endswith(".pdf"):
                text = read_pdf(file_path)
                documents.append((filename, text))
            else:
                logger.warning(f"Skipping unsupported file type: {filename}")
        except Exception as e:
            logger.error(f"Failed to load document {filename}: {e}")
            continue
    
    logger.info(f"Successfully loaded {len(documents)} documents")
    return documents


def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into chunks with overlap for better context preservation
    
    Args:
        text: Input text to split
        chunk_size: Maximum size of each chunk
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    try:
        # Split by sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence exceeds chunk size
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Keep overlap from previous chunk
                words = current_chunk.split()
                overlap_text = " ".join(words[-overlap:]) if len(words) > overlap else current_chunk
                current_chunk = overlap_text + " " + sentence + " "
            else:
                current_chunk += sentence + " "
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        logger.info(f"Split text into {len(chunks)} chunks")
        return chunks
    except Exception as e:
        logger.error(f"Error splitting text: {e}")
        # Fallback: simple character-based splitting
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size-overlap)]
        return chunks


def create_embeddings(chunks: List[str]) -> np.ndarray:
    """
    Create embeddings for text chunks
    
    Args:
        chunks: List of text chunks
        
    Returns:
        Numpy array of embeddings
        
    Raises:
        Exception: If embedding creation fails
    """
    try:
        logger.info(f"Creating embeddings for {len(chunks)} chunks...")
        embeddings = embedding_model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
        logger.info(f"Created embeddings with shape: {embeddings.shape}")
        return embeddings
    except Exception as e:
        logger.error(f"Failed to create embeddings: {e}")
        raise


def build_vector_store(embeddings: np.ndarray) -> faiss.Index:
    """
    Build FAISS vector store from embeddings
    
    Args:
        embeddings: Numpy array of embeddings
        
    Returns:
        FAISS index
        
    Raises:
        Exception: If vector store creation fails
    """
    try:
        dimension = embeddings.shape[1]
        logger.info(f"Building FAISS index with dimension: {dimension}")
        
        index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        
        logger.info(f"Added {index.ntotal} vectors to FAISS index")
        return index
    except Exception as e:
        logger.error(f"Failed to build vector store: {e}")
        raise


def search(query: str, index: faiss.Index, chunks: List[str], metadata: Optional[List[dict]] = None, 
           top_k: int = TOP_K_RESULTS) -> List[dict]:
    """
    Search for relevant chunks using semantic similarity
    
    Args:
        query: Search query
        index: FAISS index
        chunks: List of text chunks
        metadata: Optional metadata for each chunk
        top_k: Number of results to return
        
    Returns:
        List of dictionaries with chunk text, score, and metadata
    """
    try:
        query_embedding = embedding_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        scores, indices = index.search(query_embedding, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= SIMILARITY_THRESHOLD:
                result = {
                    "text": chunks[idx],
                    "score": float(score),
                    "index": int(idx)
                }
                if metadata and idx < len(metadata):
                    result["metadata"] = metadata[idx]
                results.append(result)
        
        logger.info(f"Found {len(results)} relevant chunks for query: '{query[:50]}...'")
        return results
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return []


def ask_llm(context: str, question: str) -> str:
    """
    Query the LLM with context and question
    
    Args:
        context: Retrieved context from knowledge base
        question: User's question
        
    Returns:
        LLM response
        
    Raises:
        Exception: If LLM query fails
    """
    prompt = f"""You are EnteAI, a personal knowledge assistant.
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know based on the provided documents."

Context:
{context}

Question:
{question}

Answer:"""
    
    try:
        logger.info(f"Querying LLM with model: {OLLAMA_MODEL}")
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": LLM_TEMPERATURE,
                "num_predict": LLM_MAX_TOKENS
            }
        )
        
        answer = response["message"]["content"]
        logger.info(f"LLM response length: {len(answer)} characters")
        return answer
    except Exception as e:
        logger.error(f"LLM query failed: {e}")
        raise Exception(f"Failed to get response from LLM. Is Ollama running? Error: {e}")


def save_knowledge(index: faiss.Index, chunks: List[str], metadata: Optional[List[dict]] = None) -> None:
    """
    Save knowledge base to disk
    
    Args:
        index: FAISS index
        chunks: List of text chunks
        metadata: Optional metadata for each chunk
    """
    try:
        os.makedirs(STORAGE_DIR, exist_ok=True)
        
        faiss.write_index(index, VECTOR_INDEX_PATH)
        logger.info(f"Saved FAISS index to {VECTOR_INDEX_PATH}")
        
        with open(CHUNKS_PATH, "wb") as f:
            pickle.dump(chunks, f)
        logger.info(f"Saved {len(chunks)} chunks to {CHUNKS_PATH}")
        
        if metadata:
            with open(METADATA_PATH, "wb") as f:
                pickle.dump(metadata, f)
            logger.info(f"Saved metadata to {METADATA_PATH}")
    except Exception as e:
        logger.error(f"Failed to save knowledge base: {e}")
        raise


def load_knowledge() -> Tuple[Optional[faiss.Index], Optional[List[str]], Optional[List[dict]]]:
    """
    Load knowledge base from disk
    
    Returns:
        Tuple of (index, chunks, metadata) or (None, None, None) if not found
    """
    try:
        if os.path.exists(VECTOR_INDEX_PATH) and os.path.exists(CHUNKS_PATH):
            index = faiss.read_index(VECTOR_INDEX_PATH)
            logger.info(f"Loaded FAISS index with {index.ntotal} vectors")
            
            with open(CHUNKS_PATH, "rb") as f:
                chunks = pickle.load(f)
            logger.info(f"Loaded {len(chunks)} chunks")
            
            metadata = None
            if os.path.exists(METADATA_PATH):
                with open(METADATA_PATH, "rb") as f:
                    metadata = pickle.load(f)
                logger.info(f"Loaded metadata for {len(metadata)} chunks")
            
            return index, chunks, metadata
        else:
            logger.warning("Knowledge base not found")
            return None, None, None
    except Exception as e:
        logger.error(f"Failed to load knowledge base: {e}")
        return None, None, None


if __name__ == "__main__":
    logger.info("Starting EnteAI CLI...")
    
    index, all_chunks, metadata = load_knowledge()
    
    if index is None:
        print("Building knowledge base...")
        try:
            documents = load_documents()
            
            if not documents:
                print("No documents found. Please add documents to the 'documents' folder.")
                exit(1)
            
            all_chunks = []
            chunk_metadata = []
            
            for filename, doc in documents:
                chunks = split_text(doc)
                all_chunks.extend(chunks)
                # Add metadata for each chunk
                chunk_metadata.extend([{"source": filename} for _ in chunks])
            
            embeddings = create_embeddings(all_chunks)
            index = build_vector_store(embeddings)
            
            save_knowledge(index, all_chunks, chunk_metadata)
            metadata = chunk_metadata
            print("Knowledge base saved ✅")
        except Exception as e:
            logger.error(f"Failed to build knowledge base: {e}")
            print(f"Error: {e}")
            exit(1)
    else:
        print("Knowledge base loaded ⚡")
    
    print("EnteAI is ready 🤍 (type 'exit' to quit)")
    
    while True:
        try:
            question = input("\nAsk EnteAI: ")
            if question.lower() == "exit":
                break
            
            if not question.strip():
                continue
            
            results = search(question, index, all_chunks, metadata)
            
            if not results:
                print("\n❌ EnteAI: I couldn't find anything relevant in your knowledge base.")
                continue
            
            context = "\n\n".join([r["text"] for r in results])
            answer = ask_llm(context, question)
            
            print(f"\n💡 EnteAI: {answer}")
            
            # Show sources
            print("\n📚 Sources:")
            for i, result in enumerate(results, 1):
                source = result.get("metadata", {}).get("source", "Unknown")
                score = result.get("score", 0)
                print(f"  {i}. {source} (relevance: {score:.2f})")
                
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            print(f"\n❌ Error: {e}")
