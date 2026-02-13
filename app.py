import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import ollama
import pickle
import re

DOCUMENTS_DIR = "documents"

embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

def read_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text() or ""
        page_text = page_text.replace("\n", " ")
        text += page_text

    print("\n===== PDF EXTRACTED TEXT =====\n")
    print(text[:2000])   # show first 2000 characters

    return text


def load_documents():
    texts = []
    for filename in os.listdir(DOCUMENTS_DIR):
        file_path = os.path.join(DOCUMENTS_DIR, filename)

        if filename.endswith(".txt"):
            texts.append(read_txt(file_path))
        elif filename.endswith(".pdf"):
            texts.append(read_pdf(file_path))

    return texts

def split_text(text, chunk_size=300):
    sentences = re.split(r'(?<=[.!?]) +', text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk)
            current_chunk = sentence + " "

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def create_embeddings(chunks):
    return embedding_model.encode(chunks, convert_to_numpy=True)


def build_vector_store(embeddings):
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    faiss.normalize_L2(embeddings)   # ⭐ IMPORTANT
    index.add(embeddings)

    return index


def search(query, index, chunks, top_k=5):
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)

    faiss.normalize_L2(query_embedding)   # ⭐ ALSO IMPORTANT

    _, indices = index.search(query_embedding, top_k)

    return [chunks[i] for i in indices[0]]


def ask_llm(context, question):
    prompt = f"""
You are EnteAI, a personal knowledge assistant.
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}
"""
    response = ollama.chat(
    model="phi",
    messages=[{"role": "user", "content": prompt}],
    options={
        "temperature": 0.3,
        "num_predict": 150
    }
)

    return response["message"]["content"]
def save_knowledge(index, chunks):
    faiss.write_index(index, "storage/vector.index")
    with open("storage/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

def load_knowledge():
    if os.path.exists("storage/vector.index") and os.path.exists("storage/chunks.pkl"):
        index = faiss.read_index("storage/vector.index")
        with open("storage/chunks.pkl", "rb") as f:
            chunks = pickle.load(f)
        return index, chunks
    return None, None

if __name__ == "__main__":
    index, all_chunks = load_knowledge()

    if index is None:
        print("Building knowledge base...")
        documents = load_documents()

        all_chunks = []
        for doc in documents:
            all_chunks.extend(split_text(doc))

        embeddings = create_embeddings(all_chunks)
        index = build_vector_store(embeddings)

        save_knowledge(index, all_chunks)
        print("Knowledge base saved ✅")
    else:
        print("Knowledge base loaded ⚡")

    print("EnteAI is ready 🤍 (type 'exit' to quit)")

    while True:
        question = input("\nAsk EnteAI: ")
        if question.lower() == "exit":
            break

        relevant_chunks = search(question, index, all_chunks)
        context = "\n\n".join(relevant_chunks)

        answer = ask_llm(context, question)
        print("\n EnteAI:", answer)

