"""
EnteAI - Streamlit User Interface
Enhanced with error handling, progress tracking, and source attribution
"""
import streamlit as st
import shutil
import os
import time
from typing import List, Dict

from app import (
    load_knowledge, search, ask_llm, save_knowledge, 
    split_text, create_embeddings, build_vector_store, load_documents
)
from config import ENABLE_TYPING_EFFECT, TYPING_SPEED, MAX_FILE_SIZE_MB
from logger import setup_logger

logger = setup_logger("enteai.ui")

# Page configuration
st.set_page_config(
    page_title="EnteAI - Personal Knowledge Assistant",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .source-box {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .relevance-score {
        color: #0068c9;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.title(" EnteAI")
st.write("Your Personal AI Knowledge Assistant")

# Sidebar for configuration and stats
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Display knowledge base stats
    if os.path.exists("storage/vector.index"):
        st.success("✅ Knowledge Base Active")
        try:
            index, chunks, metadata = load_knowledge()
            if index:
                st.metric("Total Chunks", len(chunks) if chunks else 0)
                st.metric("Vector Dimensions", index.d)
                
                # Show loaded documents
                if metadata:
                    sources = set(m.get("source", "Unknown") for m in metadata)
                    st.write("📚 **Loaded Documents:**")
                    for source in sources:
                        st.write(f"- {source}")
        except Exception as e:
            st.error(f"Error loading stats: {e}")
    else:
        st.warning("⚠️ No Knowledge Base Found")
        st.info("Upload documents to get started!")
    
    st.divider()
    
    # Clear chat button
    if st.button("🧹 Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    # Reset knowledge base button
    if st.button("🔄 Reset Knowledge Base", type="secondary"):
        if os.path.exists("storage"):
            shutil.rmtree("storage", ignore_errors=True)
            st.success("Knowledge base reset!")
            time.sleep(1)
            st.rerun()

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📂 Upload Documents")
    
with col2:
    # File size info
    st.caption(f"Max file size: {MAX_FILE_SIZE_MB}MB")

# File uploader
uploaded_files = st.file_uploader(
    "Upload your documents (PDF or TXT)",
    type=["txt", "pdf"],
    accept_multiple_files=True,
    help=f"Upload PDF or TXT files (max {MAX_FILE_SIZE_MB}MB each)"
)

# Validate file sizes
if uploaded_files:
    valid_files = []
    for file in uploaded_files:
        file_size_mb = file.size / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            st.error(f"❌ {file.name} exceeds {MAX_FILE_SIZE_MB}MB limit ({file_size_mb:.2f}MB)")
        else:
            valid_files.append(file)
    
    if valid_files:
        st.success(f"✅ {len(valid_files)} file(s) ready to upload")

# Process uploaded documents
if uploaded_files and st.button(" Update Knowledge Base", type="primary"):
    try:
        # Validate files again
        valid_files = [f for f in uploaded_files if f.size / (1024 * 1024) <= MAX_FILE_SIZE_MB]
        
        if not valid_files:
            st.error("No valid files to process!")
            st.stop()
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Clean up old storage
        status_text.text("🗑️ Cleaning up old storage...")
        progress_bar.progress(10)
        
        if os.path.exists("storage"):
            shutil.rmtree("storage", ignore_errors=True)
            time.sleep(0.2)
        os.makedirs("storage", exist_ok=True)
        
        # Step 2: Save uploaded files
        status_text.text("💾 Saving uploaded files...")
        progress_bar.progress(20)
        
        if os.path.exists("documents"):
            shutil.rmtree("documents", ignore_errors=True)
            time.sleep(0.2)
        os.makedirs("documents", exist_ok=True)
        
        for file in valid_files:
            with open(f"documents/{file.name}", "wb") as f:
                f.write(file.getbuffer())
        
        logger.info(f"Saved {len(valid_files)} files to documents directory")
        
        # Step 3: Load and process documents
        status_text.text("📖 Loading documents...")
        progress_bar.progress(30)
        
        documents = load_documents()
        
        if not documents:
            st.error("❌ Failed to load any documents!")
            logger.error("No documents loaded")
            st.stop()
        
        # Step 4: Split into chunks
        status_text.text(" Splitting documents into chunks...")
        progress_bar.progress(50)
        
        all_chunks = []
        chunk_metadata = []
        
        for filename, doc in documents:
            chunks = split_text(doc)
            all_chunks.extend(chunks)
            chunk_metadata.extend([{"source": filename} for _ in chunks])
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        
        # Step 5: Create embeddings
        status_text.text("🧮 Creating embeddings...")
        progress_bar.progress(70)
        
        embeddings = create_embeddings(all_chunks)
        
        # Step 6: Build vector store
        status_text.text("🗄️ Building vector database...")
        progress_bar.progress(85)
        
        index = build_vector_store(embeddings)
        
        # Step 7: Save knowledge base
        status_text.text("💾 Saving knowledge base...")
        progress_bar.progress(95)
        
        save_knowledge(index, all_chunks, chunk_metadata)
        
        # Complete
        progress_bar.progress(100)
        status_text.text("✅ Knowledge base updated successfully!")
        
        st.success(f"""
        ✅ **Knowledge Base Updated!**
        - Documents processed: {len(documents)}
        - Total chunks: {len(all_chunks)}
        - Vector dimensions: {embeddings.shape[1]}
        """)
        
        logger.info("Knowledge base update completed successfully")
        time.sleep(2)
        st.rerun()
        
    except Exception as e:
        logger.error(f"Error updating knowledge base: {e}")
        st.error(f"❌ Error: {str(e)}")
        st.info("💡 Make sure Ollama is running and all dependencies are installed.")

st.divider()

# Load knowledge base
try:
    index, chunks, metadata = load_knowledge()
except Exception as e:
    logger.error(f"Error loading knowledge base: {e}")
    st.error(f"❌ Error loading knowledge base: {e}")
    index, chunks, metadata = None, None, None

# Stop if KB missing
if index is None:
    st.warning("⚠️ Knowledge base not found. Please upload documents first.")
    st.stop()

# Chat interface
st.subheader("💬 Chat with Your Documents")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
        # Display sources if available
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📚 View Sources"):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"""
                    <div class="source-box">
                        <strong>{i}. {source['source']}</strong><br>
                        <span class="relevance-score">Relevance: {source['score']:.2%}</span><br>
                        <em>{source['text'][:200]}...</em>
                    </div>
                    """, unsafe_allow_html=True)

# Chat input
question = st.chat_input("Ask EnteAI something about your documents...")

if question:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": question})
    
    # Display user message
    with st.chat_message("user"):
        st.write(question)
    
    # Generate assistant response
    with st.chat_message("assistant"):
        try:
            with st.spinner("🤔 Thinking..."):
                # Search for relevant chunks
                results = search(question, index, chunks, metadata)
                
                if not results:
                    response = "I couldn't find anything relevant in your knowledge base. Try rephrasing your question or upload more documents."
                    st.write(response)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response
                    })
                else:
                    # Prepare context
                    context = "\n\n".join([r["text"] for r in results])
                    
                    # Get LLM response
                    answer = ask_llm(context, question)
                    
                    # Display with typing effect if enabled
                    if ENABLE_TYPING_EFFECT:
                        placeholder = st.empty()
                        typed_text = ""
                        for char in answer:
                            typed_text += char
                            placeholder.write(typed_text)
                            time.sleep(TYPING_SPEED)
                    else:
                        st.write(answer)
                    
                    # Prepare sources for display
                    sources = []
                    for result in results:
                        sources.append({
                            "source": result.get("metadata", {}).get("source", "Unknown"),
                            "score": result.get("score", 0),
                            "text": result.get("text", "")
                        })
                    
                    # Display sources
                    with st.expander("📚 View Sources"):
                        for i, source in enumerate(sources, 1):
                            st.markdown(f"""
                            <div class="source-box">
                                <strong>{i}. {source['source']}</strong><br>
                                <span class="relevance-score">Relevance: {source['score']:.2%}</span><br>
                                <em>{source['text'][:200]}...</em>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                    
                    logger.info(f"Successfully answered query: {question[:50]}...")
                    
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            error_msg = f"❌ Error: {str(e)}"
            st.error(error_msg)
            
            if "ollama" in str(e).lower():
                st.info("💡 Make sure Ollama is running. Run `ollama serve` in your terminal.")
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg
            })

# Footer
st.divider()
st.caption("Built with  using Streamlit, FAISS, and Ollama")
