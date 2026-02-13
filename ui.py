import streamlit as st
import shutil
import os
import time

from app import load_knowledge, search, ask_llm, save_knowledge, split_text, create_embeddings, build_vector_store

st.set_page_config(page_title="EnteaAI", page_icon="")

# # 🌙 Dark Mode Toggle
# dark_mode = st.toggle("🌙 Dark Mode")

# if dark_mode:
#     st.markdown(
#         """
#         <style>
#         body { background-color: #0e1117; color: white; }
#         </style>
#         """,
#         unsafe_allow_html=True
#     )

st.title(" EnteaAI")
st.write("Your Personal AI Knowledge Agent")

# 📂 File Upload
uploaded_files = st.file_uploader(
    "Upload Documents",
    type=["txt", "pdf"],
    accept_multiple_files=True
)

index, chunks = load_knowledge()



if uploaded_files and st.button("Update Knowledge Base 🚀"):

    if os.path.exists("storage"):
        shutil.rmtree("storage", ignore_errors=True)
        time.sleep(0.2)

    os.makedirs("storage", exist_ok=True)

    st.info("Processing documents...")

    if os.path.exists("documents"):
        shutil.rmtree("documents", ignore_errors=True)
        time.sleep(0.2)

    os.makedirs("documents", exist_ok=True)

    for file in uploaded_files:
        with open(f"documents/{file.name}", "wb") as f:
            f.write(file.getbuffer())

    all_chunks = []

    from app import load_documents
    documents = load_documents()

    for doc in documents:
        all_chunks.extend(split_text(doc))

    embeddings = create_embeddings(all_chunks)
    index = build_vector_store(embeddings)

    save_knowledge(index, all_chunks)

    index, chunks = load_knowledge()

    st.success("Knowledge base updated ✅")



# Stop if KB missing
if index is None:
    st.error("Knowledge base not found. Upload documents first.")
    st.stop()

# 🧹 Clear Chat Button
if st.button(" Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# 🧠 Chat Memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat Input
question = st.chat_input("Ask EnteaAI something...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    # ✅ Show user message
    with st.chat_message("user"):
        st.write(question)

    # ✅ Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking... 🤔"):

            relevant_chunks = search(question, index, chunks)

            if not relevant_chunks:
                st.write("I couldn't find anything relevant in your knowledge base.")

            else:
                context = "\n\n".join(relevant_chunks)
                answer = ask_llm(context, question)

                placeholder = st.empty()
                typed_text = ""

                import time
                for char in answer:
                    typed_text += char
                    placeholder.write(typed_text)
                    time.sleep(0.01)
