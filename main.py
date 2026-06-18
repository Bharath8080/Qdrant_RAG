# This is the main file you run with: streamlit run main.py
# It is the entire UI and it glues all the other modules together.

import streamlit as st
import yaml
import os
import sys

# Make sure Python can find our src/ folder
sys.path.insert(0, os.path.dirname(__file__))

from src.ingestion.loader import load_pdf
from src.chunking.chunker import split_text_into_chunks
from src.embeddings.embedder import Embedder
from src.vectordb.vector_store import VectorStore
from src.retrieval.retriever import Retriever
from src.llm.llm_client import LLMClient
from src.prompts.prompt_templates import build_rag_prompt


# ─── Load config ──────────────────────────────────────────────────────────────

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


# ─── Cache expensive objects so they are not reloaded on every page refresh ───

@st.cache_resource
def load_embedder():
    """Loads the embedding models once and caches them."""
    return Embedder(
        dense_model_name=config["dense_model"],
        sparse_model_name=config["sparse_model"],
    )

@st.cache_resource
def load_vector_store():
    """Connects to Qdrant once and caches the connection."""
    return VectorStore(
        qdrant_path=config["qdrant_path"],
        collection_name=config["collection_name"],
    )

@st.cache_resource
def load_llm():
    """Loads the Groq LLM client once and caches it."""
    return LLMClient(
        model=config["groq_model"],
        max_tokens=config["max_tokens"],
    )


# ─── Streamlit Page ────────────────────────────────────────────────────────────

st.set_page_config(page_title="RAG PDF Chat", page_icon="📄", layout="centered")

st.title("📄 Chat with Your PDFs")
st.caption("Powered by Qdrant (local) · FastEmbed · SPLADE · Groq LLaMA 3.3 70B")

# ─── Sidebar: PDF Upload ───────────────────────────────────────────────────────

with st.sidebar:
    st.header("Upload PDFs")
    
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF files. They will be processed and stored locally."
    )
    
    process_button = st.button("Process PDFs", type="primary")
    
    st.divider()
    
    # Option to clear the database and start fresh
    if st.button("Clear Database", type="secondary"):
        try:
            vs = load_vector_store()
            vs.delete_collection()
            # Clear the cache so the collection is re-created
            load_vector_store.clear()
            st.success("Database cleared! Refresh the page.")
        except Exception as e:
            st.error(f"Error clearing database: {e}")

# ─── Process Uploaded PDFs ────────────────────────────────────────────────────

if process_button and uploaded_files:
    embedder = load_embedder()
    vector_store = load_vector_store()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Processing: {uploaded_file.name}...")
        
        # Step 1: Extract text from PDF
        raw_text = load_pdf(uploaded_file)
        
        if not raw_text.strip():
            st.warning(f"Could not extract text from '{uploaded_file.name}'. It might be a scanned PDF.")
            continue
        
        # Step 2: Split text into chunks
        chunks = split_text_into_chunks(
            text=raw_text,
            chunk_size=config["chunk_size"],
            overlap=config["chunk_overlap"],
        )
        
        # Step 3: Create embeddings for all chunks
        embedded_chunks = embedder.embed_chunks(chunks)
        
        # Step 4: Store in Qdrant
        vector_store.store_chunks(
            embedded_chunks=embedded_chunks,
            source_filename=uploaded_file.name,
        )
        
        # Update progress bar
        progress_bar.progress((i + 1) / len(uploaded_files))
    
    status_text.text("")
    progress_bar.empty()
    st.success(f"✅ Successfully processed {len(uploaded_files)} file(s)! You can now ask questions below.")

elif process_button and not uploaded_files:
    st.warning("Please upload at least one PDF file first.")

# ─── Chat Interface ────────────────────────────────────────────────────────────

st.divider()
st.subheader("Ask a Question")

# Keep chat history so previous messages stay visible
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User types a question
user_question = st.chat_input("Ask anything about your uploaded PDFs...")

if user_question:
    # Show the user's question in the chat
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)
    
    # Generate the answer
    with st.chat_message("assistant"):
        with st.spinner("Searching your documents..."):
            try:
                embedder = load_embedder()
                vector_store = load_vector_store()
                llm = load_llm()
                
                # Step 1: Retrieve relevant chunks using hybrid search
                retriever = Retriever(
                    embedder=embedder,
                    vector_store=vector_store,
                    top_k=config["top_k"],
                )
                relevant_chunks = retriever.retrieve(user_question)
                
                if not relevant_chunks:
                    answer = "No documents found in the database. Please upload and process some PDFs first."
                else:
                    # Step 2: Build prompt with context
                    prompt = build_rag_prompt(
                        context_chunks=relevant_chunks,
                        user_question=user_question,
                    )
                    
                    # Step 3: Get answer from LLM
                    answer = llm.get_answer(prompt)
                
                st.markdown(answer)
                
                # Show which documents were used (expandable section)
                if relevant_chunks:
                    with st.expander("📚 Sources used"):
                        for j, chunk in enumerate(relevant_chunks):
                            st.caption(f"**Source {j+1}:** {chunk['source']}")
                            st.text(chunk["text"][:300] + "..." if len(chunk["text"]) > 300 else chunk["text"])
                            st.divider()
            
            except Exception as e:
                answer = f"Something went wrong: {str(e)}"
                st.error(answer)
    
    # Save assistant's answer to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer})
