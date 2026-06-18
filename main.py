# This is the main file you run with: streamlit run main.py
# It is the entire UI and it glues all the other modules together.

import streamlit as st
import yaml
import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Make sure Python can find our src/ folder
sys.path.insert(0, os.path.dirname(__file__))

from src.ingestion.loader import load_pdf
from src.chunking.chunker import split_text_into_chunks
from src.embeddings.embedder import Embedder
from src.vectordb.vector_store import VectorStore
from src.retrieval.retriever import Retriever
from src.llm.llm_client import LLMClient
from src.prompts.prompt_templates import build_rag_prompt
from src.reranking.reranker import Reranker


# ─── Load config ──────────────────────────────────────────────────────────────

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


# ─── Cache expensive objects ──────────────────────────────────────────────────

@st.cache_resource
def load_embedder():
    return Embedder(dense_model_name=config["dense_model"], sparse_model_name=config["sparse_model"])

@st.cache_resource
def load_vector_store():
    return VectorStore(qdrant_path=config["qdrant_path"], collection_name=config["collection_name"])

@st.cache_resource
def load_reranker():
    return Reranker(model_name=config["rerank_model"])

@st.cache_resource
def load_llm():
    return LLMClient(model=config["groq_model"], max_tokens=config["max_tokens"])


# ─── Streamlit Page ────────────────────────────────────────────────────────────

st.set_page_config(page_title="RAG PDF Chat", page_icon="📄", layout="centered")

st.title("📄 Chat with Your PDFs")
st.caption("Powered by Qdrant · BM25 · Cross-Encoder Reranker · Groq LLaMA 3.3 70B")


# ─── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("📂 Upload PDFs")

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF files. They will be processed and stored locally.",
    )
    process_button = st.button("Process PDFs", type="primary")

    st.divider()

    if st.button("🗑️ Clear Database", type="secondary"):
        try:
            vs = load_vector_store()
            vs.delete_collection()
            load_vector_store.clear()
            st.success("Database cleared! Refresh the page.")
        except Exception as e:
            st.error(f"Error: {e}")



# ─── Process Uploaded PDFs ────────────────────────────────────────────────────

if process_button and uploaded_files:
    embedder = load_embedder()
    vector_store = load_vector_store()
    progress_bar = st.progress(0)
    status_text = st.empty()

    import datetime
    import os

    for i, uploaded_file in enumerate(uploaded_files):
        base_name, ext = os.path.splitext(uploaded_file.name)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{base_name}_{timestamp}_{i}{ext}"

        status_text.text(f"Processing: {unique_filename}...")
        raw_text = load_pdf(uploaded_file)

        if not raw_text.strip():
            st.warning(f"Could not extract text from '{uploaded_file.name}'. Might be a scanned PDF.")
            continue

        chunks = split_text_into_chunks(
            text=raw_text,
            chunk_size=config["chunk_size"],
            overlap=config["chunk_overlap"],
        )
        embedded_chunks = embedder.embed_chunks(chunks)
        vector_store.store_chunks(embedded_chunks=embedded_chunks, source_filename=unique_filename)
        progress_bar.progress((i + 1) / len(uploaded_files))

    status_text.text("")
    progress_bar.empty()
    st.success(f"✅ Successfully processed {len(uploaded_files)} file(s)! Ask questions below.")

elif process_button and not uploaded_files:
    st.warning("Please upload at least one PDF file first.")


# ─── Chat Interface ────────────────────────────────────────────────────────────

st.divider()
st.subheader("Ask a Question")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Render latency metrics if saved
        if message.get("metrics"):
            m = message["metrics"]
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("🔍 Retrieval", f"{m['retrieval']:.3f}s")
            col2.metric("🎯 Reranking", f"{m['reranking']:.3f}s")
            col3.metric("⚡ Generation", f"{m['generation']:.3f}s")
            col4.metric("⏱️ Total", f"{m['total']:.3f}s")
            


user_question = st.chat_input("Ask anything about your uploaded PDFs...")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Searching, reranking, and generating..."):
            try:
                embedder     = load_embedder()
                vector_store = load_vector_store()
                reranker     = load_reranker()
                llm          = load_llm()

                retriever = Retriever(
                    embedder=embedder,
                    vector_store=vector_store,
                    top_k=config["top_k"],
                )

                relevant_chunks = []
                retrieval_time = 0.0
                reranking_time = 0.0
                generation_time = 0.0
                total_time = 0.0

                # ── Direct RAG execution ────────────────────────────────────────
                import time
                t0 = time.time()
                candidates = retriever.retrieve(user_question)
                retrieval_time = time.time() - t0

                if not candidates:
                    answer = "No documents found. Please upload and process some PDFs first."
                    total_time = retrieval_time
                else:
                    # ── Reranking ──────────────────────────────────────────────
                    t1 = time.time()
                    relevant_chunks = reranker.rerank(
                        query=user_question,
                        chunks=candidates,
                        top_k=config["rerank_top_k"],
                    )
                    reranking_time = time.time() - t1

                    # ── Generation ─────────────────────────────────────────────
                    t2 = time.time()
                    prompt = build_rag_prompt(
                        context_chunks=relevant_chunks,
                        user_question=user_question,
                    )
                    answer = llm.get_answer(prompt)
                    generation_time = time.time() - t2
                    total_time = time.time() - t0

                st.markdown(answer)

                # ── Latency Metrics ───────────────────────────────────────────
                if relevant_chunks:
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("🔍 Retrieval", f"{retrieval_time:.3f}s")
                    col2.metric("🎯 Reranking", f"{reranking_time:.3f}s")
                    col3.metric("⚡ Generation", f"{generation_time:.3f}s")
                    col4.metric("⏱️ Total", f"{total_time:.3f}s")



            except Exception as e:
                answer = f"Something went wrong: {str(e)}"
                st.error(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "metrics": {
            "retrieval": retrieval_time,
            "reranking": reranking_time,
            "generation": generation_time,
            "total": total_time,
        } if relevant_chunks else None,
    })

