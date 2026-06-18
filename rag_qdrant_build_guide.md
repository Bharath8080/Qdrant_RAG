# RAG Application — Complete Build Guide
**Stack:** Qdrant (local) · FastEmbed · SPLADE · Groq (LLaMA 3.3 70B) · Streamlit

---

## What We Are Building

A RAG (Retrieval-Augmented Generation) app where a user:
1. Opens a Streamlit web page
2. Uploads one or more PDF files
3. Asks questions about those PDFs
4. Gets smart answers powered by hybrid search (semantic + keyword) + Groq LLM

---

## Project Structure

```
rag_project/
├── README.md
├── requirements.txt
├── .env                        ← your API keys go here
├── .gitignore
├── config.yaml                 ← chunk size, model names, settings
│
├── src/
│   ├── ingestion/
│   │   ├── __init__.py
│   │   └── loader.py           ← reads PDF and extracts text
│   │
│   ├── chunking/
│   │   ├── __init__.py
│   │   └── chunker.py          ← splits text into small chunks
│   │
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── embedder.py         ← creates dense + sparse vectors
│   │
│   ├── vectordb/
│   │   ├── __init__.py
│   │   └── vector_store.py     ← Qdrant operations (store + search)
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   └── retriever.py        ← hybrid search logic
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   └── llm_client.py       ← Groq LLM calls
│   │
│   └── prompts/
│       ├── __init__.py
│       └── prompt_templates.py ← prompt we send to the LLM
│
├── logs/
│   └── app.log
│
└── main.py                     ← Streamlit UI entry point
```

---

## Step 1 — Set Up Your Environment

### 1.1 Create a virtual environment

```bash
python -m venv venv

# On Mac/Linux
source venv/bin/activate

# On Windows
venv\Scripts\activate
```

### 1.2 Install all dependencies

Create `requirements.txt` with this content:

```
streamlit
qdrant-client[fastembed]
fastembed
PyPDF2
groq
python-dotenv
pyyaml
```

Then install:

```bash
pip install -r requirements.txt
```

> **Why these packages?**
> - `qdrant-client[fastembed]` — Qdrant local DB + FastEmbed bundled together
> - `fastembed` — generates both dense (BGE) and sparse (SPLADE) embeddings
> - `PyPDF2` — reads text out of PDF files
> - `groq` — talks to Groq API (free, fast LLaMA 3.3 70B)
> - `python-dotenv` — loads `.env` file so API keys are not hardcoded

---

## Step 2 — Create Config Files

### `.env` file (never commit this to GitHub!)

```env
GROQ_API_KEY=your_groq_api_key_here
```

Get your free Groq API key at: https://console.groq.com

### `.gitignore`

```
.env
venv/
__pycache__/
*.pyc
qdrant_storage/
logs/
```

### `config.yaml`

```yaml
# Embedding models
dense_model: "BAAI/bge-small-en-v1.5"
sparse_model: "prithivida/Splade_PP_en_v1"

# Text chunking
chunk_size: 500
chunk_overlap: 50

# Qdrant local storage path (like SQLite, saves to disk)
qdrant_path: "./qdrant_storage"

# Collection name in Qdrant
collection_name: "rag_documents"

# How many chunks to retrieve
top_k: 5

# LLM settings
groq_model: "llama-3.3-70b-versatile"
max_tokens: 1024
```

---

## Step 3 — Write the Source Files

### `src/__init__.py` (empty file, needed for Python to recognize it as a package)

Create empty `__init__.py` files in every folder under `src/`.

```bash
# On Mac/Linux
touch src/__init__.py
touch src/ingestion/__init__.py
touch src/chunking/__init__.py
touch src/embeddings/__init__.py
touch src/vectordb/__init__.py
touch src/retrieval/__init__.py
touch src/llm/__init__.py
touch src/prompts/__init__.py
```

---

### `src/ingestion/loader.py` — Read PDFs

```python
# This file reads a PDF file and returns all the text from it

import PyPDF2


def load_pdf(file) -> str:
    """
    Takes a PDF file (from Streamlit uploader) and returns all its text.
    
    Args:
        file: the uploaded file object from Streamlit
    
    Returns:
        A single string with all the text from every page
    """
    pdf_reader = PyPDF2.PdfReader(file)
    
    all_text = ""
    
    for page_number in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_number]
        page_text = page.extract_text()
        
        if page_text:  # some pages might be empty or have only images
            all_text += page_text + "\n"
    
    return all_text
```

---

### `src/chunking/chunker.py` — Split Text into Chunks

```python
# This file splits a long text into smaller overlapping chunks
# Smaller chunks = better search results (more focused)

def split_text_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """
    Splits a big piece of text into smaller overlapping chunks.
    
    Why overlap? So that sentences near chunk boundaries are not lost.
    
    Args:
        text: the full extracted text from the PDF
        chunk_size: how many characters per chunk
        overlap: how many characters to repeat from the previous chunk
    
    Returns:
        A list of text chunks (strings)
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Get one chunk
        chunk = text[start:end]
        
        if chunk.strip():  # skip empty chunks
            chunks.append(chunk)
        
        # Move forward, but step back by 'overlap' so we have some repetition
        start = end - overlap
    
    return chunks
```

---

### `src/embeddings/embedder.py` — Create Vectors from Text

```python
# This file creates two types of vectors for every text chunk:
# 1. Dense vector  → captures MEANING  (using BAAI/bge-small-en-v1.5)
# 2. Sparse vector → captures KEYWORDS (using SPLADE)

from fastembed import TextEmbedding, SparseTextEmbedding


class Embedder:
    def __init__(self, dense_model_name: str, sparse_model_name: str):
        print("Loading embedding models... (first time takes a minute to download)")
        
        # Dense model for semantic search
        self.dense_model = TextEmbedding(model_name=dense_model_name)
        
        # Sparse model for keyword search
        self.sparse_model = SparseTextEmbedding(model_name=sparse_model_name)
        
        print("Embedding models loaded!")

    def get_dense_embedding(self, text: str):
        """Returns a dense vector (list of floats) for the given text."""
        embeddings = list(self.dense_model.embed([text]))
        return embeddings[0].tolist()

    def get_sparse_embedding(self, text: str):
        """Returns a sparse vector (indices + values) for the given text."""
        sparse_embeddings = list(self.sparse_model.embed([text]))
        sparse = sparse_embeddings[0]
        return sparse.indices.tolist(), sparse.values.tolist()

    def embed_chunks(self, chunks: list) -> list:
        """
        Takes a list of text chunks and returns a list of dicts,
        each containing the chunk text + both types of vectors.
        """
        results = []
        
        for i, chunk in enumerate(chunks):
            print(f"  Embedding chunk {i+1}/{len(chunks)}...")
            
            dense_vec = self.get_dense_embedding(chunk)
            sparse_indices, sparse_values = self.get_sparse_embedding(chunk)
            
            results.append({
                "text": chunk,
                "dense_vector": dense_vec,
                "sparse_indices": sparse_indices,
                "sparse_values": sparse_values,
            })
        
        return results
```

---

### `src/vectordb/vector_store.py` — Qdrant Storage

```python
# This file handles everything related to Qdrant:
# - Creating a collection (like a table in a database)
# - Storing embedded chunks
# - Searching for relevant chunks

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    SparseVectorParams,
    PointStruct,
    SparseVector,
    NamedVector,
    NamedSparseVector,
    SearchRequest,
)
import uuid


class VectorStore:
    def __init__(self, qdrant_path: str, collection_name: str, dense_vector_size: int = 384):
        """
        Connects to Qdrant in LOCAL mode (saves to disk like SQLite).
        No Docker, no server needed!
        
        Args:
            qdrant_path: folder path where Qdrant will save its data
            collection_name: name for the collection (like a table name)
            dense_vector_size: size of dense vectors (384 for bge-small)
        """
        # This creates a local Qdrant instance that saves data to disk
        self.client = QdrantClient(path=qdrant_path)
        self.collection_name = collection_name
        self.dense_vector_size = dense_vector_size
        
        # Create collection if it does not exist yet
        self._create_collection_if_needed()

    def _create_collection_if_needed(self):
        """Creates the Qdrant collection only if it doesn't already exist."""
        existing = [c.name for c in self.client.get_collections().collections]
        
        if self.collection_name not in existing:
            print(f"Creating new Qdrant collection: '{self.collection_name}'")
            
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    # Slot for dense vectors
                    "dense": VectorParams(
                        size=self.dense_vector_size,
                        distance=Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    # Slot for sparse vectors (SPLADE)
                    "sparse": SparseVectorParams()
                },
            )
            print("Collection created!")
        else:
            print(f"Collection '{self.collection_name}' already exists. Ready to use.")

    def store_chunks(self, embedded_chunks: list, source_filename: str):
        """
        Stores embedded chunks into Qdrant.
        
        Args:
            embedded_chunks: output from Embedder.embed_chunks()
            source_filename: name of the PDF (stored as metadata)
        """
        points = []
        
        for chunk_data in embedded_chunks:
            point = PointStruct(
                id=str(uuid.uuid4()),  # unique ID for each chunk
                vector={
                    "dense": chunk_data["dense_vector"],
                    "sparse": SparseVector(
                        indices=chunk_data["sparse_indices"],
                        values=chunk_data["sparse_values"],
                    ),
                },
                payload={
                    # Metadata — stored for display purposes only
                    "text": chunk_data["text"],
                    "source": source_filename,
                },
            )
            points.append(point)
        
        # Upload all chunks at once
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        
        print(f"Stored {len(points)} chunks from '{source_filename}' into Qdrant.")

    def hybrid_search(self, dense_query_vector: list, sparse_indices: list, sparse_values: list, top_k: int = 5) -> list:
        """
        Performs hybrid search: combines dense (semantic) + sparse (keyword) search.
        Returns the most relevant text chunks.
        
        Args:
            dense_query_vector: dense embedding of the user's question
            sparse_indices: sparse embedding indices of the question
            sparse_values: sparse embedding values of the question
            top_k: how many results to return
        
        Returns:
            List of dicts with 'text' and 'source' keys
        """
        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                # First, get candidates from dense (semantic) search
                {"query": dense_query_vector, "using": "dense", "limit": top_k * 2},
                # Also get candidates from sparse (keyword) search
                {
                    "query": SparseVector(indices=sparse_indices, values=sparse_values),
                    "using": "sparse",
                    "limit": top_k * 2,
                },
            ],
            # Qdrant fuses both results using RRF (Reciprocal Rank Fusion)
            query={"fusion": "rrf"},
            limit=top_k,
            with_payload=True,
        )
        
        # Extract text and source from results
        retrieved_chunks = []
        for point in results.points:
            retrieved_chunks.append({
                "text": point.payload.get("text", ""),
                "source": point.payload.get("source", "unknown"),
            })
        
        return retrieved_chunks

    def delete_collection(self):
        """Deletes the entire collection. Useful for a fresh start."""
        self.client.delete_collection(self.collection_name)
        print(f"Collection '{self.collection_name}' deleted.")
```

---

### `src/retrieval/retriever.py` — Tie Embedder + Search Together

```python
# This is a simple wrapper that:
# 1. Takes the user's question as text
# 2. Converts it to vectors using the Embedder
# 3. Runs hybrid search using the VectorStore
# 4. Returns the top matching chunks

from src.embeddings.embedder import Embedder
from src.vectordb.vector_store import VectorStore


class Retriever:
    def __init__(self, embedder: Embedder, vector_store: VectorStore, top_k: int = 5):
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(self, question: str) -> list:
        """
        Given a user question, retrieves the most relevant text chunks.
        
        Args:
            question: the user's question as a plain string
        
        Returns:
            List of relevant chunks (each is a dict with 'text' and 'source')
        """
        # Embed the question
        dense_vec = self.embedder.get_dense_embedding(question)
        sparse_indices, sparse_values = self.embedder.get_sparse_embedding(question)
        
        # Search Qdrant
        results = self.vector_store.hybrid_search(
            dense_query_vector=dense_vec,
            sparse_indices=sparse_indices,
            sparse_values=sparse_values,
            top_k=self.top_k,
        )
        
        return results
```

---

### `src/prompts/prompt_templates.py` — Prompt for the LLM

```python
# This file contains the prompt template we send to the LLM.
# The template tells the LLM: "here is some context from documents,
# now answer the user's question based ONLY on that context."


def build_rag_prompt(context_chunks: list, user_question: str) -> str:
    """
    Builds the final prompt that we send to the LLM.
    
    Args:
        context_chunks: list of dicts with 'text' and 'source'
        user_question: the user's question
    
    Returns:
        A formatted prompt string
    """
    # Combine all retrieved chunks into one context block
    context_text = ""
    for i, chunk in enumerate(context_chunks):
        context_text += f"\n--- Chunk {i+1} (from: {chunk['source']}) ---\n"
        context_text += chunk["text"]
        context_text += "\n"
    
    prompt = f"""You are a helpful assistant. Answer the user's question based ONLY on the context provided below.

If the answer is not in the context, say "I could not find this information in the uploaded documents."

Do not make up information.

CONTEXT:
{context_text}

USER QUESTION:
{user_question}

ANSWER:"""
    
    return prompt
```

---

### `src/llm/llm_client.py` — Groq LLM

```python
# This file handles talking to the Groq API.
# Groq gives us free, very fast access to LLaMA 3.3 70B.

import os
from groq import Groq
from dotenv import load_dotenv

# Load the GROQ_API_KEY from .env file
load_dotenv()


class LLMClient:
    def __init__(self, model: str = "llama-3.3-70b-versatile", max_tokens: int = 1024):
        self.model = model
        self.max_tokens = max_tokens
        
        # Initialize Groq client (reads GROQ_API_KEY from environment automatically)
        self.client = Groq()

    def get_answer(self, prompt: str) -> str:
        """
        Sends the prompt to Groq and returns the LLM's answer.
        
        Args:
            prompt: the full prompt string (with context + question)
        
        Returns:
            The LLM's response as a string
        """
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0.2,          # low temperature = more factual, less creative
            max_completion_tokens=self.max_tokens,
            top_p=1,
            stream=False,             # we get the full response at once (simpler)
            stop=None,
        )
        
        answer = completion.choices[0].message.content
        return answer
```

---

### `main.py` — Streamlit UI (Entry Point)

```python
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

st.set_page_config(page_title="RAG PDF Chat", page_icon="📄", layout="wide")

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
```

---

## Step 4 — Create Empty `__init__.py` Files

Every folder inside `src/` needs an empty `__init__.py` file so Python treats it as a module:

```
src/__init__.py              ← empty
src/ingestion/__init__.py    ← empty
src/chunking/__init__.py     ← empty
src/embeddings/__init__.py   ← empty
src/vectordb/__init__.py     ← empty
src/retrieval/__init__.py    ← empty
src/llm/__init__.py          ← empty
src/prompts/__init__.py      ← empty
```

Just create these as empty files. They do not need any code inside.

---

## Step 5 — Create the Logs Folder

```bash
mkdir -p logs
touch logs/app.log
```

---

## Step 6 — Run the App

```bash
streamlit run main.py
```

The app will open in your browser at `http://localhost:8501`

---

## How It Works — Step by Step

```
User uploads PDF
      ↓
loader.py extracts text from PDF
      ↓
chunker.py splits text into 500-char chunks
      ↓
embedder.py creates two vectors per chunk:
  • Dense vector  (BAAI/bge-small-en-v1.5) → captures meaning
  • Sparse vector (SPLADE)                  → captures keywords
      ↓
vector_store.py saves chunks + vectors to Qdrant (stored on disk locally)
      ↓
User types a question
      ↓
retriever.py embeds the question (same two vectors)
      ↓
Qdrant hybrid search → fuses dense + sparse results → top 5 most relevant chunks
      ↓
prompt_templates.py builds: "Here is context... Answer this question: ..."
      ↓
llm_client.py sends prompt to Groq → LLaMA 3.3 70B answers
      ↓
Streamlit displays the answer + shows which sources were used
```

---

## Key Notes

**Qdrant local mode is like SQLite** — you do not need Docker or any server. The data is saved in a folder called `qdrant_storage/` on your machine. Just point to it with `QdrantClient(path="./qdrant_storage")`.

**Hybrid search = semantic + keyword** — the dense model (BGE) finds chunks that are *conceptually similar* to your question. The sparse model (SPLADE) finds chunks that *share keywords* with your question. Qdrant fuses both results using RRF (Reciprocal Rank Fusion) to give you the best of both worlds.

**Why Python over Go for FastEmbed?** — Go is fast for networking but ONNX model inference is much slower in Go than in Python. Always use Python for the embedding/search pipeline.

**Groq is free and fast** — LLaMA 3.3 70B on Groq is one of the best free options for RAG. Response times are typically under 2 seconds even for 1024 tokens.

**`@st.cache_resource`** — this decorator makes sure the embedding models and Qdrant connection are loaded only once, not on every user click. Without it, the app would re-download models every time.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `GROQ_API_KEY not found` | Make sure `.env` exists and has your key |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again |
| Empty results from search | Make sure you clicked "Process PDFs" before asking questions |
| Scanned PDF returns no text | Scanned PDFs need OCR — PyPDF2 only reads text-based PDFs |
| Qdrant collection error | Click "Clear Database" in the sidebar and re-upload PDFs |

---

## What to Build Next (Future Improvements)

- Add a `data/` folder and a script to ingest PDFs from it automatically (batch ingestion)
- Add per-user collections so each user has their own document space
- Add a reranker model (FastEmbed cross-encoder) for even better results
- Add streaming output from Groq so answers appear word-by-word
- Add logging to `logs/app.log` using Python's `logging` module
- Switch to Docker for Qdrant when going to production for better performance

---

*Built with Qdrant local · FastEmbed · SPLADE · BAAI/bge-small-en-v1.5 · Groq LLaMA 3.3 70B · Streamlit*
