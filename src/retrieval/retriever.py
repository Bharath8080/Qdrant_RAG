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
