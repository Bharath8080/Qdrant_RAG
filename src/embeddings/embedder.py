# This file creates two types of vectors for every text chunk:
# 1. Dense vector  → captures MEANING  (using BAAI/bge-small-en-v1.5)
# 2. Sparse vector → captures KEYWORDS (using BM25)

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
