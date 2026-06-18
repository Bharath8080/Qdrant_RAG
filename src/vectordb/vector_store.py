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
    Prefetch,
    FusionQuery,
    Fusion,
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
                Prefetch(query=dense_query_vector, using="dense", limit=top_k * 2),
                # Also get candidates from sparse (keyword) search
                Prefetch(
                    query=SparseVector(indices=sparse_indices, values=sparse_values),
                    using="sparse",
                    limit=top_k * 2,
                ),
            ],
            # Qdrant fuses both results using RRF (Reciprocal Rank Fusion)
            query=FusionQuery(fusion=Fusion.RRF),
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
