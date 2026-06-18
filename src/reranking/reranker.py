from fastembed.rerank.cross_encoder import TextCrossEncoder
from functools import lru_cache


class Reranker:
    """
    Cross-encoder reranker using fastembed's TextCrossEncoder.
    Scores each (query, passage) pair jointly — much more accurate than
    the initial bi-encoder retrieval scores.
    """

    def __init__(self, model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2"):
        print(f"Loading reranker model: {model_name}...")
        self._model = TextCrossEncoder(model_name=model_name)
        print("Reranker loaded!")

    def rerank(self, query: str, chunks: list, top_k: int = 3) -> list:
        """
        Re-score and re-sort chunks by relevance to query.

        Args:
            query:  The user's question.
            chunks: List of dicts with at least a 'text' key (from retriever).
            top_k:  How many top chunks to return after reranking.

        Returns:
            Top-k chunks sorted by rerank score descending,
            each dict gets an extra 'rerank_score' field.
        """
        if not chunks:
            return []

        docs = [c["text"] for c in chunks]
        scores = list(self._model.rerank(query, docs))

        # Attach score to each chunk and sort descending
        ranked = sorted(
            [dict(chunk, rerank_score=float(score)) for chunk, score in zip(chunks, scores)],
            key=lambda x: x["rerank_score"],
            reverse=True,
        )
        return ranked[:top_k]
