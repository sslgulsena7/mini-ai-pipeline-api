"""
RAG Modülü
----------
Recursive chunking + SentenceTransformer embedding + FAISS arama.
"""

import re
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class RagResult:
    question: str
    answer: str
    source_chunks: List[str]
    scores: List[float]
    metrics: Dict[str, Any] = field(default_factory=dict)


class RecursiveChunker:
    """Metni örtüşen parçalara böler."""

    def __init__(self, chunk_size: int = 300, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, text: str) -> List[str]:
        words = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk = " ".join(words[i:i + self.chunk_size])
            chunks.append(chunk)
            i += self.chunk_size - self.overlap
        return [c for c in chunks if len(c.strip()) > 20]


class RagModule:
    """
    Retrieval-Augmented Generation modülü.

    Kullanım:
        rag = RagModule()
        rag.build_index(long_text)
        result = rag.query("Türkiye'nin başkenti neresidir?")
        print(result.answer)
    """

    EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self, chunk_size: int = 300, overlap: int = 50, top_k: int = 3):
        self.chunker = RecursiveChunker(chunk_size, overlap)
        self.top_k = top_k
        self._encoder = None
        self._index = None
        self._chunks: List[str] = []

    def _load_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.EMBED_MODEL)

    def build_index(self, text: str):
        """Metni parçalar, vektörleştirir ve FAISS indeksine yazar."""
        import faiss
        self._load_encoder()
        self._chunks = self.chunker.split(text)
        embeddings = self._encoder.encode(
            self._chunks, batch_size=32, show_progress_bar=False
        ).astype("float32")
        faiss.normalize_L2(embeddings)
        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)

    def query(self, question: str) -> RagResult:
        """Soruya en ilgili parçaları getirir ve cevap üretir."""
        if self._index is None or not self._chunks:
            raise RuntimeError("Önce build_index() çağrılmalı.")
        import faiss
        self._load_encoder()
        q_vec = self._encoder.encode([question]).astype("float32")
        faiss.normalize_L2(q_vec)
        scores, indices = self._index.search(q_vec, self.top_k)
        retrieved = [self._chunks[i] for i in indices[0] if i < len(self._chunks)]
        score_list = scores[0].tolist()
        answer = retrieved[0] if retrieved else "Cevap bulunamadı."
        return RagResult(
            question=question,
            answer=answer,
            source_chunks=retrieved,
            scores=score_list
        )
