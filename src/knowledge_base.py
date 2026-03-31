from __future__ import annotations

import json
from pathlib import Path
from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class KnowledgeBase:
    def __init__(
        self,
        data_path: str = "data/knowledge_base.json",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.data_path = Path(data_path)
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.documents = self._load_documents()
        self.document_count = len(self.documents)
        self._embeddings = self._embed_documents()
        self.index = self._build_index()

    def _load_documents(self) -> List[dict]:
        with self.data_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _embed_documents(self) -> np.ndarray:
        texts = [doc["content"] for doc in self.documents]
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.astype("float32")

    def _build_index(self) -> faiss.Index:
        dim = self._embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(self._embeddings)
        return index

    def search(self, query: str, top_k: int = 3) -> List[dict]:
        query_vector = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        scores, indices = self.index.search(query_vector, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            doc = dict(self.documents[idx])
            doc["score"] = float(score)
            results.append(doc)
        return results
