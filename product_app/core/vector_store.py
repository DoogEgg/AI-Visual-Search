from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np


class VectorStore:
    """Persistent image-vector store with optional FAISS acceleration."""

    def __init__(
        self,
        embeddings: np.ndarray,
        metadata: Sequence[dict],
        model_name: str | None = None,
    ) -> None:
        embeddings = np.asarray(embeddings, dtype="float32")
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a two-dimensional matrix.")
        if len(embeddings) != len(metadata):
            raise ValueError("Embedding and metadata counts do not match.")
        if embeddings.shape[1] < 1 or not np.isfinite(embeddings).all():
            raise ValueError("Embeddings must have a positive dimension and finite values.")

        denominator = np.linalg.norm(embeddings, axis=1, keepdims=True)
        if np.any(denominator <= 1e-12):
            raise ValueError("Image embeddings cannot be zero vectors.")
        self.embeddings = embeddings / np.clip(denominator, 1e-12, None)
        self.metadata = list(metadata)
        self.model_name = model_name
        self.backend = "NumPy"
        self._index = None

        try:
            import faiss

            self._index = faiss.IndexFlatIP(self.embeddings.shape[1])
            self._index.add(self.embeddings)
            self.backend = "FAISS"
        except ImportError:
            self._index = None

    def __len__(self) -> int:
        return len(self.metadata)

    def search(self, query: np.ndarray, top_k: int = 12) -> list[dict]:
        if not len(self):
            return []
        query = np.array(query, dtype="float32", copy=True).reshape(1, -1)
        if query.shape[1] != self.embeddings.shape[1]:
            raise ValueError("Query vector dimension does not match the image index.")
        if not np.isfinite(query).all() or np.linalg.norm(query) <= 1e-12:
            raise ValueError("Query must be a finite, nonzero vector.")
        query /= np.clip(np.linalg.norm(query, axis=1, keepdims=True), 1e-12, None)
        top_k = max(1, min(int(top_k), len(self)))

        if self._index is not None:
            scores, indices = self._index.search(query, top_k)
            score_list = scores[0].tolist()
            index_list = indices[0].tolist()
        else:
            similarities = self.embeddings @ query[0]
            index_list = np.argsort(-similarities)[:top_k].tolist()
            score_list = similarities[index_list].tolist()

        return [
            {**self.metadata[index], "score": float(score)}
            for index, score in zip(index_list, score_list)
            if index >= 0
        ]

    def save(self, embeddings_path: Path, metadata_path: Path) -> None:
        embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(embeddings_path, self.embeddings)
        metadata_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "model_name": self.model_name,
                    "dimension": self.embeddings.shape[1],
                    "images": self.metadata,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, embeddings_path: Path, metadata_path: Path) -> "VectorStore":
        if not embeddings_path.exists() or not metadata_path.exists():
            raise FileNotFoundError("No saved photo index was found.")
        embeddings = np.load(embeddings_path, allow_pickle=False)
        manifest = json.loads(metadata_path.read_text(encoding="utf-8"))
        if isinstance(manifest, list):
            # Legacy indexes remain readable, but their encoder is unknown.
            return cls(embeddings=embeddings, metadata=manifest)
        if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
            raise ValueError("Unsupported index manifest format. Rebuild the index.")
        if embeddings.ndim != 2 or embeddings.shape[1] != manifest.get("dimension"):
            raise ValueError("Index dimension does not match the saved manifest.")
        return cls(
            embeddings=embeddings,
            metadata=manifest["images"],
            model_name=manifest.get("model_name"),
        )

    def validate_model(self, model_name: str) -> None:
        if self.model_name is None:
            raise ValueError("旧索引没有模型信息，请重新建立索引后再检索。")
        if self.model_name != model_name:
            raise ValueError(
                f"索引用 {self.model_name} 建立；当前选择 {model_name}。"
                "请切回原模型，或用新模型重新建立索引。"
            )
