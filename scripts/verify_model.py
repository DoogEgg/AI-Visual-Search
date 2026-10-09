"""Opt-in real-model smoke test using only synthetic demo assets."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from product_app.core.config import DEFAULT_MODEL_NAME
from product_app.core.clip_encoder import CLIPEncoder
from product_app.core.index_builder import build_photo_index
from product_app.core.vector_store import VectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Use cached weights only.")
    parser.add_argument("--output", type=Path, default=ROOT / "tmp" / "model_verification.json")
    arguments = parser.parse_args()
    if arguments.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    encoder = CLIPEncoder(DEFAULT_MODEL_NAME)
    embeddings_path = ROOT / "tmp" / "demo_index" / "vectors.npy"
    metadata_path = ROOT / "tmp" / "demo_index" / "manifest.json"
    started = time.perf_counter()
    store, stats = build_photo_index(
        ROOT / "examples" / "images", encoder,
        embeddings_path, metadata_path, batch_size=3,
    )
    build_seconds = time.perf_counter() - started
    reloaded = VectorStore.load(embeddings_path, metadata_path)
    reloaded.validate_model(DEFAULT_MODEL_NAME)
    assert len(reloaded) == 6
    matched = 0
    for item in reloaded.metadata:
        query = encoder.encode_images([item["path"]])[0]
        result = reloaded.search(query, top_k=1)[0]
        assert result["name"] == item["name"], "Image self-retrieval failed."
        matched += 1
    queries = ["a red car", "a blue car", "a forest", "a house", "sunset over the sea", "fruit"]
    vectors = encoder.encode_texts(queries)
    results = []
    for query, vector in zip(queries, vectors):
        started = time.perf_counter()
        hits = reloaded.search(vector, top_k=3)
        results.append({
            "query": query,
            "top_3": [{"name": hit["name"], "cosine_similarity": round(hit["score"], 4)} for hit in hits],
            "vector_search_ms": round((time.perf_counter() - started) * 1000, 3),
        })
    report = {
        "purpose": "Synthetic-demo functional smoke test; not an accuracy benchmark.",
        "model": DEFAULT_MODEL_NAME,
        "device": encoder.device,
        "indexed": stats["indexed"], "rejected": stats["rejected"],
        "dimension": reloaded.embeddings.shape[1], "backend": reloaded.backend,
        "build_seconds": round(build_seconds, 3),
        "self_retrieval_passed": matched,
        "environment": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy", "Pillow")},
        "text_queries": results,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
