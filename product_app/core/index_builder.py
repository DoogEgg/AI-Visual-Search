from __future__ import annotations

from pathlib import Path
from typing import Callable

from .clip_encoder import CLIPEncoder
from .image_scanner import record_paths, scan_images
from .vector_store import VectorStore


def build_photo_index(
    photo_root: Path,
    encoder: CLIPEncoder,
    embeddings_path: Path,
    metadata_path: Path,
    batch_size: int = 16,
    progress: Callable[[str, float], None] | None = None,
) -> tuple[VectorStore, dict]:
    if batch_size < 1:
        raise ValueError("Batch size must be positive.")
    def scan_progress(done: int, total: int) -> None:
        if progress:
            progress("正在检查图片、排除损坏文件和重复图片", 0.25 * done / max(total, 1))

    records, rejected = scan_images(photo_root, progress=scan_progress)
    if not records:
        raise ValueError("所选文件夹中没有可建立索引的有效图片。")

    paths = record_paths(records)
    embeddings_parts = []
    total = len(paths)
    for start in range(0, total, batch_size):
        batch = paths[start : start + batch_size]
        embeddings_parts.append(encoder.encode_images(batch, batch_size=batch_size))
        if progress:
            completed = min(start + len(batch), total)
            progress("正在提取CLIP图片语义向量", 0.25 + 0.7 * completed / total)

    import numpy as np

    embeddings = np.vstack(embeddings_parts).astype("float32")
    metadata = [record.to_dict() for record in records]
    store = VectorStore(embeddings, metadata, model_name=encoder.model_name)
    store.save(embeddings_path, metadata_path)

    if progress:
        progress("图片索引已保存", 1.0)

    stats = {
        "indexed": len(records),
        "rejected": len(rejected),
        "rejected_items": rejected,
        "backend": store.backend,
    }
    return store, stats
