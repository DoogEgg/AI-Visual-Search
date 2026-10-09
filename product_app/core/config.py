from __future__ import annotations

import os
from pathlib import Path


os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRODUCT_DATA_DIR = Path(
    os.environ.get("IMAGE_SEARCH_DATA_DIR", str(PROJECT_ROOT / "product_data"))
).expanduser().resolve()
INDEX_DIR = PRODUCT_DATA_DIR / "vector_index"
EMBEDDINGS_PATH = INDEX_DIR / "image_embeddings.npy"
METADATA_PATH = INDEX_DIR / "image_metadata.json"

DEFAULT_MODEL_NAME = "openai/clip-vit-base-patch32"
SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


def ensure_product_directories() -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
