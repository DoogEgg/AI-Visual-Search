from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from product_app.core.image_scanner import scan_images
from product_app.core.vector_store import VectorStore


class ImageScannerTests(unittest.TestCase):
    def test_duplicate_images_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            image = Image.new("RGB", (12, 8), color="red")
            image.save(root / "first.png")
            image.save(root / "second.png")

            records, rejected = scan_images(root)

            self.assertEqual(len(records), 1)
            self.assertEqual(len(rejected), 1)
            self.assertEqual(rejected[0]["reason"], "duplicate")


class VectorStoreTests(unittest.TestCase):
    def test_cosine_search_returns_the_closest_item(self) -> None:
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
        metadata = [{"name": "red"}, {"name": "blue"}]
        store = VectorStore(embeddings, metadata)

        result = store.search(np.array([0.9, 0.1], dtype="float32"), top_k=1)

        self.assertEqual(result[0]["name"], "red")
        self.assertGreater(result[0]["score"], 0.9)

    def test_store_round_trip(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            embeddings_path = root / "vectors.npy"
            metadata_path = root / "metadata.json"
            store = VectorStore(
                np.array([[1.0, 0.0]], dtype="float32"),
                [{"name": "sample"}],
            )
            store.save(embeddings_path, metadata_path)

            loaded = VectorStore.load(embeddings_path, metadata_path)

            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded.metadata[0]["name"], "sample")


if __name__ == "__main__":
    unittest.main()
