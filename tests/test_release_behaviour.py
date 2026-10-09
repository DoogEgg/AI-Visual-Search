from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from product_app.core.image_scanner import scan_images
from product_app.core.index_builder import build_photo_index
from product_app.core.vector_store import VectorStore
from scripts.check_release import check_names


class FakeEncoder:
    model_name = "test-encoder"

    def __init__(self, *args, **kwargs):
        self.batch_lengths = []

    def encode_images(self, images, batch_size=16):
        self.batch_lengths.append(len(images))
        return np.array([[1.0, 0.5] for _ in images], dtype="float32")

    def encode_texts(self, texts):
        return np.array([[1.0, 0.0] for _ in texts], dtype="float32")


class ScannerSafetyTests(unittest.TestCase):
    def test_corrupt_files_are_skipped(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            Image.new("RGB", (8, 6), "red").save(root / "valid.PNG")
            (root / "broken.jpg").write_bytes(b"not an image")
            (root / "ignored.txt").write_text("not indexed", encoding="utf-8")
            records, rejected = scan_images(root)
            self.assertEqual(len(records), 1)
            self.assertEqual(len(rejected), 1)
            self.assertEqual((records[0].width, records[0].height), (8, 6))

    def test_recursive_scan_and_progress(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            (root / "nested").mkdir()
            Image.new("RGB", (8, 6), "blue").save(root / "nested" / "sample.png")
            progress = []
            records, rejected = scan_images(root, lambda done, total: progress.append((done, total)))
            self.assertEqual(len(records), 1)
            self.assertFalse(rejected)
            self.assertEqual(progress[-1], (1, 1))

    def test_missing_folder_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            with self.assertRaises(ValueError):
                scan_images(Path(directory) / "missing")


class IndexBuilderTests(unittest.TestCase):
    def test_build_reload_records_model_and_batches(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            photos = root / "photos"
            photos.mkdir()
            for index, colour in enumerate(("red", "blue", "green")):
                Image.new("RGB", (8, 6), colour).save(photos / f"demo-{index}.png")
            encoder = FakeEncoder()
            progress = []
            vectors, manifest = root / "vectors.npy", root / "manifest.json"
            store, stats = build_photo_index(
                photos, encoder, vectors, manifest, batch_size=2,
                progress=lambda message, value: progress.append(value),
            )
            self.assertEqual(encoder.batch_lengths, [2, 1])
            self.assertEqual(stats["indexed"], 3)
            self.assertEqual(progress[-1], 1.0)
            loaded = VectorStore.load(vectors, manifest)
            self.assertEqual(loaded.model_name, encoder.model_name)
            self.assertTrue(np.allclose(store.embeddings, loaded.embeddings))

    def test_empty_folder_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                build_photo_index(root, FakeEncoder(), root / "v.npy", root / "m.json")

    def test_invalid_batch_size_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                build_photo_index(root, FakeEncoder(), root / "v.npy", root / "m.json", batch_size=0)


class VectorValidationTests(unittest.TestCase):
    def setUp(self):
        self.store = VectorStore(np.eye(2, dtype="float32"), [{"name": "one"}, {"name": "two"}], "test")

    def test_model_mismatch_is_rejected(self):
        self.store.validate_model("test")
        with self.assertRaises(ValueError):
            self.store.validate_model("different")

    def test_query_is_not_modified(self):
        query = np.array([2.0, 0.0], dtype="float32")
        self.store.search(query)
        np.testing.assert_array_equal(query, [2.0, 0.0])

    def test_invalid_queries_are_rejected(self):
        for query in ([1.0, 2.0, 3.0], [float("nan"), 0.0], [0.0, 0.0]):
            with self.subTest(query=query), self.assertRaises(ValueError):
                self.store.search(query)

    def test_invalid_vectors_are_rejected(self):
        for vectors in (np.zeros((1, 2)), np.array([[float("inf"), 0.0]]), np.zeros((1, 0))):
            with self.subTest(shape=vectors.shape), self.assertRaises(ValueError):
                VectorStore(vectors, [{}])

    def test_count_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            VectorStore(np.eye(2), [{}])

    def test_top_k_is_clamped_and_scores_descend(self):
        hits = self.store.search([1.0, 0.0], top_k=100)
        self.assertEqual(len(hits), 2)
        self.assertGreaterEqual(hits[0]["score"], hits[1]["score"])

    def test_empty_store_returns_no_hits(self):
        store = VectorStore(np.empty((0, 2), dtype="float32"), [])
        self.assertEqual(store.search([1.0, 0.0]), [])

    def test_legacy_index_requires_rebuild_for_queries(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            np.save(root / "v.npy", np.eye(2, dtype="float32"))
            (root / "m.json").write_text(json.dumps([{"name": "one"}, {"name": "two"}]), encoding="utf-8")
            loaded = VectorStore.load(root / "v.npy", root / "m.json")
            self.assertEqual(len(loaded), 2)
            with self.assertRaises(ValueError):
                loaded.validate_model("test")

    def test_manifest_dimension_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            self.store.save(root / "v.npy", root / "m.json")
            manifest = json.loads((root / "m.json").read_text(encoding="utf-8"))
            manifest["dimension"] = 999
            (root / "m.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ValueError):
                VectorStore.load(root / "v.npy", root / "m.json")


class ReleasePrivacyTests(unittest.TestCase):
    def test_private_files_are_not_allowed(self):
        files = ["product_data/vector_index/image_metadata.json", "photos/family.jpg",
                 "data/train.csv", "hf_cache/model.safetensors", "model_L.pt",
                 ".streamlit/secrets.toml", "tests/__pycache__/sample.pyc"]
        self.assertEqual(check_names(files), files)

    def test_public_code_and_generated_assets_are_allowed(self):
        self.assertFalse(check_names(["README.md", "product_app/core/vector_store.py", "examples/images/red_car.png"]))


if __name__ == "__main__":
    unittest.main()
