from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import streamlit as st
from PIL import Image
from streamlit.testing.v1 import AppTest

from product_app.core import config
from product_app.core.vector_store import VectorStore
from test_release_behaviour import FakeEncoder

APP = Path(__file__).resolve().parents[1] / "product_app" / "app.py"


class AppSmokeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.root = Path(self.directory.name)
        self.patches = [
            patch.object(config, "EMBEDDINGS_PATH", self.root / "v.npy"),
            patch.object(config, "METADATA_PATH", self.root / "m.json"),
            patch("product_app.core.clip_encoder.CLIPEncoder", FakeEncoder),
        ]
        for item in self.patches:
            item.start()
        st.cache_resource.clear()

    def tearDown(self):
        st.cache_resource.clear()
        for item in reversed(self.patches):
            item.stop()
        self.directory.cleanup()

    def app(self, with_store=False):
        if with_store:
            Image.new("RGB", (8, 6), "red").save(self.root / "demo.png")
            VectorStore(
                np.array([[1.0, 0.0]], dtype="float32"),
                [{"name": "demo.png", "path": str(self.root / "demo.png"),
                  "width": 8, "height": 6, "size_bytes": 100}],
                model_name=config.DEFAULT_MODEL_NAME,
            ).save(config.EMBEDDINGS_PATH, config.METADATA_PATH)
        app = AppTest.from_file(str(APP), default_timeout=30).run()
        self.assertFalse(app.exception)
        return app

    def test_empty_homepage_has_five_tabs(self):
        app = self.app()
        self.assertEqual(len(app.tabs), 5)
        self.assertEqual(app.title[0].value, "AI智能相册与语义检索平台")

    def test_search_without_index_warns(self):
        app = self.app()
        app.button[1].click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("请先" in warning.value for warning in app.warning))

    def test_blank_query_warns(self):
        app = self.app(with_store=True)
        app.button[1].click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("请输入" in warning.value for warning in app.warning))

    def test_text_search_renders_with_saved_index(self):
        app = self.app(with_store=True)
        next(item for item in app.text_input if item.label == "描述你想找的图片").set_value("a red car")
        app.button[1].click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("相似度" in caption.value for caption in app.caption))

    def test_label_analysis_returns_table(self):
        app = self.app(with_store=True)
        next(item for item in app.button if item.label == "分析语义标签").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.dataframe), 1)
        self.assertIn("语义相似度", app.dataframe[0].value.columns)


if __name__ == "__main__":
    unittest.main()
