from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image


class CLIPEncoder:
    """Lazy CLIP wrapper for image and text embedding generation."""

    def __init__(self, model_name: str, device: str | None = None) -> None:
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise RuntimeError(
                "CLIP dependencies are missing. Install the product requirements first."
            ) from exc

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @staticmethod
    def _normalise(array: np.ndarray) -> np.ndarray:
        denominator = np.linalg.norm(array, axis=1, keepdims=True)
        return array / np.clip(denominator, 1e-12, None)

    def encode_images(
        self,
        images: Sequence[Path | str | Image.Image],
        batch_size: int = 16,
    ) -> np.ndarray:
        if batch_size < 1:
            raise ValueError("Batch size must be positive.")
        all_embeddings: list[np.ndarray] = []

        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            opened_images: list[Image.Image] = []
            owned_images: list[Image.Image] = []
            try:
                for item in batch:
                    if isinstance(item, Image.Image):
                        image = item.convert("RGB")
                    else:
                        with Image.open(item) as source:
                            image = source.convert("RGB")
                    opened_images.append(image)
                    owned_images.append(image)

                inputs = self.processor(
                    images=opened_images,
                    return_tensors="pt",
                    padding=True,
                )
                pixel_values = inputs["pixel_values"].to(self.device)
                with self.torch.inference_mode():
                    vision_output = self.model.vision_model(pixel_values=pixel_values)
                    features = self.model.visual_projection(vision_output.pooler_output)
                all_embeddings.append(features.detach().cpu().numpy().astype("float32"))
            finally:
                for image in owned_images:
                    image.close()

        if not all_embeddings:
            return np.empty((0, self.model.config.projection_dim), dtype="float32")
        return self._normalise(np.vstack(all_embeddings).astype("float32"))

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.model.config.projection_dim), dtype="float32")
        inputs = self.processor(
            text=list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        with self.torch.inference_mode():
            text_output = self.model.text_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
            )
            features = self.model.text_projection(text_output.pooler_output)
        embeddings = features.detach().cpu().numpy().astype("float32")
        return self._normalise(embeddings)
