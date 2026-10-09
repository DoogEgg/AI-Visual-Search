from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

from PIL import Image, UnidentifiedImageError

from .config import SUPPORTED_IMAGE_EXTENSIONS


@dataclass(frozen=True)
class ImageRecord:
    path: str
    name: str
    extension: str
    size_bytes: int
    modified_time: float
    width: int
    height: int
    sha256: str

    def to_dict(self) -> dict:
        return asdict(self)


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def discover_image_paths(root: Path) -> list[Path]:
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Photo folder does not exist: {root}")
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )


def scan_images(
    root: Path,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[list[ImageRecord], list[dict]]:
    paths = discover_image_paths(root)
    records: list[ImageRecord] = []
    rejected: list[dict] = []
    seen_hashes: set[str] = set()

    for index, path in enumerate(paths, start=1):
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                width, height = image.size

            file_hash = _sha256(path)
            if file_hash in seen_hashes:
                rejected.append({"path": str(path), "reason": "duplicate"})
            else:
                seen_hashes.add(file_hash)
                stat = path.stat()
                records.append(
                    ImageRecord(
                        path=str(path.resolve()),
                        name=path.name,
                        extension=path.suffix.lower(),
                        size_bytes=stat.st_size,
                        modified_time=stat.st_mtime,
                        width=width,
                        height=height,
                        sha256=file_hash,
                    )
                )
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            rejected.append({"path": str(path), "reason": str(exc)})

        if progress:
            progress(index, len(paths))

    return records, rejected


def record_paths(records: Iterable[ImageRecord]) -> list[Path]:
    return [Path(record.path) for record in records]
