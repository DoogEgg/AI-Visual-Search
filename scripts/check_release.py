"""Check Git's actual release list without opening any private photo library."""
from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TOP_LEVEL = {
    ".gitignore", "README.md", "requirements.txt", "requirements-core.txt",
    "requirements-product.txt", "run_product_app.py",
    ".streamlit/config.toml", ".github/workflows/app-checks.yml",
}
DEMO_NAMES = {"red_car.png", "blue_car.png", "forest.png", "house.png", "sunset.png", "fruit.png"}
SCREENSHOTS = {"import.jpg", "text-search.jpg", "insights.jpg", "status.jpg"}


def allowed_file(name: str) -> bool:
    path = Path(name)
    parts = path.parts
    if name in TOP_LEVEL:
        return True
    if parts and parts[0] in {"product_app", "tests", "scripts"}:
        return path.suffix == ".py" and "__pycache__" not in parts
    if len(parts) == 2 and parts[0] == "docs":
        return path.suffix == ".md"
    if len(parts) == 3 and parts[:2] == ("examples", "images"):
        return path.name in DEMO_NAMES
    if len(parts) == 3 and parts[:2] == ("docs", "screenshots"):
        return path.name in SCREENSHOTS
    if name == "examples/README.md":
        return True
    return False


def check_names(names: list[str]) -> list[str]:
    return [name for name in names if not allowed_file(name)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-git", action="store_true")
    args = parser.parse_args()
    completed = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True)
    if completed.returncode:
        raise SystemExit("No Git release list available. Initialise Git and stage reviewed files first.")
    names = [name for name in completed.stdout.decode("utf-8").split("\0") if name]
    if not names:
        raise SystemExit("Git release list is empty.")
    disallowed = check_names(names)
    if disallowed:
        raise SystemExit(f"Unexpected files in release: {disallowed}")
    secrets = re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9]{24,})")
    private_absolute_paths = re.compile(r"[A-Za-z]:[\\/](?:Users[\\/]|文件[\\/]|相册[\\/])")
    for name in names:
        if Path(name).suffix.lower() in {".png", ".jpg"}:
            continue
        content = (ROOT / name).read_text(encoding="utf-8")
        if secrets.search(content):
            raise SystemExit(f"Possible credential in {name}; do not publish.")
        if private_absolute_paths.search(content):
            raise SystemExit(f"Possible private absolute path in {name}; do not publish.")
    from generate_demo_images import generate_demo_images
    with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
        generated = Path(directory)
        generate_demo_images(generated)
        for name in names:
            if name.startswith("examples/images/"):
                with Image.open(ROOT / name) as actual, Image.open(generated / Path(name).name) as expected:
                    if actual.mode != expected.mode or actual.size != expected.size or actual.tobytes() != expected.tobytes():
                        raise SystemExit(f"Demo asset differs from the synthetic generator: {name}")
    print(f"PASS: {len(names)} reviewed files; no private-data paths, weights, index files or obvious credentials.")
    print("Screenshots still require visual review; .gitignore is not a substitute for this audit.")


if __name__ == "__main__":
    main()
