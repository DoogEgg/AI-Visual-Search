"""Generate original, non-personal illustrations for a reproducible demo."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DEMO_NAMES = (
    "red_car.png", "blue_car.png", "forest.png",
    "house.png", "sunset.png", "fruit.png",
)


def generate_demo_images(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in DEMO_NAMES:
        image = Image.new("RGB", (480, 320), "#eff6ff")
        draw = ImageDraw.Draw(image)
        if "car" in name:
            draw.rectangle((0, 235, 480, 320), fill="#94a3b8")
            colour = "#dc2626" if name.startswith("red") else "#2563eb"
            draw.rounded_rectangle((60, 150, 420, 240), radius=24, fill=colour)
            draw.polygon([(140, 150), (180, 85), (310, 85), (360, 150)], fill=colour)
            draw.polygon([(180, 100), (235, 100), (235, 140), (155, 140)], fill="#bfdbfe")
            draw.polygon([(250, 100), (302, 100), (335, 140), (250, 140)], fill="#bfdbfe")
            for x in (135, 345):
                draw.ellipse((x - 30, 210, x + 30, 270), fill="#1e293b")
                draw.ellipse((x - 14, 226, x + 14, 254), fill="#cbd5e1")
        elif name == "forest.png":
            draw.rectangle((0, 230, 480, 320), fill="#65a30d")
            for x, y in ((75, 55), (185, 25), (295, 50), (405, 30)):
                draw.rectangle((x - 10, y + 80, x + 10, 265), fill="#92400e")
                draw.polygon([(x, y), (x - 60, y + 145), (x + 60, y + 145)], fill="#15803d")
        elif name == "house.png":
            draw.rectangle((0, 255, 480, 320), fill="#86efac")
            draw.rectangle((110, 135, 370, 265), fill="#fef3c7")
            draw.polygon([(85, 140), (240, 40), (395, 140)], fill="#b91c1c")
            draw.rectangle((215, 190, 265, 265), fill="#92400e")
            for x in (140, 295):
                draw.rectangle((x, 165, x + 45, 210), fill="#60a5fa")
        elif name == "sunset.png":
            draw.rectangle((0, 0, 480, 190), fill="#fb923c")
            draw.ellipse((175, 70, 305, 200), fill="#fde047")
            draw.rectangle((0, 180, 480, 320), fill="#0284c7")
            for y in (200, 225, 250, 275):
                draw.line((185 - (y - 180), y, 295 + (y - 180), y), fill="#fcd34d", width=4)
        else:
            draw.ellipse((50, 210, 430, 275), fill="#cbd5e1")
            draw.ellipse((90, 105, 230, 245), fill="#dc2626")
            draw.line((160, 85, 160, 115), fill="#78350f", width=10)
            draw.ellipse((245, 105, 385, 245), fill="#f97316")
            draw.ellipse((162, 85, 205, 110), fill="#16a34a")
        image.save(destination / name)
        image.close()


if __name__ == "__main__":
    generate_demo_images(ROOT / "examples" / "images")
    print(f"Generated {len(DEMO_NAMES)} synthetic demo illustrations.")
