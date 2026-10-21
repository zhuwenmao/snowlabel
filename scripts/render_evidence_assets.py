"""Render public bitmap assets for the v0.1 evidence package."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "evidence" / "v0.1" / "architecture.png"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "segoeuib.ttf" if bold else "segoeui.ttf"
    path = Path("C:/Windows/Fonts") / name
    return ImageFont.truetype(str(path), size=size)


def box(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    title: str,
    lines: list[str],
    accent: str,
) -> None:
    x0, y0, x1, y1 = xy
    draw.rounded_rectangle(xy, radius=8, fill="#ffffff", outline="#d7dce5", width=2)
    draw.rectangle((x0, y0, x0 + 8, y1), fill=accent)
    draw.text((x0 + 26, y0 + 20), title, font=font(28, True), fill="#111827")
    y = y0 + 68
    for line in lines:
        draw.text((x0 + 26, y), line, font=font(20), fill="#4b5563")
        y += 31


def arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: str = "#6b7280",
    width: int = 4,
) -> None:
    draw.line((start, end), fill=color, width=width)
    x, y = end
    if abs(end[0] - start[0]) >= abs(end[1] - start[1]):
        direction = 1 if end[0] > start[0] else -1
        points = [(x, y), (x - 14 * direction, y - 9), (x - 14 * direction, y + 9)]
    else:
        direction = 1 if end[1] > start[1] else -1
        points = [(x, y), (x - 9, y - 14 * direction), (x + 9, y - 14 * direction)]
    draw.polygon(points, fill=color)


def main() -> int:
    image = Image.new("RGB", (1800, 1050), "#f5f7fa")
    draw = ImageDraw.Draw(image)
    draw.text((80, 55), "SnowLabel v0.1 Architecture", font=font(50, True), fill="#101828")
    draw.text(
        (82, 120),
        "Progressive annotation prototype: verified corrections become searchable vector anchors",
        font=font(24),
        fill="#667085",
    )

    box(
        draw,
        (80, 210, 390, 440),
        "1. Interaction",
        ["Image input", "Point / box prompt", "Everything mode"],
        "#2563eb",
    )
    box(
        draw,
        (490, 210, 800, 440),
        "2. Segmentation",
        ["SAM2.1 tiny", "Interactive mask", "Automatic proposals"],
        "#06b6d4",
    )
    box(
        draw,
        (900, 210, 1210, 440),
        "3. Features",
        ["DINOv3-L, 512 px", "Whole + 4x4 parts", "Geometry descriptor"],
        "#7c3aed",
    )
    box(
        draw,
        (1310, 210, 1720, 440),
        "4. Retrieval + Gate",
        ["Top-N whole retrieval", "Parts / geometry rerank", "Auto / suggest / silent"],
        "#f59e0b",
    )
    arrow(draw, (390, 325), (490, 325))
    arrow(draw, (800, 325), (900, 325))
    arrow(draw, (1210, 325), (1310, 325))

    box(
        draw,
        (80, 585, 390, 820),
        "Image Store",
        ["MinIO images / previews", "MariaDB metadata", "Content-addressed cache"],
        "#475467",
    )
    box(
        draw,
        (490, 585, 800, 820),
        "Verified Feedback",
        ["Accept or correct label", "Repair failed mask", "Enter commits anchor"],
        "#16a34a",
    )
    box(
        draw,
        (900, 585, 1210, 820),
        "Vector Memory",
        ["NumPy or Milvus", "User-scoped anchors", "No model retraining"],
        "#db2777",
    )
    arrow(draw, (235, 440), (235, 585), "#475467")
    arrow(draw, (800, 700), (900, 700), "#db2777")
    arrow(draw, (1055, 585), (1055, 500), "#16a34a")
    draw.line((1055, 500, 1515, 500, 1515, 440), fill="#16a34a", width=4)
    draw.polygon([(1515, 440), (1506, 454), (1524, 454)], fill="#16a34a")

    draw.rounded_rectangle((80, 910, 1720, 985), radius=8, fill="#101828")
    draw.text(
        (110, 931),
        "Current evidence boundary: strong cached reuse; industrial micro-defect localization remains the main bottleneck.",
        font=font(24, True),
        fill="#ffffff",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT, optimize=True)
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
