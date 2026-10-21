"""Built-in sample dataset registry for the public SnowLabel demo.

The real industrial datasets referenced here are often large or carry
redistribution limits. For a zero-download demo, industrial entries generate
small synthetic inspection scenes that mimic the task family; local real data
can still be plugged into ``data/datasets`` without changing the UI contract.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}


@dataclass(frozen=True)
class SampleClass:
    label: str
    folder: str


@dataclass(frozen=True)
class SampleDataset:
    slug: str
    title: str
    group: str
    source_name: str
    source_url: str
    license_note: str
    summary: str
    classes: tuple[SampleClass, ...]
    kind: str
    source_dir: str
    per_class: int = 5
    style: str = ""

    def public_dict(self, project_root: Path) -> dict:
        available = self.kind == "generated" or bool(_collect_images(project_root, self))
        return {
            "slug": self.slug,
            "title": self.title,
            "group": self.group,
            "source_name": self.source_name,
            "source_url": self.source_url,
            "license_note": self.license_note,
            "summary": self.summary,
            "classes": [c.label for c in self.classes],
            "per_class": self.per_class,
            "available": available,
            "image_count": _expected_count(project_root, self),
            "synthetic": self.kind == "generated",
        }


SAMPLE_DATASETS: tuple[SampleDataset, ...] = (
    SampleDataset(
        slug="industrial_mvtec_ad2",
        title="合成挑战：微小与透明缺陷",
        group="industrial",
        source_name="任务参考 MVTec AD 2",
        source_url="https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2",
        license_note="内置为合成演示图；真实 MVTec AD 2 数据需按 CC BY-NC-SA 4.0 自行下载。",
        summary="覆盖透明/重叠、暗场光照漂移、极小缺陷，比传统表面划痕更接近真实难题。",
        classes=(
            SampleClass("极小缺陷", "micro_defect"),
            SampleClass("透明重叠", "transparent_overlap"),
            SampleClass("光照漂移", "lighting_shift"),
        ),
        kind="generated",
        source_dir="data/generated_samples/industrial_mvtec_ad2/images",
        style="ad2",
    ),
    SampleDataset(
        slug="industrial_loco_logical",
        title="合成挑战：装配逻辑异常",
        group="industrial",
        source_name="任务参考 MVTec LOCO AD",
        source_url="https://www.mvtec.com/research-teaching/datasets/mvtec-loco-ad",
        license_note="内置为合成演示图；真实 MVTec LOCO AD 数据需按 CC BY-NC-SA 4.0 自行下载。",
        summary="不仅看局部纹理，还要处理缺件、错位、多余件这类结构和数量约束异常。",
        classes=(
            SampleClass("缺件槽位", "missing_slot"),
            SampleClass("错位零件", "misplaced_part"),
            SampleClass("多余零件", "extra_part"),
        ),
        kind="generated",
        source_dir="data/generated_samples/industrial_loco_logical/images",
        style="loco",
    ),
    SampleDataset(
        slug="industrial_realiad_multiview",
        title="合成挑战：多视角零件",
        group="industrial",
        source_name="任务参考 Real-IAD",
        source_url="https://realiad4ad.github.io/Real-IAD/",
        license_note="内置为合成演示图；真实 Real-IAD 数据请按原项目条款自行下载。",
        summary="模拟多视角、背景变化、材质反光和缺陷尺度变化，适合测试记忆泛化。",
        classes=(
            SampleClass("磨损", "abrasion"),
            SampleClass("变形", "deformation"),
            SampleClass("异物", "foreign_object"),
        ),
        kind="generated",
        source_dir="data/generated_samples/industrial_realiad_multiview/images",
        style="realiad",
    ),
    SampleDataset(
        slug="industrial_3d_geometry",
        title="合成挑战：几何表面缺陷",
        group="industrial",
        source_name="任务参考 MVTec 3D-AD",
        source_url="https://www.mvtec.com/research-teaching/datasets/mvtec-3d-ad",
        license_note="内置为合成演示图；真实 MVTec 3D-AD 数据需按 CC BY-NC-SA 4.0 自行下载。",
        summary="把深度/法线图里的凹陷、凸起、孔洞转成 RGB 演示图，强调几何异常。",
        classes=(
            SampleClass("凹陷", "dent"),
            SampleClass("凸起", "bump"),
            SampleClass("孔洞", "hole"),
        ),
        kind="generated",
        source_dir="data/generated_samples/industrial_3d_geometry/images",
        style="geometry3d",
    ),
    SampleDataset(
        slug="industrial_hss_iad",
        title="合成挑战：同类异构零件",
        group="industrial",
        source_name="任务参考 HSS-IAD",
        source_url="https://github.com/Qiqigeww/HSS-IAD-Dataset",
        license_note="内置为合成演示图；真实 HSS-IAD 数据请按原项目条款自行下载。",
        summary="同一种金属件存在结构差异、加工纹和油污干扰，缺陷更细微、更贴近产线。",
        classes=(
            SampleClass("微裂", "micro_crack"),
            SampleClass("毛刺", "burr"),
            SampleClass("缺口", "notch"),
        ),
        kind="generated",
        source_dir="data/generated_samples/industrial_hss_iad/images",
        style="hss",
    ),
    SampleDataset(
        slug="common_coco_vehicles",
        title="常识物体：车辆",
        group="common",
        source_name="COCO common objects",
        source_url="https://cocodataset.org/",
        license_note="复用仓库内 demo_images 小样本；用于常识物体体验。",
        summary="汽车、卡车、公交车是普通用户能立即理解的类别，用来验证基础体验。",
        classes=(
            SampleClass("汽车", "car"),
            SampleClass("卡车", "truck"),
            SampleClass("公交车", "bus"),
        ),
        kind="coco_manifest",
        source_dir="demo_images",
        per_class=6,
    ),
    SampleDataset(
        slug="common_cub_birds",
        title="鸟类细分类",
        group="common",
        source_name="CUB-200 local subset",
        source_url="https://www.vision.caltech.edu/datasets/cub_200_2011/",
        license_note="仅当本机 data/datasets/cub3_demo 存在时可用；不随公开仓库分发。",
        summary="保留原来的 3 类鸟示例，用于演示细粒度类别记忆。",
        classes=(
            SampleClass("赫氏鸥", "061.Heermann_Gull"),
            SampleClass("裏海燕鸥", "143.Caspian_Tern"),
            SampleClass("仙人掌鹪鹩", "194.Cactus_Wren"),
        ),
        kind="class_dirs",
        source_dir="data/datasets/cub3_demo/images",
    ),
)


def list_sample_datasets(project_root: Path) -> list[dict]:
    return [ds.public_dict(project_root) for ds in SAMPLE_DATASETS]


def default_sample_slug(project_root: Path) -> str:
    for slug in ("industrial_mvtec_ad2", "common_coco_vehicles"):
        ds = get_sample_dataset(slug)
        if ds.kind == "generated" or _collect_images(project_root, ds):
            return slug
    return SAMPLE_DATASETS[0].slug


def get_sample_dataset(slug: str) -> SampleDataset:
    for ds in SAMPLE_DATASETS:
        if ds.slug == slug:
            return ds
    raise KeyError(slug)


def sample_images_by_class(project_root: Path, slug: str) -> tuple[SampleDataset, dict[str, list[Path]]]:
    ds = get_sample_dataset(slug)
    if ds.kind == "generated":
        _ensure_generated(project_root, ds)
    groups = _collect_images(project_root, ds)
    if not any(groups.values()):
        raise FileNotFoundError(slug)
    return ds, groups


def sample_preview_paths(project_root: Path, slug: str, limit: int = 3) -> list[Path]:
    """Return representative preview images, preferably one per class."""
    _ds, groups = sample_images_by_class(project_root, slug)
    previews: list[Path] = []
    for paths in groups.values():
        if paths:
            previews.append(paths[0])
        if len(previews) >= limit:
            return previews
    for paths in groups.values():
        for path in paths[1:]:
            previews.append(path)
            if len(previews) >= limit:
                return previews
    return previews


def _expected_count(project_root: Path, ds: SampleDataset) -> int:
    if ds.kind == "generated":
        return len(ds.classes) * ds.per_class
    return sum(len(v) for v in _collect_images(project_root, ds).values())


def _collect_images(project_root: Path, ds: SampleDataset) -> dict[str, list[Path]]:
    root = project_root / ds.source_dir
    groups: dict[str, list[Path]] = {c.label: [] for c in ds.classes}
    if ds.kind in ("generated", "class_dirs"):
        if not root.is_dir():
            return groups
        for cls in ds.classes:
            d = root / cls.folder
            if d.is_dir():
                groups[cls.label] = sorted(
                    p for p in d.iterdir()
                    if p.is_file() and p.suffix.lower() in IMG_EXT
                )
        return groups
    if ds.kind == "coco_manifest":
        manifest = root / "demo_manifest.json"
        if not manifest.is_file():
            return groups
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception:
            return groups
        label_by_folder = {c.folder: c.label for c in ds.classes}
        for row in data.get("order", []):
            label = label_by_folder.get(str(row.get("cls", "")))
            path = root / str(row.get("file", ""))
            if label and path.is_file() and path.suffix.lower() in IMG_EXT:
                groups[label].append(path)
        return groups
    return groups


def _ensure_generated(project_root: Path, ds: SampleDataset) -> None:
    root = project_root / ds.source_dir
    root.mkdir(parents=True, exist_ok=True)
    for class_idx, cls in enumerate(ds.classes):
        out_dir = root / cls.folder
        out_dir.mkdir(parents=True, exist_ok=True)
        existing = [
            p for p in out_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMG_EXT
        ]
        if len(existing) >= ds.per_class:
            continue
        for image_idx in range(ds.per_class):
            out = out_dir / f"{ds.slug}_{cls.folder}_{image_idx + 1:02d}.jpg"
            if out.exists():
                continue
            rng = _rng(ds.slug, cls.folder, image_idx)
            img = _draw_sample(ds.style, class_idx, image_idx, rng)
            img.save(out, "JPEG", quality=88, optimize=True)


def _rng(*parts: object) -> random.Random:
    key = ":".join(map(str, parts)).encode("utf-8")
    seed = int(hashlib.sha256(key).hexdigest()[:16], 16)
    return random.Random(seed)


def _draw_sample(style: str, class_idx: int, image_idx: int, rng: random.Random) -> Image.Image:
    if style == "ad2":
        return _draw_ad2(class_idx, image_idx, rng)
    if style == "loco":
        return _draw_loco(class_idx, image_idx, rng)
    if style == "realiad":
        return _draw_realiad(class_idx, image_idx, rng)
    if style == "geometry3d":
        return _draw_geometry3d(class_idx, image_idx, rng)
    if style == "hss":
        return _draw_hss(class_idx, image_idx, rng)
    if style == "steel":
        return _draw_steel(class_idx, image_idx, rng)
    if style == "pcb":
        return _draw_pcb(class_idx, image_idx, rng)
    if style == "texture":
        return _draw_texture(class_idx, image_idx, rng)
    return _draw_surface(class_idx, image_idx, rng)


def _base(size=(720, 480), color=(112, 116, 120)) -> Image.Image:
    return Image.new("RGB", size, color)


def _add_noise(img: Image.Image, rng: random.Random, strength: int = 18) -> Image.Image:
    px = img.load()
    w, h = img.size
    for y in range(0, h, 2):
        for x in range(0, w, 2):
            delta = rng.randint(-strength, strength)
            r, g, b = px[x, y]
            c = tuple(max(0, min(255, v + delta)) for v in (r, g, b))
            px[x, y] = c
            if x + 1 < w:
                px[x + 1, y] = c
            if y + 1 < h:
                px[x, y + 1] = c
            if x + 1 < w and y + 1 < h:
                px[x + 1, y + 1] = c
    return img.filter(ImageFilter.GaussianBlur(0.35))


def _poly(cx: int, cy: int, r: int, n: int, rot: float = 0.0) -> list[tuple[float, float]]:
    return [
        (
            cx + r * math.cos(rot + 2 * math.pi * i / n),
            cy + r * math.sin(rot + 2 * math.pi * i / n),
        )
        for i in range(n)
    ]


def _draw_ad2(class_idx: int, image_idx: int, rng: random.Random) -> Image.Image:
    img = _add_noise(_base(size=(760, 520), color=(88, 92, 96)), rng, 13)
    d = ImageDraw.Draw(img, "RGBA")
    if class_idx == 0:
        d.rounded_rectangle((96, 74, 664, 446), radius=22, fill=(126, 131, 135, 255), outline=(202, 205, 207, 180), width=2)
        for y in range(106, 420, 34):
            d.line((118, y, 642, y + rng.randint(-2, 2)), fill=(93, 98, 101, 90), width=1)
        for x in range(122, 640, 42):
            d.line((x, 90, x + rng.randint(-5, 5), 430), fill=(164, 168, 171, 50), width=1)
        for _ in range(95):
            x, y = rng.randint(130, 630), rng.randint(105, 410)
            shade = rng.randint(82, 118)
            d.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(shade, shade, shade, 135))
        for _ in range(4):
            x, y = rng.randint(160, 610), rng.randint(120, 392)
            r = rng.randint(3, 6)
            d.ellipse((x - r, y - r, x + r, y + r), fill=(32, 34, 36, 220))
            d.ellipse((x - r - 3, y - r - 3, x + r + 3, y + r + 3), outline=(214, 218, 221, 80), width=1)
    elif class_idx == 1:
        img = _add_noise(_base(size=(760, 520), color=(52, 57, 63)), rng, 8)
        d = ImageDraw.Draw(img, "RGBA")
        for i in range(3):
            x0 = 120 + i * 102 + rng.randint(-18, 18)
            y0 = 110 + i * 32 + rng.randint(-14, 14)
            poly = [(x0, y0 + 26), (x0 + 280, y0), (x0 + 334, y0 + 208), (x0 + 42, y0 + 236)]
            d.polygon(poly, fill=(160, 205, 220, 48), outline=(225, 246, 255, 155))
            d.line((poly[0], poly[2]), fill=(245, 255, 255, 82), width=2)
        for _ in range(7):
            x, y = rng.randint(170, 610), rng.randint(140, 390)
            d.ellipse((x - 18, y - 11, x + 21, y + 12), fill=(180, 215, 226, 42), outline=(229, 246, 250, 110), width=1)
        x = 310 + rng.randint(-55, 60)
        y = 235 + rng.randint(-46, 56)
        d.line((x, y, x + 116, y + rng.randint(-28, 28)), fill=(247, 255, 255, 190), width=3)
    else:
        img = _add_noise(_base(size=(760, 520), color=(19, 23, 28)), rng, 8)
        d = ImageDraw.Draw(img, "RGBA")
        for i in range(9):
            x = -80 + i * 118 + rng.randint(-18, 18)
            d.polygon([(x, 0), (x + 74, 0), (x + 210, 520), (x + 92, 520)], fill=(180, 190, 210, 18 + i % 3 * 10))
        d.rounded_rectangle((120, 92, 640, 426), radius=30, fill=(61, 67, 75, 255), outline=(142, 151, 165, 145), width=2)
        for y in range(126, 402, 26):
            d.line((146, y, 612, y + rng.randint(-2, 2)), fill=(84, 92, 101, 100), width=2)
        x = rng.randint(260, 500)
        y = rng.randint(170, 340)
        d.ellipse((x - 46, y - 18, x + 54, y + 22), fill=(92, 96, 105, 130))
        d.arc((x - 50, y - 22, x + 58, y + 26), 188, 22, fill=(230, 236, 245, 95), width=4)
    return img


def _draw_loco(class_idx: int, image_idx: int, rng: random.Random) -> Image.Image:
    img = _add_noise(_base(size=(760, 520), color=(53, 58, 65)), rng, 10)
    d = ImageDraw.Draw(img, "RGBA")
    d.rounded_rectangle((96, 62, 664, 458), radius=32, fill=(86, 93, 103, 255), outline=(168, 176, 188, 110), width=3)
    slots: list[tuple[int, int]] = []
    for row in range(3):
        for col in range(4):
            cx, cy = 178 + col * 134, 148 + row * 112
            slots.append((cx, cy))
            d.rounded_rectangle((cx - 46, cy - 34, cx + 46, cy + 34), radius=16, fill=(42, 47, 54, 230), outline=(130, 139, 151, 105), width=2)
    target = (image_idx * 3 + class_idx * 2) % len(slots)
    for idx, (cx, cy) in enumerate(slots):
        skip = class_idx == 0 and idx == target
        if skip:
            d.rounded_rectangle((cx - 37, cy - 25, cx + 37, cy + 25), radius=13, fill=(24, 28, 33, 240), outline=(203, 212, 225, 85), width=2)
            continue
        dx = rng.randint(-4, 4)
        dy = rng.randint(-4, 4)
        if class_idx == 1 and idx == target:
            dx += rng.choice([-36, 38])
            dy += rng.choice([-24, 26])
        d.ellipse((cx - 28 + dx, cy - 28 + dy, cx + 28 + dx, cy + 28 + dy), fill=(194, 200, 207, 255), outline=(246, 249, 252, 140), width=2)
        d.ellipse((cx - 10 + dx, cy - 10 + dy, cx + 10 + dx, cy + 10 + dy), fill=(72, 78, 86, 255))
    if class_idx == 2:
        cx, cy = slots[target]
        ex = cx + rng.choice([-58, 62])
        ey = cy + rng.choice([-42, 44])
        d.ellipse((ex - 24, ey - 24, ex + 24, ey + 24), fill=(222, 199, 130, 255), outline=(255, 240, 186, 145), width=2)
        d.ellipse((ex - 8, ey - 8, ex + 8, ey + 8), fill=(89, 68, 34, 255))
    return img


def _draw_realiad(class_idx: int, image_idx: int, rng: random.Random) -> Image.Image:
    bg = (rng.randint(65, 96), rng.randint(70, 102), rng.randint(76, 110))
    img = _add_noise(_base(size=(760, 520), color=bg), rng, 16)
    d = ImageDraw.Draw(img, "RGBA")
    for i in range(7):
        x = -40 + i * 132
        shade = 28 + rng.randint(0, 20)
        d.polygon([(x, 0), (x + 82, 0), (x + 178, 520), (x + 32, 520)], fill=(shade, shade + 6, shade + 12, 24))
    cx = 380 + rng.randint(-28, 28)
    cy = 260 + rng.randint(-22, 22)
    d.ellipse((cx - 202, cy + 88, cx + 214, cy + 148), fill=(17, 20, 24, 80))
    body = [(cx - 196, cy - 92), (cx + 138, cy - 122), (cx + 202, cy + 86), (cx - 140, cy + 118)]
    d.polygon(body, fill=(143, 151, 158, 255), outline=(232, 237, 240, 145))
    d.line((body[0], body[2]), fill=(216, 222, 226, 90), width=3)
    d.line((body[1], body[3]), fill=(63, 69, 76, 120), width=3)
    for _ in range(7):
        x, y = rng.randint(cx - 145, cx + 145), rng.randint(cy - 74, cy + 74)
        d.ellipse((x - 15, y - 15, x + 15, y + 15), fill=(108, 116, 124, 100), outline=(207, 212, 216, 90))
    if class_idx == 0:
        x, y = cx + rng.randint(-105, 95), cy + rng.randint(-46, 56)
        for i in range(8):
            d.line((x - 42 + i * 10, y - 19 + rng.randint(-4, 4), x + 18 + i * 8, y + 18 + rng.randint(-4, 4)), fill=(48, 51, 55, 190), width=2)
    elif class_idx == 1:
        x, y = cx + rng.randint(-94, 96), cy + rng.randint(-42, 50)
        d.ellipse((x - 54, y - 31, x + 58, y + 36), fill=(82, 90, 99, 210))
        d.arc((x - 60, y - 34, x + 60, y + 40), 200, 35, fill=(231, 237, 241, 140), width=5)
    else:
        x, y = cx + rng.randint(-112, 112), cy + rng.randint(-56, 58)
        color = rng.choice([(44, 125, 164, 255), (170, 92, 42, 255), (36, 40, 45, 255)])
        d.rounded_rectangle((x - 35, y - 18, x + 37, y + 20), radius=8, fill=color, outline=(245, 248, 250, 80), width=1)
    return img


def _draw_geometry3d(class_idx: int, image_idx: int, rng: random.Random) -> Image.Image:
    img = _base(size=(760, 520), color=(102, 112, 128))
    d = ImageDraw.Draw(img, "RGBA")
    for y in range(0, 520, 4):
        shade = 78 + int(48 * math.sin((y + image_idx * 17) / 58)) + rng.randint(-4, 4)
        d.rectangle((0, y, 760, y + 4), fill=(shade, shade + 10, shade + 22, 255))
    d.rounded_rectangle((122, 82, 638, 438), radius=44, fill=(126, 139, 156, 235), outline=(232, 239, 246, 130), width=3)
    for i in range(8):
        x = 164 + i * 62
        d.line((x, 104, x - 24, 418), fill=(87, 97, 112, 80), width=4)
    for x in range(185, 590, 128):
        for y in range(160, 380, 92):
            d.ellipse((x - 23, y - 23, x + 23, y + 23), fill=(153, 166, 181, 180), outline=(72, 82, 98, 100), width=2)
    x = rng.randint(260, 510)
    y = rng.randint(175, 350)
    if class_idx == 0:
        d.ellipse((x - 74, y - 42, x + 78, y + 45), fill=(58, 67, 82, 190))
        d.ellipse((x - 48, y - 25, x + 48, y + 28), fill=(91, 101, 117, 210))
        d.arc((x - 75, y - 44, x + 80, y + 45), 210, 38, fill=(238, 244, 249, 130), width=5)
    elif class_idx == 1:
        d.ellipse((x - 66, y - 46, x + 70, y + 48), fill=(190, 202, 214, 210))
        d.arc((x - 64, y - 46, x + 70, y + 48), 32, 210, fill=(54, 62, 76, 120), width=6)
        d.ellipse((x - 28, y - 20, x + 30, y + 22), fill=(216, 226, 235, 180))
    else:
        d.ellipse((x - 50, y - 50, x + 50, y + 50), fill=(42, 49, 59, 235), outline=(229, 235, 242, 120), width=5)
        d.ellipse((x - 25, y - 25, x + 25, y + 25), fill=(15, 18, 22, 255))
    return img.filter(ImageFilter.GaussianBlur(0.2))


def _draw_hss(class_idx: int, image_idx: int, rng: random.Random) -> Image.Image:
    img = _add_noise(_base(size=(760, 520), color=(82, 86, 90)), rng, 12)
    d = ImageDraw.Draw(img, "RGBA")
    cx, cy = 382 + rng.randint(-24, 24), 262 + rng.randint(-18, 18)
    r = 154 + rng.randint(-10, 16)
    d.ellipse((cx - r - 30, cy - r - 10, cx + r + 30, cy + r + 10), fill=(35, 38, 42, 70))
    if image_idx % 3 == 0:
        body = _poly(cx, cy, r, 8, rot=math.pi / 8)
        d.polygon(body, fill=(151, 155, 158, 255), outline=(234, 238, 240, 120))
    elif image_idx % 3 == 1:
        d.rounded_rectangle((cx - 174, cy - 116, cx + 174, cy + 116), radius=54, fill=(151, 155, 158, 255), outline=(234, 238, 240, 120), width=3)
    else:
        d.ellipse((cx - 174, cy - 122, cx + 174, cy + 122), fill=(151, 155, 158, 255), outline=(234, 238, 240, 120), width=3)
    for angle in range(0, 360, 45):
        hx = cx + int(96 * math.cos(math.radians(angle)))
        hy = cy + int(70 * math.sin(math.radians(angle)))
        d.ellipse((hx - 20, hy - 20, hx + 20, hy + 20), fill=(83, 88, 94, 230), outline=(221, 226, 231, 80))
    for _ in range(28):
        x = rng.randint(cx - 150, cx + 150)
        y = rng.randint(cy - 100, cy + 100)
        d.line((x - 28, y, x + 30, y + rng.randint(-3, 3)), fill=(108, 113, 118, 75), width=1)
    x = cx + rng.randint(-95, 95)
    y = cy + rng.randint(-62, 62)
    if class_idx == 0:
        pts = [(x, y)]
        for _ in range(5):
            x += rng.randint(18, 34)
            y += rng.randint(-18, 18)
            pts.append((x, y))
        d.line(pts, fill=(38, 40, 43, 230), width=4)
        d.line([(px, py - 2) for px, py in pts], fill=(212, 216, 218, 70), width=1)
    elif class_idx == 1:
        base_x, base_y = cx + rng.choice([-154, 154]), cy + rng.randint(-65, 65)
        for i in range(5):
            tip = (base_x + rng.choice([-1, 1]) * rng.randint(18, 34), base_y + i * 10 - 20)
            d.polygon([(base_x, base_y + i * 10 - 25), (base_x, base_y + i * 10 - 12), tip], fill=(205, 211, 215, 230))
    else:
        nx = cx + rng.choice([-125, 128])
        ny = cy + rng.randint(-54, 54)
        d.pieslice((nx - 44, ny - 44, nx + 44, ny + 44), 110, 250, fill=(65, 69, 74, 255))
        d.arc((nx - 47, ny - 47, nx + 47, ny + 47), 105, 255, fill=(236, 239, 242, 95), width=4)
    return img


def _draw_surface(class_idx: int, image_idx: int, rng: random.Random) -> Image.Image:
    img = _add_noise(_base(color=(126, 128, 130)), rng, 11)
    d = ImageDraw.Draw(img)
    for y in range(40, 460, 32):
        c = 118 + rng.randint(-10, 12)
        d.line((34, y, 686, y + rng.randint(-3, 3)), fill=(c, c, c), width=2)
    d.rounded_rectangle((86, 76, 634, 404), radius=26, outline=(190, 194, 197), width=3)
    d.rounded_rectangle((108, 98, 612, 382), radius=18, outline=(84, 86, 90), width=2)
    if class_idx == 0:
        for _ in range(3):
            x = rng.randint(180, 520)
            y = rng.randint(150, 330)
            d.line((x, y, x + rng.randint(80, 150), y + rng.randint(-32, 32)),
                   fill=(42, 45, 48), width=rng.randint(4, 7))
    elif class_idx == 1:
        x = rng.randint(260, 430)
        y = rng.randint(170, 310)
        d.ellipse((x - 52, y - 34, x + 54, y + 36), fill=(66, 68, 72))
        d.ellipse((x - 38, y - 24, x + 36, y + 22), fill=(92, 94, 98))
        d.arc((x - 52, y - 34, x + 54, y + 36), 210, 40, fill=(210, 214, 216), width=4)
    else:
        for _ in range(4):
            x = rng.randint(170, 540)
            y = rng.randint(130, 335)
            d.ellipse((x - 30, y - 18, x + 36, y + 22), fill=(118, 82, 44))
            d.ellipse((x - 19, y - 10, x + 22, y + 13), fill=(152, 103, 55))
    return img


def _draw_steel(class_idx: int, image_idx: int, rng: random.Random) -> Image.Image:
    img = _add_noise(_base(color=(146, 150, 153)), rng, 20)
    d = ImageDraw.Draw(img)
    for y in range(18, 470, 10):
        c = 132 + rng.randint(-12, 18)
        d.line((0, y, 720, y + rng.randint(-2, 2)), fill=(c, c, c), width=1)
    if class_idx == 0:
        for _ in range(5):
            x = rng.randint(120, 560)
            y = rng.randint(90, 390)
            d.ellipse((x - 22, y - 14, x + 28, y + 17), fill=(54, 57, 60))
    elif class_idx == 1:
        for _ in range(36):
            x = rng.randint(100, 620)
            y = rng.randint(80, 400)
            r = rng.randint(5, 13)
            d.ellipse((x - r, y - r, x + r, y + r), fill=(82, 84, 87))
    else:
        for _ in range(4):
            x = rng.randint(120, 560)
            y = rng.randint(105, 375)
            d.line((x, y, x + rng.randint(110, 210), y + rng.randint(-20, 20)),
                   fill=(35, 37, 39), width=rng.randint(3, 6))
    return img


def _draw_pcb(class_idx: int, image_idx: int, rng: random.Random) -> Image.Image:
    img = _add_noise(_base(color=(28, 104, 82)), rng, 9)
    d = ImageDraw.Draw(img)
    copper = (219, 177, 82)
    dark = (13, 54, 44)
    ys = [90, 150, 210, 270, 330, 390]
    for y in ys:
        d.line((60, y, 660, y + rng.randint(-8, 8)), fill=copper, width=8)
    for x in range(95, 670, 95):
        for y in ys[::2]:
            d.ellipse((x - 15, y - 15, x + 15, y + 15), fill=copper)
            d.ellipse((x - 6, y - 6, x + 6, y + 6), fill=dark)
    if class_idx == 0:
        y = ys[image_idx % len(ys)]
        x = 250 + 45 * (image_idx % 3)
        d.rectangle((x - 12, y - 16, x + 18, y + 16), fill=dark)
        d.ellipse((x - 24, y - 24, x + 30, y + 24), outline=(240, 80, 70), width=4)
    elif class_idx == 1:
        x = 300 + 25 * (image_idx % 4)
        d.line((x, 150, x + 58, 210), fill=copper, width=12)
        d.ellipse((x - 16, 136, x + 74, 224), outline=(240, 80, 70), width=4)
    else:
        x = 180 + 70 * (image_idx % 5)
        y = 210 + 30 * (image_idx % 3)
        d.ellipse((x - 26, y - 26, x + 26, y + 26), fill=copper)
        d.ellipse((x - 10, y - 10, x + 10, y + 10), fill=(18, 20, 24))
        d.ellipse((x - 30, y - 30, x + 30, y + 30), outline=(240, 80, 70), width=4)
    return img


def _draw_texture(class_idx: int, image_idx: int, rng: random.Random) -> Image.Image:
    img = _base(color=(136, 136, 136))
    d = ImageDraw.Draw(img)
    for x in range(-80, 760, 18):
        shade = 126 + rng.randint(-15, 18)
        d.line((x, 0, x + 160, 480), fill=(shade, shade, shade), width=3)
    for y in range(0, 500, 22):
        shade = 145 + rng.randint(-18, 15)
        d.line((0, y, 720, y + int(8 * math.sin(y / 24))), fill=(shade, shade, shade), width=2)
    img = img.filter(ImageFilter.GaussianBlur(0.7))
    d = ImageDraw.Draw(img)
    if class_idx == 0:
        x = rng.randint(130, 520)
        y = rng.randint(105, 360)
        d.line((x, y, x + 170, y + rng.randint(-42, 42)), fill=(55, 55, 55), width=9)
    elif class_idx == 1:
        for _ in range(4):
            x = rng.randint(160, 560)
            y = rng.randint(110, 360)
            r = rng.randint(18, 36)
            d.ellipse((x - r, y - r, x + r, y + r), fill=(58, 58, 58))
    else:
        x = rng.randint(230, 460)
        y = rng.randint(150, 320)
        d.ellipse((x - 82, y - 34, x + 82, y + 34), fill=(62, 62, 62))
    return img
