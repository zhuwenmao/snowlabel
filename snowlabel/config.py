"""Centralized configuration for SnowLabel.

Read from environment variables so the same code base runs locally and in
containers without code changes. Override any of these via env vars; the
defaults are aimed at a single-machine dev setup on Windows + RTX 4060.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(key: str, default: str) -> str:
    v = os.environ.get(key)
    return v if v else default


@dataclass(frozen=True)
class SnowConfig:
    """All defaults are evaluated **per instantiation** (default_factory) so
    that re-reading env vars after they change works as expected — important
    for scripts that run multiple experiments back-to-back.
    """
    # ------- File system -------
    project_root: Path = field(
        default_factory=lambda: Path(_env(
            "SNOWLABEL_ROOT",
            str(Path(__file__).resolve().parent.parent),
        )),
    )
    memory_dir: Path = field(init=False)
    cache_dir: Path = field(init=False)
    dataset_dir: Path = field(init=False)

    # ------- Models -------
    # DINOv2 is the open default. Set SNOWLABEL_DINO to a local HF snapshot
    # path or to "dinov3_vitb16" (gated) to swap.
    dino_variant: str = field(
        default_factory=lambda: _env("SNOWLABEL_DINO", "dinov2_vitb14"),
    )
    sam_variant: str = field(
        default_factory=lambda: _env("SNOWLABEL_SAM", "sam2.1_t.pt"),
    )
    input_resolution: int = field(
        default_factory=lambda: int(_env("SNOWLABEL_RES", "224")),
    )
    device: str = field(
        default_factory=lambda: _env("SNOWLABEL_DEVICE", "auto"),
    )

    # ------- Snowball algorithm -------
    k_neighbors: int = field(
        default_factory=lambda: int(_env("SNOWLABEL_K", "5")),
    )
    # Thresholds on the winner's ABSOLUTE rerank blend score (see predictor;
    # class-count invariant). Calibrated 2026-06-12 by supervise.py sweeps on
    # TWO open-set seeds (strict: zero false-auto on both calibration runs;
    # the binding case is near-class confusables like cup->bottle at ~0.68).
    conf_auto_apply: float = field(
        default_factory=lambda: float(_env("SNOWLABEL_CONF_AUTO", "0.68")),
    )
    conf_suggest: float = field(
        default_factory=lambda: float(_env("SNOWLABEL_CONF_SUGGEST", "0.55")),
    )
    predict_mode: str = field(
        # "knn" (default) or "centroid" (NCM — one prototype per class)
        default_factory=lambda: _env("SNOWLABEL_PREDICT_MODE", "knn"),
    )
    # Cold-start guards: a class must have at least this many stored examples
    # before its predictions are allowed to reach the given tier. Prevents the
    # "1 sample => conf 1.0 => auto-apply" over-confidence that makes the early
    # snowball look like random guessing. Raise for stricter demos.
    min_support_auto: int = field(
        default_factory=lambda: int(_env("SNOWLABEL_MIN_SUPPORT_AUTO", "3")),
    )
    min_support_suggest: int = field(
        default_factory=lambda: int(_env("SNOWLABEL_MIN_SUPPORT_SUGGEST", "1")),
    )
    # ------- Feature-group retrieval (crop-encode + parts + geom rerank) -------
    # Stage 1 retrieves this many neighbors by v_whole, then they are re-ranked
    # by part-level MaxSim + geometry.
    rerank_n: int = field(
        default_factory=lambda: int(_env("SNOWLABEL_RERANK_N", "24")),
    )
    # Open-set guard: if the BEST neighbor's v_whole cosine is below this, the
    # query is treated as "unknown" (silent) — nothing in memory looks like it.
    open_set_min_sim: float = field(
        default_factory=lambda: float(_env("SNOWLABEL_OPEN_MIN_SIM", "0.45")),
    )
    # A prediction must beat the runner-up class's blend score by at least
    # this absolute gap to be eligible for the auto tier (filters one-class-
    # in-candidates flukes like bear->dog that pass the conf threshold).
    margin_auto: float = field(
        default_factory=lambda: float(_env("SNOWLABEL_MARGIN_AUTO", "0.08")),
    )
    # Each mask's crop patch grid is split into parts_grid x parts_grid cells,
    # each soft-pooled into one part vector (v_parts has parts_grid**2 rows).
    # Higher = finer spatial structure (needs a finer DINO patch grid, i.e. a
    # higher SNOWLABEL_RES, to stay robust). 2->4 quadruples the part count.
    parts_grid: int = field(
        default_factory=lambda: int(_env("SNOWLABEL_PARTS_GRID", "4")),
    )

    # ------- Memory backend -------
    memory_backend: str = field(
        # "numpy" (default, no deps) or "milvus" (uses milvus-lite if no uri)
        default_factory=lambda: _env("SNOWLABEL_MEMORY_BACKEND", "numpy"),
    )
    milvus_uri: str = field(
        # Leave blank to use milvus-lite at data/memory/snowlabel_milvus.db
        default_factory=lambda: _env("SNOWLABEL_MILVUS_URI", ""),
    )
    milvus_token: str = field(
        default_factory=lambda: _env("SNOWLABEL_MILVUS_TOKEN", ""),
    )

    # ------- 图片处理缓存(三库) -------
    # 开关:开启后图片处理结果(SAM mask + DINO 向量)持久化入库,同一张图再来直接
    # 读库、跳过 SAM/DINO。三库分工:MinIO 存图片本体、MySQL/MariaDB 存 mask 元信息、
    # Milvus 存向量。默认关闭以保持向后兼容;公网 demo 启动脚本里显式打开。
    imgcache_enabled: bool = field(
        default_factory=lambda: _env("SNOWLABEL_IMGCACHE", "0").lower()
        in ("1", "true", "yes"),
    )
    # MySQL/MariaDB:连接串走 mysql+pymysql 方言,连 MariaDB 与 MySQL 官方版完全一致,
    # 将来换 MySQL 只改 host/账号即可,代码零改动。
    mysql_host: str = field(default_factory=lambda: _env("SNOWLABEL_MYSQL_HOST", "127.0.0.1"))
    mysql_port: int = field(default_factory=lambda: int(_env("SNOWLABEL_MYSQL_PORT", "3306")))
    mysql_user: str = field(default_factory=lambda: _env("SNOWLABEL_MYSQL_USER", "root"))
    mysql_password: str = field(default_factory=lambda: _env("SNOWLABEL_MYSQL_PASSWORD", ""))
    mysql_db: str = field(default_factory=lambda: _env("SNOWLABEL_MYSQL_DB", "snowlabel"))
    # MinIO:S3 兼容对象存储,存图片本体(按内容哈希命名,天然去重)。
    minio_endpoint: str = field(default_factory=lambda: _env("SNOWLABEL_MINIO_ENDPOINT", "127.0.0.1:9000"))
    minio_access_key: str = field(default_factory=lambda: _env("SNOWLABEL_MINIO_ACCESS_KEY", "minioadmin"))
    minio_secret_key: str = field(default_factory=lambda: _env("SNOWLABEL_MINIO_SECRET_KEY", "minioadmin"))
    minio_bucket: str = field(default_factory=lambda: _env("SNOWLABEL_MINIO_BUCKET", "snowlabel"))
    minio_secure: bool = field(
        default_factory=lambda: _env("SNOWLABEL_MINIO_SECURE", "0").lower()
        in ("1", "true", "yes"),
    )
    # 缓存向量用的 Milvus collection(独立于记忆库 collection,互不干扰)。
    imgcache_collection: str = field(
        default_factory=lambda: _env("SNOWLABEL_IMGCACHE_COLLECTION", "mask_cache"))

    # ------- Server -------
    host: str = field(
        default_factory=lambda: _env("SNOWLABEL_HOST", "0.0.0.0"),
    )
    port: int = field(
        default_factory=lambda: int(_env("SNOWLABEL_PORT", "9090")),
    )
    # Multi-tenant public demo: when on, /api/* isolates each visitor by the
    # X-Snow-User header (their own uploaded images + their own memory anchors).
    # Off (default) = single-tenant: header ignored, one shared store (the local
    # car library) — behaves exactly as before.
    multi_user: bool = field(
        default_factory=lambda: _env("SNOWLABEL_MULTIUSER", "0").lower()
        in ("1", "true", "yes"),
    )

    def __post_init__(self) -> None:
        # mutate frozen via object.__setattr__ for init=False fields
        # SNOWLABEL_MEMORY_DIR lets the public demo point at an isolated store
        # (e.g. data/memory_public) so it never touches the local car library.
        _mem = os.environ.get("SNOWLABEL_MEMORY_DIR", "")
        if _mem:
            _mp = Path(_mem)
            _mp = _mp if _mp.is_absolute() else (self.project_root / _mem)
        else:
            _mp = self.project_root / "data" / "memory"
        object.__setattr__(self, "memory_dir", _mp)
        object.__setattr__(self, "cache_dir", self.project_root / "data" / "cache")
        object.__setattr__(self, "dataset_dir", self._resolve_dataset_dir())
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_dataset_dir(self) -> Path:
        """Folder of images the web labeling tool (labeler.html) browses.

        Priority: SNOWLABEL_DATASET_DIR (absolute, or relative to project_root)
        -> demo_images/ if it has files -> data/coco/val2017 fallback. Returns a
        path even if nothing exists; the API surfaces an empty dataset cleanly.
        """
        env = os.environ.get("SNOWLABEL_DATASET_DIR", "")
        if env:
            p = Path(env)
            return p if p.is_absolute() else (self.project_root / p)
        demo = self.project_root / "demo_images"
        if demo.is_dir() and any(demo.iterdir()):
            return demo
        coco = self.project_root / "data" / "coco" / "val2017"
        if coco.is_dir():
            return coco
        return demo

    @property
    def mysql_url(self) -> str:
        """SQLAlchemy 连接串。mysql+pymysql 方言对 MariaDB / MySQL 官方版通用。"""
        from urllib.parse import quote_plus
        pw = quote_plus(self.mysql_password)
        return (f"mysql+pymysql://{self.mysql_user}:{pw}@{self.mysql_host}:"
                f"{self.mysql_port}/{self.mysql_db}?charset=utf8mb4")


# Singleton accessor — import this everywhere.
def get_config() -> SnowConfig:
    return SnowConfig()
