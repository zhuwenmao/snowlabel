"""MilvusSnowballMemory — Milvus-backed store with a synchronous RAM mirror.

Why the RAM mirror: milvus-lite (embedded) has a multi-second insert -> ANN-
search visibility lag, and its `flush()` is broken on Windows (WinError 183 on
the manifest rename). For an *interactive* labeling loop that is fatal — a label
you just committed would be invisible to the very next prediction. So Milvus is
used as the durable, write-through store, while reads (ANN top-k, class counts,
centroids) run against an in-RAM mirror that is updated synchronously on every
add/forget. At demo/portfolio scale (<~100k entries) a brute-force cosine over
the mirror is sub-millisecond and also removes the per-call Milvus full scans.

Schema (collection "snowlabel_memory"):
    entry_id INT64 pk · vector FLOAT_VECTOR(dim)=v_whole · label VARCHAR(256)
    image_uri VARCHAR(1024) · confidence FLOAT · verified BOOL · created_at INT64
    parts VARCHAR(65535)  base64(float16) of [P,D] part vectors  (rerank only)
    geom  VARCHAR(1024)   JSON of geometry descriptors           (rerank only)
"""
from __future__ import annotations

import base64
import json
import logging
import threading
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .memory import MemoryEntry  # reuse the dataclass

log = logging.getLogger("snowlabel.memory_milvus")

COLLECTION_NAME = "snowlabel_memory"
# Milvus VARCHAR max_length is measured in BYTES (UTF-8: up to 3 per CJK char).
LABEL_MAX_LEN = 256
URI_MAX_LEN = 1024
PARTS_MAX_LEN = 65535   # 4x1024 float16 -> ~11k base64 chars
GEOM_MAX_LEN = 1024


def _fit_bytes(s: str, max_bytes: int) -> str:
    """Truncate `s` so its UTF-8 encoding fits in `max_bytes` without splitting a
    multi-byte character (Milvus counts VARCHAR length in bytes, Python slices by
    character)."""
    b = s.encode("utf-8")
    return s if len(b) <= max_bytes else b[:max_bytes].decode("utf-8", "ignore")


def _enc_f16(arr) -> str:
    return base64.b64encode(np.asarray(arr, np.float16).tobytes()).decode("ascii")


def _dec_parts(s: str, dim: int):
    if not s:
        return None
    a = np.frombuffer(base64.b64decode(s), np.float16).astype(np.float32)
    return a.reshape(-1, dim) if a.size else None


def _dec_geom(s: str):
    if not s:
        return None
    try:
        return np.asarray(json.loads(s), np.float32)
    except Exception:
        return None


class MilvusSnowballMemory:
    """Durable Milvus store + synchronous in-RAM read mirror.

    Same public surface as SnowballMemory: __len__, classes, add, add_batch,
    forget, query, predict_label, iter_entries, stats, flush.
    """

    def __init__(
        self,
        root: Path,
        dim: int,
        *,
        uri: str | None = None,
        token: str | None = None,
    ) -> None:
        try:
            from pymilvus import MilvusClient, DataType
        except ImportError as e:
            raise ImportError(
                "pymilvus required for MilvusSnowballMemory. "
                "pip install pymilvus milvus-lite"
            ) from e

        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.dim = dim
        self._lock = threading.RLock()
        self._next_id_path = self.root / "milvus_next_id.json"

        self._uri = uri or str(self.root / "snowlabel_milvus.db")
        log.info("Connecting MilvusClient(uri=%s)", self._uri)
        self._client = MilvusClient(uri=self._uri, token=token or "")
        self._DataType = DataType
        self._ensure_collection()
        self._next_id = self._load_next_id()

        # ---- in-RAM mirror (source of truth for all reads) ----
        self._ids: list[int] = []
        self._V: np.ndarray = np.zeros((0, self.dim), dtype=np.float32)
        self._labels: list[str] = []
        self._uris: list[str] = []
        self._parts: list = []
        self._geoms: list = []
        self._load_cache()
        self._check_feature_space()

    def _check_feature_space(self) -> None:
        """Refuse a non-empty store written under a different feature space
        (model / resolution / parts grid / crop pad) — cosine against those
        vectors is silently meaningless. Identity kept in a sidecar file."""
        from .features import feature_space_id
        fid = feature_space_id(self.dim)
        fs_path = self.root / "feature_space.json"
        if fs_path.exists() and len(self._ids) > 0:
            try:
                old = json.loads(fs_path.read_text(encoding="utf-8")
                                 ).get("feature_space")
            except Exception:
                old = None
            if old and old != fid:
                raise RuntimeError(
                    f"Milvus store at {self.root} was written under feature "
                    f"space '{old}' but the current config produces '{fid}'. "
                    f"Restore the old config, or clear the store and re-label.")
        fs_path.write_text(json.dumps({"feature_space": fid},
                                      ensure_ascii=False), encoding="utf-8")

    # -------------------- schema bootstrap --------------------
    def _ensure_collection(self) -> None:
        if self._client.has_collection(COLLECTION_NAME):
            info = self._client.describe_collection(COLLECTION_NAME)
            for f in info["fields"]:
                if f["name"] == "vector":
                    existing_dim = int(f["params"]["dim"])
                    if existing_dim != self.dim:
                        raise ValueError(
                            f"Existing Milvus collection has dim={existing_dim}, "
                            f"current backbone produces dim={self.dim}. Drop the "
                            f"collection or switch backbones."
                        )
            self._client.load_collection(COLLECTION_NAME)
            return

        log.info("Creating Milvus collection %s (dim=%d)", COLLECTION_NAME, self.dim)
        schema = self._client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("entry_id", self._DataType.INT64, is_primary=True)
        schema.add_field("vector", self._DataType.FLOAT_VECTOR, dim=self.dim)
        schema.add_field("label", self._DataType.VARCHAR, max_length=LABEL_MAX_LEN)
        schema.add_field("image_uri", self._DataType.VARCHAR, max_length=URI_MAX_LEN)
        schema.add_field("confidence", self._DataType.FLOAT)
        schema.add_field("verified", self._DataType.BOOL)
        schema.add_field("created_at", self._DataType.INT64)
        schema.add_field("parts", self._DataType.VARCHAR, max_length=PARTS_MAX_LEN)
        schema.add_field("geom", self._DataType.VARCHAR, max_length=GEOM_MAX_LEN)

        self._client.create_collection(COLLECTION_NAME, schema=schema, index_params=None)
        idx = self._client.prepare_index_params()
        idx.add_index(field_name="vector", index_type="FLAT",
                      metric_type="COSINE", index_name="vec_idx")
        self._client.create_index(COLLECTION_NAME, idx)
        self._client.load_collection(COLLECTION_NAME)

    def _load_next_id(self) -> int:
        if self._next_id_path.exists():
            try:
                return int(json.loads(self._next_id_path.read_text()).get("next_id", 0))
            except Exception:
                pass
        try:
            res = self._client.query(COLLECTION_NAME, filter="",
                                     output_fields=["entry_id"], limit=100_000)
            return max((r["entry_id"] for r in res), default=-1) + 1
        except Exception:
            return 0

    def _save_next_id(self) -> None:
        self._next_id_path.write_text(json.dumps({"next_id": self._next_id}))

    def _load_cache(self) -> None:
        """Pull every row from Milvus into the RAM mirror (once, at startup)."""
        try:
            res = self._client.query(
                COLLECTION_NAME, filter="",
                output_fields=["entry_id", "vector", "label", "image_uri",
                               "parts", "geom"],
                limit=100_000,
            )
        except Exception as e:
            log.warning("cache load failed (%s); starting empty", e)
            res = []
        res.sort(key=lambda r: int(r["entry_id"]))
        vs = []
        for r in res:
            self._ids.append(int(r["entry_id"]))
            vs.append(np.asarray(r["vector"], np.float32))
            self._labels.append(str(r.get("label", "")))
            self._uris.append(str(r.get("image_uri", "")))
            self._parts.append(_dec_parts(r.get("parts", ""), self.dim))
            self._geoms.append(_dec_geom(r.get("geom", "")))
        self._V = (np.stack(vs).astype(np.float32) if vs
                   else np.zeros((0, self.dim), dtype=np.float32))
        log.info("RAM mirror loaded: %d entries", len(self._ids))

    # -------------------- mutation (write-through) --------------------
    def add(
        self,
        vector: np.ndarray,
        label: str,
        *,
        image_uri: str = "",
        confidence: float = 1.0,
        verified: bool = True,
        user_id: str = "",   # accepted for parity; not persisted (see query note)
        extra: dict | None = None,
        parts: np.ndarray | None = None,
        geom: np.ndarray | None = None,
    ) -> int:
        if vector.ndim != 1 or vector.shape[0] != self.dim:
            raise ValueError(
                f"Expected 1D vector of dim {self.dim}, got shape {vector.shape}"
            )
        v = vector.astype(np.float32)
        with self._lock:
            entry_id = self._next_id
            self._next_id += 1
            self._client.insert(COLLECTION_NAME, [{
                "entry_id": entry_id,
                "vector": v.tolist(),
                "label": _fit_bytes(label, LABEL_MAX_LEN),
                "image_uri": _fit_bytes(image_uri, URI_MAX_LEN),
                "confidence": float(confidence),
                "verified": bool(verified),
                "created_at": int(time.time() * 1000),
                "parts": _enc_f16(parts) if parts is not None else "",
                "geom": json.dumps([float(x) for x in geom]) if geom is not None else "",
            }])
            self._save_next_id()
            # mirror
            self._ids.append(entry_id)
            self._V = np.vstack([self._V, v.reshape(1, -1)])
            self._labels.append(label)
            self._uris.append(image_uri)
            self._parts.append(np.asarray(parts, np.float32) if parts is not None else None)
            self._geoms.append(np.asarray(geom, np.float32) if geom is not None else None)
        return entry_id

    def add_batch(
        self,
        vectors: np.ndarray,
        labels: Sequence[str],
        *,
        image_uri: str = "",
        user_id: str = "",   # accepted for parity; not persisted (see query note)
        parts: np.ndarray | None = None,   # [N, P, D]
        geom: np.ndarray | None = None,    # [N, G]
    ) -> list[int]:
        if vectors.shape[0] != len(labels):
            raise ValueError("vectors and labels length mismatch")
        ids: list[int] = []
        rows = []
        now = int(time.time() * 1000)
        V = vectors.astype(np.float32)
        with self._lock:
            for i, (v, lbl) in enumerate(zip(V, labels)):
                eid = self._next_id
                self._next_id += 1
                ids.append(eid)
                rows.append({
                    "entry_id": eid,
                    "vector": v.tolist(),
                    "label": _fit_bytes(lbl, LABEL_MAX_LEN),
                    "image_uri": _fit_bytes(image_uri, URI_MAX_LEN),
                    "confidence": 1.0,
                    "verified": True,
                    "created_at": now,
                    "parts": _enc_f16(parts[i]) if parts is not None else "",
                    "geom": (json.dumps([float(x) for x in geom[i]])
                             if geom is not None else ""),
                })
            self._client.insert(COLLECTION_NAME, rows)
            self._save_next_id()
            # mirror
            self._ids.extend(ids)
            self._V = np.vstack([self._V, V]) if self._V.size else V.copy()
            for i, lbl in enumerate(labels):
                self._labels.append(lbl)
                self._uris.append(image_uri)
                self._parts.append(np.asarray(parts[i], np.float32) if parts is not None else None)
                self._geoms.append(np.asarray(geom[i], np.float32) if geom is not None else None)
        return ids

    def forget(self, entry_id: int) -> bool:
        with self._lock:
            try:
                self._client.delete(COLLECTION_NAME, filter=f"entry_id == {int(entry_id)}")
            except Exception as e:
                log.warning("forget(%d) Milvus delete failed: %s", entry_id, e)
            try:
                i = self._ids.index(int(entry_id))
            except ValueError:
                return False
            self._ids.pop(i); self._labels.pop(i); self._uris.pop(i)
            self._parts.pop(i); self._geoms.pop(i)
            self._V = np.delete(self._V, i, axis=0)
            return True

    def flush(self) -> None:
        # No-op: milvus-lite flush is broken on Windows; the RAM mirror is the
        # read source and Milvus inserts are already durable per-call.
        return

    # -------------------- query (from RAM mirror) --------------------
    def __len__(self) -> int:
        return len(self._ids)

    @property
    def classes(self) -> list[str]:
        return sorted(set(self._labels))

    def _entry(self, i: int, *, groups: bool = False) -> MemoryEntry:
        return MemoryEntry(
            entry_id=self._ids[i], label=self._labels[i], image_uri=self._uris[i],
            confidence=1.0, verified=True,
            parts=self._parts[i] if groups else None,
            geom=self._geoms[i] if groups else None,
        )

    def query(self, vector: np.ndarray, k: int = 5, user_id: str | None = None):
        # The Milvus backend has no per-user isolation (its schema has no
        # user_id field); user_id is accepted only for interface parity with the
        # numpy backend. The multi-tenant public demo runs on numpy; the local
        # single-tenant car library never passes a non-empty user_id, so this is
        # a no-op here.
        n = len(self._ids)
        if n == 0:
            return [], np.empty((0,), dtype=np.float32)
        v = vector.astype(np.float32).reshape(-1)
        sims = self._V @ v                      # cosine: both sides L2-normalized
        k = min(k, n)
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [self._entry(int(i), groups=True) for i in idx], sims[idx].astype(np.float32)

    def predict_label(self, vector: np.ndarray, k: int = 5, mode: str = "knn",
                      user_id: str | None = None):
        if mode == "centroid":
            return self._predict_via_centroid(vector)
        entries, sims = self.query(vector, k=k, user_id=user_id)
        if not entries:
            return None, 0.0, []
        weights = np.clip(sims, 0.0, None).astype(float)
        scores: dict[str, float] = {}
        for w, e in zip(weights, entries):
            scores[e.label] = scores.get(e.label, 0.0) + float(w)
        total = sum(scores.values())
        if total <= 0:
            return None, 0.0, []
        winner, ws = max(scores.items(), key=lambda kv: kv[1])
        return winner, ws / total, sorted(scores.items(), key=lambda kv: -kv[1])

    def _predict_via_centroid(self, vector: np.ndarray):
        if len(self._ids) == 0:
            return None, 0.0, []
        by_class: dict[str, list[np.ndarray]] = {}
        for i, lbl in enumerate(self._labels):
            by_class.setdefault(lbl, []).append(self._V[i])
        labels = list(by_class.keys())
        C = np.stack([
            (lambda m: m / (np.linalg.norm(m) + 1e-8))(np.mean(np.stack(vs), axis=0))
            for vs in by_class.values()
        ]).astype(np.float32)
        v = vector.astype(np.float32).reshape(-1)
        sims = C @ v
        weights = np.exp((sims - sims.max()) * 4.0)
        probs = weights / weights.sum()
        wi = int(np.argmax(probs))
        breakdown = sorted([(labels[i], float(probs[i])) for i in range(len(labels))],
                           key=lambda kv: -kv[1])
        return labels[wi], float(probs[wi]), breakdown

    # -------------------- introspection (from RAM mirror) --------------------
    def iter_entries(self) -> Iterable[MemoryEntry]:
        for i in range(len(self._ids)):
            yield self._entry(i)

    def stats(self) -> dict:
        from collections import Counter
        counts = Counter(self._labels)
        return {
            "total": len(self._ids),
            "dim": self.dim,
            "classes": dict(counts),
            "verified": len(self._ids),
            "backend": "milvus",
            "uri": self._uri,
        }
