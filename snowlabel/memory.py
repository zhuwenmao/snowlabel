"""SnowballMemory — persistent vector store for user-corrected labels.

Phase 1 implementation: numpy arrays on disk + JSON metadata. Simple, fast
enough up to ~50k entries, no external services. Phase 2 adds a parallel
Milvus implementation (`memory_milvus.MilvusSnowballMemory`) with the same
interface; select via `SNOWLABEL_MEMORY_BACKEND`.

The store is *append-mostly* by design — the only deletions are user-initiated
"forget this sample" actions. We expose those as `forget(entry_id)`.

File layout:
    data/memory/
        embeddings.npy        # (N, D) float32
        metadata.jsonl        # one entry per row, same order as embeddings
        index.json            # version + class catalog
"""
from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


def get_memory(root: Path, dim: int):
    """Factory: instantiate the configured memory backend.

    Reads SNOWLABEL_MEMORY_BACKEND from config (env). Falls back to numpy.
    Used by SnowPredictor so callers don't care which backend is live.
    """
    from .config import get_config
    cfg = get_config()
    backend = (cfg.memory_backend or "numpy").lower()
    if backend == "milvus":
        from .memory_milvus import MilvusSnowballMemory
        return MilvusSnowballMemory(
            root=root, dim=dim,
            uri=cfg.milvus_uri or None,
            token=cfg.milvus_token or None,
        )
    if backend != "numpy":
        import logging
        logging.getLogger("snowlabel.memory").warning(
            "Unknown SNOWLABEL_MEMORY_BACKEND=%r; falling back to numpy.",
            backend,
        )
    return SnowballMemory(root, dim=dim)


@dataclass
class MemoryEntry:
    """One labeled instance stored in the snowball memory."""
    entry_id: int
    label: str               # human-readable class name
    image_uri: str           # what the model saw (path or URL)
    confidence: float        # 1.0 = user verified, lower = model self-tag
    verified: bool = True    # user-confirmed (vs auto-added)
    user_id: str = ""        # multi-tenant isolation tag ("" = global/shared)
    extra: dict | None = None
    # Feature-group extras — only populated by query() for the rerank stage,
    # never serialized into metadata (kept in parallel .npy arrays).
    parts: np.ndarray | None = None   # [P, D]
    geom: np.ndarray | None = None    # [G]


class SnowballMemory:
    """Append-only vector memory with k-NN retrieval.

    The interface is intentionally narrow so we can swap the storage backend
    (numpy -> Milvus) without touching callers.
    """

    def __init__(self, root: Path, dim: int) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.dim = dim
        self._lock = threading.RLock()

        self._emb_path = self.root / "embeddings.npy"
        self._meta_path = self.root / "metadata.jsonl"
        self._index_path = self.root / "index.json"
        self._parts_path = self.root / "parts.npy"   # (N, P, D) feature-group parts
        self._geom_path = self.root / "geom.npy"     # (N, G) geometry descriptors

        self._embeddings: np.ndarray  # (N, D) float32
        self._entries: list[MemoryEntry] = []
        self._parts: np.ndarray | None = None
        self._geom: np.ndarray | None = None
        self._next_id: int = 0
        self._load()

    # -------------------- persistence --------------------
    def _check_feature_space(self, n_entries: int) -> str:
        """Refuse to serve vectors written under a different feature space
        (model / resolution / parts grid / crop pad) — similarities against
        them are silently meaningless."""
        from .features import feature_space_id
        fid = feature_space_id(self.dim)
        if n_entries and self._index_path.exists():
            try:
                old = json.loads(self._index_path.read_text(encoding="utf-8")
                                 ).get("feature_space")
            except Exception:
                old = None
            if old and old != fid:
                raise RuntimeError(
                    f"Memory at {self.root} was written under feature space "
                    f"'{old}' but the current config produces '{fid}'. Old "
                    f"vectors are incompatible — restore the old config, or "
                    f"clear the store and re-label.")
        return fid

    def _load(self) -> None:
        if self._emb_path.exists() and self._meta_path.exists():
            self._embeddings = np.load(self._emb_path).astype(np.float32)
            if self._embeddings.ndim != 2 or self._embeddings.shape[1] != self.dim:
                raise ValueError(
                    f"Stored embeddings shape {self._embeddings.shape} does not "
                    f"match expected dim {self.dim}. Clear data/memory/ to reset."
                )
            with open(self._meta_path, "r", encoding="utf-8") as f:
                self._entries = [MemoryEntry(**json.loads(line)) for line in f if line.strip()]
            if len(self._entries) != len(self._embeddings):
                raise ValueError(
                    f"Memory inconsistency: {len(self._entries)} meta rows, "
                    f"{len(self._embeddings)} vectors."
                )
            self._next_id = (max((e.entry_id for e in self._entries), default=-1) + 1)
        else:
            self._embeddings = np.zeros((0, self.dim), dtype=np.float32)
            self._entries = []
            self._next_id = 0
        self._parts = (np.load(self._parts_path).astype(np.float32)
                       if self._parts_path.exists() else None)
        self._geom = (np.load(self._geom_path).astype(np.float32)
                      if self._geom_path.exists() else None)
        self._fid = self._check_feature_space(len(self._entries))
        self._save_index()

    def _save_index(self) -> None:
        with open(self._index_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "version": 1,
                    "dim": self.dim,
                    "feature_space": self._fid,
                    "n_entries": len(self._entries),
                    "classes": sorted({e.label for e in self._entries}),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    def flush(self) -> None:
        """Persist current state to disk. Called after each add()."""
        with self._lock:
            np.save(self._emb_path, self._embeddings)
            if self._parts is not None:
                np.save(self._parts_path, self._parts)
            if self._geom is not None:
                np.save(self._geom_path, self._geom)
            with open(self._meta_path, "w", encoding="utf-8") as f:
                for e in self._entries:
                    f.write(json.dumps({
                        "entry_id": e.entry_id, "label": e.label,
                        "image_uri": e.image_uri, "confidence": e.confidence,
                        "verified": e.verified, "user_id": e.user_id,
                        "extra": e.extra,
                    }, ensure_ascii=False) + "\n")
            self._save_index()

    # -------------------- mutation --------------------
    def _append_group(self, parts: np.ndarray | None, geom: np.ndarray | None) -> None:
        """Append one row to the parallel parts/geom arrays (keeps them aligned
        with self._entries). Missing groups are stored as zeros so indices stay
        in lock-step."""
        if parts is not None:
            p = np.asarray(parts, np.float32)[None, ...]
            self._parts = p if self._parts is None else np.concatenate([self._parts, p], 0)
        if geom is not None:
            g = np.asarray(geom, np.float32).reshape(1, -1)
            self._geom = g if self._geom is None else np.concatenate([self._geom, g], 0)

    def add(
        self,
        vector: np.ndarray,
        label: str,
        *,
        image_uri: str = "",
        confidence: float = 1.0,
        verified: bool = True,
        user_id: str = "",
        extra: dict | None = None,
        parts: np.ndarray | None = None,
        geom: np.ndarray | None = None,
    ) -> int:
        """Insert one labeled vector (+ optional feature group). Returns entry_id."""
        if vector.ndim != 1 or vector.shape[0] != self.dim:
            raise ValueError(
                f"Expected 1D vector of dim {self.dim}, got shape {vector.shape}"
            )
        with self._lock:
            entry = MemoryEntry(
                entry_id=self._next_id,
                label=label,
                image_uri=image_uri,
                confidence=confidence,
                verified=verified,
                user_id=user_id,
                extra=extra,
            )
            self._next_id += 1
            self._entries.append(entry)
            self._embeddings = np.vstack(
                [self._embeddings, vector.astype(np.float32).reshape(1, -1)]
            )
            self._append_group(parts, geom)
        self.flush()
        return entry.entry_id

    def add_batch(
        self,
        vectors: np.ndarray,
        labels: Sequence[str],
        *,
        image_uri: str = "",
        user_id: str = "",
        parts: np.ndarray | None = None,   # [N, P, D]
        geom: np.ndarray | None = None,    # [N, G]
    ) -> list[int]:
        """Insert many at once (one flush at the end)."""
        if vectors.shape[0] != len(labels):
            raise ValueError("vectors and labels length mismatch")
        ids = []
        with self._lock:
            for v, lbl in zip(vectors, labels):
                entry = MemoryEntry(
                    entry_id=self._next_id,
                    label=lbl,
                    image_uri=image_uri,
                    confidence=1.0,
                    verified=True,
                    user_id=user_id,
                )
                self._next_id += 1
                self._entries.append(entry)
                ids.append(entry.entry_id)
            self._embeddings = np.vstack(
                [self._embeddings, vectors.astype(np.float32)]
            )
            if parts is not None:
                p = np.asarray(parts, np.float32)
                self._parts = p if self._parts is None else np.concatenate([self._parts, p], 0)
            if geom is not None:
                g = np.asarray(geom, np.float32)
                self._geom = g if self._geom is None else np.concatenate([self._geom, g], 0)
        self.flush()
        return ids

    def forget(self, entry_id: int) -> bool:
        """Remove an entry by ID. Returns True if found."""
        with self._lock:
            for i, e in enumerate(self._entries):
                if e.entry_id == entry_id:
                    self._entries.pop(i)
                    self._embeddings = np.delete(self._embeddings, i, axis=0)
                    if self._parts is not None and i < len(self._parts):
                        self._parts = np.delete(self._parts, i, axis=0)
                    if self._geom is not None and i < len(self._geom):
                        self._geom = np.delete(self._geom, i, axis=0)
                    self.flush()
                    return True
        return False

    # -------------------- query --------------------
    def __len__(self) -> int:
        return len(self._entries)

    @property
    def classes(self) -> list[str]:
        return sorted({e.label for e in self._entries})

    def query(
        self,
        vector: np.ndarray,
        k: int = 5,
        user_id: str | None = None,
    ) -> tuple[list[MemoryEntry], np.ndarray]:
        """Return (top-k entries, their cosine similarities) for one query.

        If ``user_id`` is non-empty, retrieval is restricted to that user's own
        anchors (multi-tenant isolation for the public demo). ``None`` / ``""``
        searches the whole store — the original single-tenant behavior the local
        labeler and the supervise harness rely on.
        """
        if len(self._entries) == 0:
            return [], np.empty((0,), dtype=np.float32)
        v = vector.astype(np.float32).reshape(-1)
        sims = self._embeddings @ v
        if user_id:
            pool = np.fromiter(
                (i for i, e in enumerate(self._entries) if e.user_id == user_id),
                dtype=int)
            if pool.size == 0:
                return [], np.empty((0,), dtype=np.float32)
            pool_sims = sims[pool]
            k_eff = min(k, pool.size)
            local = np.argpartition(-pool_sims, k_eff - 1)[:k_eff]
            top_idx = pool[local[np.argsort(-pool_sims[local])]]
        else:
            k_eff = min(k, len(self._entries))
            top_idx = np.argpartition(-sims, k_eff - 1)[:k_eff]
            top_idx = top_idx[np.argsort(-sims[top_idx])]
        out: list[MemoryEntry] = []
        for i in top_idx:
            e = self._entries[i]
            out.append(MemoryEntry(
                entry_id=e.entry_id, label=e.label, image_uri=e.image_uri,
                confidence=e.confidence, verified=e.verified, user_id=e.user_id,
                extra=e.extra,
                parts=(self._parts[i] if self._parts is not None and i < len(self._parts) else None),
                geom=(self._geom[i] if self._geom is not None and i < len(self._geom) else None),
            ))
        return out, sims[top_idx]

    def predict_label(
        self,
        vector: np.ndarray,
        k: int = 5,
        mode: str = "knn",
        user_id: str | None = None,
    ) -> tuple[str | None, float, list[tuple[str, float]]]:
        """Predict label with the snowball memory.

        Modes:
            "knn"      : k-NN cosine vote over individual entries (default).
            "centroid" : nearest-class-mean (NCM). One prototype per class,
                         computed on the fly as the L2-normalized mean of all
                         that class's entries. Often more stable than k-NN
                         once each class has ≥3 examples, and gives more
                         calibrated confidences with high-dim backbones like
                         DINOv3-L.

        Returns: (label, confidence, top-k breakdown).
        """
        if mode == "centroid":
            return self._predict_via_centroid(vector, user_id=user_id)
        return self._predict_via_knn(vector, k=k, user_id=user_id)

    def _predict_via_knn(
        self,
        vector: np.ndarray,
        k: int,
        user_id: str | None = None,
    ) -> tuple[str | None, float, list[tuple[str, float]]]:
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
        winner, winner_score = max(scores.items(), key=lambda kv: kv[1])
        breakdown = sorted(scores.items(), key=lambda kv: -kv[1])
        return winner, winner_score / total, breakdown

    def _predict_via_centroid(
        self,
        vector: np.ndarray,
        user_id: str | None = None,
    ) -> tuple[str | None, float, list[tuple[str, float]]]:
        """Nearest-class-mean over L2-normalized per-class centroids."""
        if len(self._entries) == 0:
            return None, 0.0, []
        v = vector.astype(np.float32).reshape(-1)

        # Build (or use cached) per-class centroids.
        centroids = self._compute_centroids(user_id=user_id)
        labels = list(centroids.keys())
        if not labels:
            return None, 0.0, []
        C = np.stack(list(centroids.values()))  # (n_classes, D), L2-normed
        sims = C @ v
        # Softmax-ish: convert cosine sims to a probability over classes.
        # We use a temperature of 1.0 — the difference between best and second
        # is meaningful; users can tune in config if needed.
        shifted = sims - sims.max()
        weights = np.exp(shifted * 4.0)  # temperature controls sharpness
        probs = weights / weights.sum()
        winner_idx = int(np.argmax(probs))
        breakdown = sorted(
            [(labels[i], float(probs[i])) for i in range(len(labels))],
            key=lambda kv: -kv[1],
        )
        return labels[winner_idx], float(probs[winner_idx]), breakdown

    def _compute_centroids(self, user_id: str | None = None) -> dict[str, np.ndarray]:
        """L2-normalized mean of every class's verified entries."""
        by_class: dict[str, list[np.ndarray]] = {}
        for i, e in enumerate(self._entries):
            if user_id and e.user_id != user_id:
                continue
            # Trust verified entries; fall back to all if a class has none.
            by_class.setdefault(e.label, []).append(self._embeddings[i])
        out: dict[str, np.ndarray] = {}
        for label, vecs in by_class.items():
            m = np.mean(np.stack(vecs), axis=0)
            n = np.linalg.norm(m)
            if n > 1e-8:
                out[label] = (m / n).astype(np.float32)
        return out

    # -------------------- introspection --------------------
    def iter_entries(self) -> Iterable[MemoryEntry]:
        return iter(self._entries)

    def stats(self) -> dict:
        from collections import Counter
        counts = Counter(e.label for e in self._entries)
        return {
            "total": len(self._entries),
            "dim": self.dim,
            "classes": dict(counts),
            "verified": sum(1 for e in self._entries if e.verified),
        }
