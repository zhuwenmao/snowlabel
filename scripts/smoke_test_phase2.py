"""Phase 2 smoke — exercise the new memory backends and centroid mode.

Matrix:
    numpy backend × knn mode      (legacy default)
    numpy backend × centroid mode (NCM classifier)
    milvus backend × knn mode     (production target)
    milvus backend × centroid mode

For each cell we:
    1. Wipe the relevant memory dir.
    2. Build a small fake snowball (50 synthetic vectors across 5 classes).
    3. Query 20 fresh vectors, measure correctness.
    4. Test forget(), search via memory inspector function, etc.

This is purely about the storage / retrieval layer — no SAM, no DINO, no
real images. Fast (~5 sec total). For end-to-end with real models, see
smoke_test_phase1.py.

Usage:
    python scripts/smoke_test_phase2.py
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DIM = 256
N_CLASSES = 5
N_PER_CLASS = 10
N_QUERIES_PER_CLASS = 4
SEED = 1234


def make_synthetic_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (train_vecs, train_labels, query_vecs, query_labels).

    Each class lives on a tight cluster around a random unit vector, so
    cosine-NN should achieve near-perfect accuracy if storage works at all.
    """
    rng = np.random.default_rng(SEED)
    centers = rng.standard_normal((N_CLASSES, DIM)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    train_vecs, train_lbls = [], []
    query_vecs, query_lbls = [], []
    for c in range(N_CLASSES):
        for _ in range(N_PER_CLASS):
            v = centers[c] + 0.1 * rng.standard_normal(DIM).astype(np.float32)
            v /= np.linalg.norm(v)
            train_vecs.append(v); train_lbls.append(f"class_{c}")
        for _ in range(N_QUERIES_PER_CLASS):
            v = centers[c] + 0.1 * rng.standard_normal(DIM).astype(np.float32)
            v /= np.linalg.norm(v)
            query_vecs.append(v); query_lbls.append(f"class_{c}")
    return (
        np.stack(train_vecs),
        np.asarray(train_lbls),
        np.stack(query_vecs),
        np.asarray(query_lbls),
    )


def run_cell(backend: str, mode: str, root: Path) -> dict:
    print(f"\n--- backend={backend}  mode={mode} ---")
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)

    if backend == "numpy":
        from snowlabel.memory import SnowballMemory
        mem = SnowballMemory(root, dim=DIM)
    elif backend == "milvus":
        from snowlabel.memory_milvus import MilvusSnowballMemory
        mem = MilvusSnowballMemory(root, dim=DIM)
    else:
        raise ValueError(backend)

    tr_v, tr_l, q_v, q_l = make_synthetic_dataset()

    # Insert in batches of 10 for milvus parity.
    for start in range(0, len(tr_v), 10):
        mem.add_batch(tr_v[start:start + 10], list(tr_l[start:start + 10]))
    print(f"  inserted: {len(mem)}")

    # Query
    correct = 0
    for v, lbl in zip(q_v, q_l):
        pred, conf, _ = mem.predict_label(v, k=5, mode=mode)
        if pred == lbl:
            correct += 1
    acc = correct / len(q_v)
    print(f"  query accuracy: {correct}/{len(q_v)} = {acc:.2%}")

    # forget the first entry, verify count goes down
    first_id = next(iter(mem.iter_entries())).entry_id
    forgot = mem.forget(first_id)
    print(f"  forget(entry_id={first_id}): ok={forgot} new_len={len(mem)}")

    # search interface
    entries, sims = mem.query(q_v[0], k=3)
    print(f"  query top-3 first sim: {float(sims[0]):.3f} label={entries[0].label}")

    classes = mem.classes
    stats = mem.stats()
    print(f"  classes: {classes}")
    print(f"  stats : total={stats['total']}  backend={stats.get('backend', 'numpy')}")

    return {
        "backend": backend, "mode": mode,
        "accuracy": acc, "size_after_forget": len(mem),
        "classes": classes, "stats": stats,
    }


def main() -> int:
    cells = [
        ("numpy", "knn",      PROJECT_ROOT / "data" / "memory_smoke_p2_np_knn"),
        ("numpy", "centroid", PROJECT_ROOT / "data" / "memory_smoke_p2_np_cent"),
        ("milvus", "knn",     PROJECT_ROOT / "data" / "memory_smoke_p2_mv_knn"),
        ("milvus", "centroid", PROJECT_ROOT / "data" / "memory_smoke_p2_mv_cent"),
    ]
    results = []
    failures: list[str] = []
    for backend, mode, root in cells:
        try:
            r = run_cell(backend, mode, root)
            results.append(r)
        except Exception as e:
            print(f"  FAIL: {type(e).__name__}: {e}")
            failures.append(f"{backend}/{mode}: {e}")

    print("\n" + "=" * 60)
    print("PHASE 2 SMOKE SUMMARY")
    print("=" * 60)
    for r in results:
        status = "PASS" if r["accuracy"] >= 0.90 else "WEAK"
        print(f"  {status:>4}  backend={r['backend']:<6} mode={r['mode']:<8} "
              f"acc={r['accuracy']:.2%}  size={r['size_after_forget']}")
    if failures:
        print()
        for f in failures:
            print(f"  FAIL: {f}")
        return 1
    if all(r["accuracy"] >= 0.90 for r in results):
        print("\nALL CELLS PASSED")
        return 0
    print("\nSome cells under-performed (< 90% on synthetic clusters)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
