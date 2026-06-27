"""Verify the feature-space fingerprint guard refuses mismatched stores.

Both backends: write one entry under parts_grid=4, reload under parts_grid=2,
expect a RuntimeError (cosine against vectors from another feature space is
silently meaningless, so loading must refuse). Fully self-contained — uses
temp dirs, so it's re-runnable regardless of any existing store's state.
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ok = True


def unit(dim):
    v = np.ones(dim, np.float32)
    return v / np.linalg.norm(v)


# --- case 1: numpy backend
os.environ["SNOWLABEL_PARTS_GRID"] = "4"
from snowlabel.memory import SnowballMemory  # noqa: E402

tmp1 = Path(tempfile.mkdtemp(prefix="snowfp_np_"))
try:
    m = SnowballMemory(tmp1, dim=8)
    m.add(unit(8), label="t", parts=np.zeros((16, 8), np.float32),
          geom=np.zeros(4, np.float32))
    del m
    os.environ["SNOWLABEL_PARTS_GRID"] = "2"
    try:
        m2 = SnowballMemory(tmp1, dim=8)
        print(f"[FAIL] numpy guard did not fire (loaded {len(m2)} entries)")
        ok = False
    except RuntimeError as e:
        print("[OK] numpy guard fired:", str(e)[:110], "...")
finally:
    shutil.rmtree(tmp1, ignore_errors=True)

# --- case 2: milvus backend
from snowlabel.memory_milvus import MilvusSnowballMemory  # noqa: E402

tmp2 = Path(tempfile.mkdtemp(prefix="snowfp_mv_"))
try:
    os.environ["SNOWLABEL_PARTS_GRID"] = "4"
    m1 = MilvusSnowballMemory(tmp2, dim=8)
    m1.add(unit(8), label="t")
    m1 = None
    os.environ["SNOWLABEL_PARTS_GRID"] = "2"
    try:
        m3 = MilvusSnowballMemory(tmp2, dim=8)
        print(f"[FAIL] milvus guard did not fire (loaded {len(m3)} entries)")
        ok = False
    except RuntimeError as e:
        print("[OK] milvus guard fired:", str(e)[:110], "...")
finally:
    shutil.rmtree(tmp2, ignore_errors=True)

print("\nGUARD OK" if ok else "\nGUARD BROKEN")
sys.exit(0 if ok else 1)
