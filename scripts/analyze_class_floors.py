"""Is there a gap between intra-class similarity and near-class impostors?

Loads the isolated supervise memory (125 entries from the last benchmark) and
prints, per class, the leave-one-out BEST cosine of each member to the rest of
its class — the distribution a per-class adaptive open-set floor would use.
If genuine members sit well above where the cup->bottle / truck->car probes
landed (~0.65-0.73 raw cosine), an adaptive floor can separate them.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MEM = ROOT / "data" / "memory_supervise"

V = np.load(MEM / "embeddings.npy").astype(np.float32)
labels = [json.loads(l)["label"] for l in
          open(MEM / "metadata.jsonl", encoding="utf-8") if l.strip()]
labels = np.array(labels)
print(f"{len(labels)} entries, classes: {sorted(set(labels))}\n")

print(f"{'class':>8} {'n':>4} {'min':>6} {'q10':>6} {'q25':>6} {'med':>6} {'mean':>6}")
for c in sorted(set(labels)):
    idx = np.where(labels == c)[0]
    Vc = V[idx]
    S = Vc @ Vc.T
    np.fill_diagonal(S, -1)              # leave-one-out
    best = S.max(axis=1)                  # each member's best match in-class
    print(f"{c:>8} {len(idx):>4} {best.min():>6.3f} "
          f"{np.quantile(best, 0.10):>6.3f} {np.quantile(best, 0.25):>6.3f} "
          f"{np.median(best):>6.3f} {best.mean():>6.3f}")

print("\ncross-class impostor view: best cosine from members of class A to "
      "memory of class B (max over pairs)")
classes = sorted(set(labels))
print(f"{'':>8}" + "".join(f"{c:>9}" for c in classes))
for a in classes:
    Va = V[labels == a]
    row = []
    for b in classes:
        if a == b:
            row.append("    -")
            continue
        Vb = V[labels == b]
        row.append(f"{(Va @ Vb.T).max():>9.3f}")
    print(f"{a:>8}" + "".join(f"{x:>9}" if not x.startswith(" ") else x for x in row))
