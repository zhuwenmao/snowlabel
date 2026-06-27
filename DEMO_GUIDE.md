# SnowLabel — Demo & Screen-Recording Guide

Everything needed to record the **click → mask → snowball → auto-label** demo.
The backend is fully verified (see *Backend already proven*); this guide covers
the browser steps that need a human + a screen recorder.

English | [中文](DEMO_GUIDE_CN.md)

## 1. Start everything

```powershell
cd <repo-root>
powershell -ExecutionPolicy Bypass -File start_demo.ps1
```

> The `cd` matters — the script lives in the project folder, not your home
> directory. Or run it by full path:
> `powershell -ExecutionPolicy Bypass -File <repo-root>\start_demo.ps1`

Three windows open — ML backend (:9090), image server (:8081), Label Studio
(:8080). Wait ~40 s for the backend window to print *Application startup
complete* (it loads DINOv3-L). Check it:

```powershell
irm http://localhost:9090/stats     # shows models, backend, thresholds
```

## 2. Open Label Studio

Browser → http://localhost:8080 → login `demo@snowlabel.ai` / `snowlabel123` →
open project **SnowLabel Demo**. The 20 tasks are grouped by class
(cat×4, dog×4, …) so labeling 4 of one class shows the whole snowball.

> Keep the Label Studio window in the **foreground** while recording — screen
> recorders (and the CDP screenshot API) only capture a visible window.

## 3. Record the snowball (the money shot)

Label the four cats in a row:

1. Click **Label All Tasks** (or open task 1).
2. Turn on **Auto-Annotation** (bottom-left toggle).
3. Select the **KeyPoint** tool — it's the smart trigger (`smart="true"`).
4. **Click on the object.** SAM segments it; the snowball memory predicts a class:
   - cat #1 → **silent** (memory empty) — pick `cat` yourself → **Submit**
   - cat #2–3 → **suggest** (low support) — confirm `cat` → **Submit**
   - cat #4 → **auto** (support ≥ 3) — class pre-filled `cat` → just **Submit**
5. That **silent → suggest → auto** progression *is* the snowball. Record it.
   The Regions panel and the Choices radio show the predicted class + confidence.

(Repeat with dog/bird/… if you want a longer clip.)

## 4. Show SAM 3 concept batch recall

Once memory knows "cat", recall **every** cat in an image in one call:

```powershell
$b = @{ tasks=@(@{data=@{image="http://localhost:8081/<a-cat-image>.jpg"}}); labels=@("cat"); score_threshold=0.4 } | ConvertTo-Json -Depth 6
irm http://localhost:9090/preannotate -Method Post -Body $b -ContentType "application/json"
```

It returns a brush mask for every detected cat (`per_label_counts` in the meta).

## 5. Show the Memory Inspector

Browser → http://localhost:9090/inspector — a live view of what the snowball has
learned: class-distribution bars + every stored entry. Refresh while you label
to watch it grow.

## Backend already proven (no browser needed)

| What | How | Result |
|------|-----|--------|
| End-to-end snowball, **real SAM masks** | `scripts/demo_snowball_e2e.py` | 86.0% overall, 93.4% late-half, **auto-tier 95% over 80% of clicks** — `results/demo_e2e/snowball_e2e.png` |
| LS protocol layer | `scripts/test_protocol.py` | prompt parsing + result conversion all pass |
| SAM 3 batch recall | `scripts/sam3_batch_demo.py` | verified across 5 classes |

## Troubleshooting

- **Click only drops a keypoint, no mask** — make sure Auto-Annotation is ON and
  the KeyPoint (smart) tool is selected; the backend window should log
  `Interactive predict: point`, not `Auto predict (no prompt)`.
- **Everything slow / endless spinner** — the backend got hit with auto-predict
  for every task. It's disabled (`evaluate_predictions_automatically=false`) in the
  provided LS config; if it returns, turn it off in the LS project settings.
- **Recording/screenshot blank** — the LS window must be visible & foreground.
- **Stop all services**:
  ```powershell
  Get-NetTCPConnection -LocalPort 9090,8081,8080 -State Listen | %{ Stop-Process -Id $_.OwningProcess -Force }
  ```
