"""Helper: set up gated Meta weights for SAM 3 + DINOv3.

Both models are gated behind Meta access approvals on HuggingFace. This
script walks you through the steps and verifies each one — it does NOT
bypass the approvals. You still need to click "agree" on Meta's pages.

Usage:
    python scripts/setup_sam3_dinov3.py            # interactive walkthrough
    python scripts/setup_sam3_dinov3.py --check    # just verify current state
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

SAM3_HF_PAGE = "https://huggingface.co/facebook/sam3"
SAM3_WEIGHT_NAME = "sam3.pt"
DINOV3_HF_REPO = "facebook/dinov3-vitb16-pretrain-lvd1689m"
DINOV3_HF_PAGE = f"https://huggingface.co/{DINOV3_HF_REPO}"


def step(num: int, total: int, msg: str) -> None:
    print(f"\n[{num}/{total}] {msg}")


def check_hf_login() -> tuple[bool, str]:
    """Return (logged_in, username_or_message)."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        whoami = api.whoami()
        return True, whoami.get("name", whoami.get("email", "<unknown>"))
    except ImportError:
        return False, "huggingface_hub not installed"
    except Exception as e:
        return False, str(e)


def check_sam3_weights() -> Path | None:
    """Return Path to sam3.pt if found locally, else None."""
    candidates = [
        PROJECT_ROOT / SAM3_WEIGHT_NAME,
        Path.cwd() / SAM3_WEIGHT_NAME,
        PROJECT_ROOT / "data" / "weights" / SAM3_WEIGHT_NAME,
    ]
    for c in candidates:
        if c.is_file() and c.stat().st_size > 1024 * 1024:  # > 1 MB sanity
            return c.resolve()
    return None


def check_dinov3_cached() -> bool:
    """Return True if DINOv3 weights are in the HF cache."""
    try:
        from huggingface_hub import scan_cache_dir
        cache = scan_cache_dir()
        for repo in cache.repos:
            if repo.repo_id == DINOV3_HF_REPO and any(
                r.size_on_disk > 1024 * 1024 for r in repo.revisions
            ):
                return True
    except ImportError:
        return False
    except Exception:
        return False
    return False


def cmd_check() -> int:
    ok, who = check_hf_login()
    print(f"  HuggingFace login : {'YES (' + who + ')' if ok else 'NO (' + who + ')'}")

    sam3 = check_sam3_weights()
    print(f"  SAM 3 weights     : {'YES (' + str(sam3) + ')' if sam3 else 'NO'}")

    dinov3 = check_dinov3_cached()
    print(f"  DINOv3 cached     : {'YES' if dinov3 else 'NO'}")

    everything_ok = ok and sam3 and dinov3
    if everything_ok:
        print("\n  ALL SET. To use, set:")
        print(f"    $env:SNOWLABEL_SAM = '{sam3}'")
        print(f"    $env:SNOWLABEL_DINO = 'dinov3_vitb16'")
        return 0
    else:
        print("\n  Some pieces missing. Run without --check for the walkthrough.")
        return 1


def cmd_walkthrough() -> int:
    total = 5

    # --- Step 1: huggingface_hub installed?
    step(1, total, "Verify huggingface_hub is installed")
    try:
        import huggingface_hub  # noqa
        print("  OK: huggingface_hub", huggingface_hub.__version__)
    except ImportError:
        print("  Installing huggingface_hub ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "--quiet", "huggingface_hub"])

    # --- Step 2: HF login
    step(2, total, "HuggingFace login")
    ok, who = check_hf_login()
    if ok:
        print(f"  OK: logged in as {who}")
    else:
        print(f"  NOT LOGGED IN ({who})")
        print()
        print("  Get a token from https://huggingface.co/settings/tokens")
        print("  (READ scope is enough), then run:")
        print("      huggingface-cli login")
        print("  paste the token and re-run this script.")
        return 1

    # --- Step 3: SAM 3 access approval
    step(3, total, "SAM 3 access approval + manual download")
    sam3 = check_sam3_weights()
    if sam3:
        print(f"  OK: found weights at {sam3}")
    else:
        print(f"  SAM 3 weights '{SAM3_WEIGHT_NAME}' not found.")
        print()
        print("  Manual steps:")
        print(f"    1. Open {SAM3_HF_PAGE}")
        print("    2. Click 'Agree and access repository' (Meta's gate).")
        print("    3. Download `sam3.pt` (the main checkpoint).")
        print(f"    4. Place the file in: {PROJECT_ROOT}")
        print("    5. Re-run this script.")
        return 1

    # --- Step 4: DINOv3 access approval
    step(4, total, "DINOv3 access approval + cache warm-up")
    if check_dinov3_cached():
        print("  OK: DINOv3 weights already in HF cache.")
    else:
        print(f"  Downloading DINOv3 weights from {DINOV3_HF_REPO} ...")
        print("  (If this 401s, open the HF page below and click 'Agree')")
        print(f"  -> {DINOV3_HF_PAGE}")
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=DINOV3_HF_REPO, allow_patterns=[
                "*.safetensors", "*.json", "*.txt"
            ])
            print("  OK: DINOv3 downloaded and cached.")
        except Exception as e:
            print(f"  FAILED: {e}")
            print()
            print(f"  Open {DINOV3_HF_PAGE}")
            print("  Click 'Agree and access repository', then re-run.")
            return 1

    # --- Step 5: Done
    step(5, total, "Ready to use!")
    sam3 = check_sam3_weights()
    print()
    print("  Set these env vars before starting the backend:")
    print()
    print("      # PowerShell")
    print(f"      $env:SNOWLABEL_SAM = '{sam3}'")
    print("      $env:SNOWLABEL_DINO = 'dinov3_vitb16'")
    print()
    print("      # bash")
    print(f"      export SNOWLABEL_SAM='{sam3}'")
    print("      export SNOWLABEL_DINO='dinov3_vitb16'")
    print()
    print("  Then:  python -m snowlabel.server")
    print()
    print("  NOTE: switching DINO variants invalidates the existing snowball")
    print("  memory (different embedding dim). To start fresh:")
    print("      Remove-Item -Recurse data\\memory      # PowerShell")
    print("      rm -rf data/memory                    # bash")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="just print current state, don't run walkthrough")
    args = ap.parse_args()
    if args.check:
        return cmd_check()
    return cmd_walkthrough()


if __name__ == "__main__":
    raise SystemExit(main())
