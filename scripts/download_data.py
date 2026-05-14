#!/usr/bin/env python
"""Download the SteerBoost dataset from the Hugging Face Hub into ./data/.

The dataset ships:
  cache/                 # feature .npz + judgment .json caches per (method, model, layer)
  result__*.tar.zst      # per (method, model) tarball of steered generations + GPT labels
  training/              # GPT-generated positive/negative example banks per concept

Hidden-state .pt files are NOT distributed (>500 GB on disk). Reproducing them
requires running scripts/01_run_steering.sh and scripts/02_extract_raw_hidden_states.sh.

Usage:
    python scripts/download_data.py                       # everything
    python scripts/download_data.py --no-result           # skip the 7.5GB result tarballs
    python scripts/download_data.py --repo-id ORG/NAME    # override default HF repo
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

DEFAULT_REPO_ID = "Fcr09/SteerBoost-data"
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _extract_tarballs(data_dir: Path) -> None:
    for tarball in sorted(data_dir.glob("result__*.tar.zst")):
        print(f"[extract] {tarball.name} -> {data_dir}/result/")
        subprocess.check_call(
            ["tar", "--use-compress-program=unzstd", "-xf", str(tarball), "-C", str(data_dir)]
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--no-result", action="store_true", help="Skip the result/ tarballs.")
    parser.add_argument("--no-extract", action="store_true", help="Do not extract tarballs.")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    allow = ["cache/*", "training/*", "concepts.json", "alpaca_eval.json"]
    if not args.no_result:
        allow.append("result__*.tar.zst")

    print(f"[hf] snapshot_download repo_id={args.repo_id} -> {data_dir}")
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(data_dir),
        local_dir_use_symlinks=False,
        allow_patterns=allow,
        token=args.token,
    )

    if not args.no_result and not args.no_extract:
        _extract_tarballs(data_dir)

    print("[done]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
