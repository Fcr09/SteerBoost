#!/usr/bin/env python
"""Build per-(method, model) tarballs and upload the SteerBoost dataset to HF.

Run from a machine with read access to the raw `data/` tree (e.g. nexus-scratch
in the original research repo). The dataset card from `tools/dataset_card.md`
is uploaded as the repo README.

Layout produced on the Hub (mirrors what scripts/download_data.py expects):

    SteerBoost-data/
    ├── README.md
    ├── concepts.json
    ├── alpaca_eval.json
    ├── cache/                                   # ~1.9 GB
    │   ├── features__{method}__{model}__L{layer}.npz
    │   ├── features__{method}__{model}__L{layer}_meta.json
    │   └── judgments__{method}__{model}__L{layer}.json
    ├── training/{cid}.json                      # ~10 MB
    └── result__{method}__{model}.tar.zst        # ~1–2 GB total compressed

Each result__*.tar.zst extracts to `result/{method}/{model}/{cid}/{layer}-{alpha}.json`
when unpacked with `tar -C <data_dir> --use-compress-program=unzstd -xf <file>`.

Usage:
    export HF_TOKEN=hf_...
    python tools/upload_data.py \\
        --src /fs/nexus-scratch/cfan42/SteerPred/data \\
        --repo-id Fcr09/SteerBoost-data \\
        --staging /fs/cml-scratch/cfan42/SteerBoost-staging

Add --skip-build to reuse already-built tarballs in staging/, or --skip-upload
to stop after building. --dry-run prints actions without doing anything.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo

DEFAULT_REPO_ID = "Fcr09/SteerBoost-data"
DEFAULT_SRC = "/fs/nexus-scratch/cfan42/SteerPred/data"
DEFAULT_STAGING = "/fs/cml-scratch/cfan42/SteerBoost-staging"

DATASET_CARD_PATH = Path(__file__).resolve().parent / "dataset_card.md"


def _run(cmd: list[str], dry_run: bool) -> None:
    print("[run]", " ".join(cmd))
    if not dry_run:
        subprocess.check_call(cmd)


def build_result_tarballs(src: Path, staging: Path, dry_run: bool) -> list[Path]:
    """One tar.zst per (method, model) directory under src/result/."""
    result_root = src / "result"
    if not result_root.is_dir():
        print(f"[warn] {result_root} not found; skipping result tarballs.")
        return []

    out: list[Path] = []
    for method_dir in sorted(p for p in result_root.iterdir() if p.is_dir()):
        for model_dir in sorted(p for p in method_dir.iterdir() if p.is_dir()):
            method = method_dir.name
            model = model_dir.name
            rel = f"result/{method}/{model}"
            tarball = staging / f"result__{method}__{model}.tar.zst"
            if tarball.exists():
                print(f"[skip] {tarball.name} already exists")
                out.append(tarball)
                continue
            print(f"[tar]  {rel} -> {tarball.name}")
            _run(
                [
                    "tar",
                    "--use-compress-program=zstd -T0 -19",
                    "-cf", str(tarball),
                    "-C", str(src),
                    rel,
                ],
                dry_run=dry_run,
            )
            out.append(tarball)
    return out


def stage_small_assets(src: Path, staging: Path, dry_run: bool) -> None:
    """Symlink cache/, training/, concepts.json, alpaca_eval.json into staging/."""
    items: list[tuple[Path, Path]] = [
        (src / "cache", staging / "cache"),
        (src / "training", staging / "training"),
    ]
    # concepts.json + alpaca_eval.json live in supplementary/data/ in the public
    # tree; fall back to src/ if they aren't there.
    supp_data = Path(__file__).resolve().parent.parent / "data"
    for fname in ("concepts.json", "alpaca_eval.json"):
        candidate = supp_data / fname if (supp_data / fname).exists() else src / fname
        items.append((candidate, staging / fname))

    for source, target in items:
        if not source.exists():
            print(f"[warn] missing {source}, skipping")
            continue
        if target.exists() or target.is_symlink():
            print(f"[skip] {target} already staged")
            continue
        print(f"[link] {target} -> {source}")
        if not dry_run:
            os.symlink(source, target)


def stage_dataset_card(staging: Path, dry_run: bool) -> None:
    if not DATASET_CARD_PATH.exists():
        print(f"[warn] {DATASET_CARD_PATH} not found; skipping README")
        return
    target = staging / "README.md"
    print(f"[copy] {DATASET_CARD_PATH} -> {target}")
    if not dry_run:
        shutil.copyfile(DATASET_CARD_PATH, target)


def upload(repo_id: str, staging: Path, private: bool, token: str | None, dry_run: bool) -> None:
    print(f"[hf] create_repo {repo_id} (dataset, exist_ok=True, private={private})")
    if not dry_run:
        create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True,
            token=token,
        )

    print(f"[hf] upload_large_folder {staging} -> {repo_id}")
    if not dry_run:
        api = HfApi(token=token)
        api.upload_large_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(staging),
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=DEFAULT_SRC, help="source data/ directory")
    parser.add_argument("--staging", default=DEFAULT_STAGING)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    src = Path(args.src).resolve()
    staging = Path(args.staging).resolve()
    staging.mkdir(parents=True, exist_ok=True)

    if not args.skip_build:
        build_result_tarballs(src, staging, args.dry_run)
        stage_small_assets(src, staging, args.dry_run)
        stage_dataset_card(staging, args.dry_run)

    if not args.skip_upload:
        if not args.token and not args.dry_run:
            print("ERROR: HF_TOKEN not set. `export HF_TOKEN=hf_...` first.", file=sys.stderr)
            return 1
        upload(args.repo_id, staging, args.private, args.token, args.dry_run)

    print("[done]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
