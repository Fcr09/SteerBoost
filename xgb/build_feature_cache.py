#!/usr/bin/env python3
"""
Build a compressed .npz feature matrix (+ sidecar meta.json) from hidden-state
.pt pairs via utils.load_paired_data and xgb.train.extract_features.

Use with:
  train_weighted_search.py --features_cache /path/to/cache.npz
  alphasearch.py --feature_cache /path/to/cache.npz
  steerability.py --feature_cache /path/to/cache.npz

Example:
  python script/build_feature_cache.py --steering_method diffmean --llm gemma-2-2b-it \\
    --curr_layer 10 --output data/cache/features_diffmean_gemma_l10.npz
"""

import argparse
import importlib.util
import json
import os
import sys

import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils import load_paired_data  # noqa: E402
from xgb.feature_cache import (  # noqa: E402
    default_meta_path,
    save_feature_cache,
)

_train_path = os.path.join(ROOT_DIR, "xgb", "train.py")
_spec = importlib.util.spec_from_file_location("xgb_train", _train_path)
_train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_train)
extract_features = _train.extract_features
build_label_map = _train.build_label_map

_cp_path = os.path.join(ROOT_DIR, "xgb", "concept_partition.py")
_spec_cp = importlib.util.spec_from_file_location("concept_partition", _cp_path)
_cp = importlib.util.module_from_spec(_spec_cp)
_spec_cp.loader.exec_module(_cp)
build_static_partition = _cp.build_static_partition
build_stratified_shuffle_partition = _cp.build_stratified_shuffle_partition

_xgb_utils_path = os.path.join(ROOT_DIR, "xgb", "utils.py")
_spec_x = importlib.util.spec_from_file_location("xgb_utils", _xgb_utils_path)
_xgb_u = importlib.util.module_from_spec(_spec_x)
_spec_x.loader.exec_module(_xgb_u)
ALL_FEATURE_GROUPS = _xgb_u.ALL_FEATURE_GROUPS


def main():
    p = argparse.ArgumentParser(description="Build handcrafted feature .npz cache for XGB pipelines.")
    p.add_argument("--steering_method", type=str, default="diffmean")
    p.add_argument("--llm", type=str, required=True)
    p.add_argument("--curr_layer", type=int, default=10)
    p.add_argument("--k_list", type=str, default="1,2,3,5,10,15")
    p.add_argument("--target_token_n_list", type=str, default="1,3,5")
    p.add_argument("--alpha_min", type=float, default=None)
    p.add_argument("--alpha_max", type=float, default=None)
    p.add_argument("--id_concept_max", type=int, default=119)
    p.add_argument("--ood_concept_min", type=int, default=120)
    p.add_argument("--ood_concept_max", type=int, default=149)
    p.add_argument(
        "--id_ood_partition",
        type=str,
        choices=("stratified_shuffle", "static_ranges"),
        default="stratified_shuffle",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size_raw", type=int, default=8)
    p.add_argument("--force_regenerate_raw", action="store_true")
    p.add_argument(
        "--feature_groups",
        type=str,
        default=None,
        help="Comma-separated; default: all groups.",
    )
    p.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output .npz (meta written alongside as <stem>_meta.json).",
    )
    p.add_argument("--force", action="store_true", help="Overwrite existing output.")
    args = p.parse_args()

    k_list = [int(x.strip()) for x in args.k_list.split(",")]
    n_list = [int(x.strip()) for x in args.target_token_n_list.split(",")]
    feature_groups = (
        [g.strip() for g in args.feature_groups.split(",")]
        if args.feature_groups
        else None
    )

    if args.id_ood_partition == "stratified_shuffle":
        id_concepts, ood_concepts, _cp_meta = build_stratified_shuffle_partition(
            args.seed
        )
    else:
        id_concepts, ood_concepts, _cp_meta = build_static_partition(
            args.id_concept_max, args.ood_concept_min, args.ood_concept_max
        )

    all_cids = list(id_concepts) + list(ood_concepts)

    out_npz = args.output
    meta_path = default_meta_path(out_npz)
    if os.path.exists(out_npz) and not args.force:
        print(f"ERROR: {out_npz} exists. Use --force.")
        sys.exit(1)

    print(f"Loading paired data for {len(all_cids)} concepts...")
    pairs, labels, meta = load_paired_data(
        method=args.steering_method,
        model=args.llm,
        layer=args.curr_layer,
        n_list=n_list,
        k_list=k_list,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        concept_ids=all_cids,
        force_regenerate_raw=args.force_regenerate_raw,
        return_metadata=True,
        batch_size_raw=args.batch_size_raw,
    )
    if not pairs:
        print("ERROR: No samples loaded.")
        sys.exit(1)

    label_map = build_label_map(labels)
    X, y, idx_kept = extract_features(
        pairs,
        labels,
        label_map,
        args.curr_layer,
        k_list,
        n_list,
        metadata=meta,
        feature_groups=feature_groups,
        return_valid_indices=True,
    )

    meta_kept = [meta[int(i)] for i in idx_kept]
    concept_id = np.array([m[0] for m in meta_kept], dtype=np.int32)
    alpha = np.array([m[1] for m in meta_kept], dtype=np.float32)
    sample_id = np.array([m[2] for m in meta_kept], dtype=np.int32)
    n = len(y)

    meta_payload = {
        "steering_method": args.steering_method,
        "llm": args.llm,
        "curr_layer": args.curr_layer,
        "k_list": k_list,
        "target_token_n_list": n_list,
        "alpha_min": args.alpha_min,
        "alpha_max": args.alpha_max,
        "feature_groups": feature_groups or list(ALL_FEATURE_GROUPS),
        "id_ood_partition": args.id_ood_partition,
        "seed": args.seed,
        "label_map": {str(k): int(v) for k, v in label_map.items()},
        "num_rows": int(n),
        "feature_dim": int(X.shape[1]),
    }

    save_feature_cache(
        out_npz,
        meta_path,
        X,
        y,
        concept_id,
        alpha,
        sample_id,
        meta_payload,
    )
    print(f"Wrote {out_npz} ({n} rows, {X.shape[1]} features)")
    print(f"Meta: {meta_path}")


if __name__ == "__main__":
    main()
