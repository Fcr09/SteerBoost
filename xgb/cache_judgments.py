#!/usr/bin/env python3
"""
Build a single JSON judgment cache from data/result/{method}/{llm}/ to speed up
alphasearch.py.

Example:
  python script/cache_judgments.py --steering_method diffmean --llm gemma-2-2b-it --layer 10
  python script/cache_judgments.py ... --include_token_stats --output data/cache/judgments.json

Re-run with --force to overwrite an existing cache.
"""

import argparse
import json
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils import (  # noqa: E402
    build_judgment_cache_payload,
    iter_concept_ids_in_result_dir,
)


def main():
    p = argparse.ArgumentParser(description="Build judgment JSON cache for AlphaSearch.")
    p.add_argument("--steering_method", type=str, required=True)
    p.add_argument("--llm", type=str, required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument(
        "--result_dir",
        type=str,
        default=None,
        help="Defaults to data/result/{method}/{llm} under repo root.",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Defaults to data/cache/judgments__{method}__{llm}__L{layer}.json",
    )
    p.add_argument(
        "--concept_ids",
        type=str,
        default=None,
        help="Comma-separated concept ids; default = all numeric dirs under result_dir.",
    )
    p.add_argument(
        "--include_token_stats",
        action="store_true",
        help="Compute avg_response_tokens (loads tokenizer; slower).",
    )
    p.add_argument("--force", action="store_true", help="Overwrite existing output file.")
    args = p.parse_args()

    result_dir = args.result_dir
    if result_dir is None:
        result_dir = os.path.join(
            ROOT_DIR, "data", "result", args.steering_method, args.llm
        )

    if not os.path.isdir(result_dir):
        print(f"ERROR: result_dir not found: {result_dir}")
        sys.exit(1)

    out_path = args.output
    if out_path is None:
        cache_dir = os.path.join(ROOT_DIR, "data", "cache")
        os.makedirs(cache_dir, exist_ok=True)
        out_path = os.path.join(
            cache_dir,
            f"judgments__{args.steering_method}__{args.llm}__L{args.layer}.json",
        )

    if os.path.exists(out_path) and not args.force:
        print(f"ERROR: {out_path} exists. Use --force to overwrite.")
        sys.exit(1)

    concept_ids = None
    if args.concept_ids:
        concept_ids = [int(x.strip()) for x in args.concept_ids.split(",") if x.strip()]
    else:
        concept_ids = iter_concept_ids_in_result_dir(result_dir)
        print(f"Discovered {len(concept_ids)} concept directories under {result_dir}")

    print(f"Building cache (layer={args.layer}, concepts={len(concept_ids)})...")
    payload = build_judgment_cache_payload(
        result_dir=result_dir,
        layer=args.layer,
        steering_method=args.steering_method,
        llm=args.llm,
        concept_ids=concept_ids,
        include_token_stats=args.include_token_stats,
    )

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {out_path}")
    if payload.get("avg_response_tokens") is not None:
        print(f"  avg_response_tokens={payload['avg_response_tokens']:.2f} "
              f"(n={payload.get('response_token_count', 0)})")


if __name__ == "__main__":
    main()
