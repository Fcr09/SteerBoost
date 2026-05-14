"""
Precomputed handcrafted feature matrices (.npz + meta.json) to avoid repeated
torch.load of hidden-state .pt files in training and XGB inference.
"""

from __future__ import annotations

import importlib.util
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

FEATURE_CACHE_VERSION = 1

# Lazy-load build_feature_names (avoids circular import / heavy deps at import time)
_build_feature_names = None

def _get_build_feature_names():
    global _build_feature_names
    if _build_feature_names is None:
        _train_path = os.path.join(os.path.dirname(__file__), "train.py")
        _spec = importlib.util.spec_from_file_location("xgb_train", _train_path)
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        _build_feature_names = _mod.build_feature_names
    return _build_feature_names


def subset_feature_columns(
    cache_feature_groups: List[str],
    desired_feature_groups: List[str],
    curr_layer: int,
    k_list: List[int],
    n_list: List[int],
    include_alpha: bool = True,
) -> Optional[np.ndarray]:
    """
    Return column indices to select from a cache built with `cache_feature_groups`
    when only `desired_feature_groups` are wanted.

    Returns None if the sets are identical (no subsetting needed).
    Raises ValueError if the desired groups are not a subset of cache groups.
    """
    if set(desired_feature_groups) == set(cache_feature_groups):
        return None

    if not set(desired_feature_groups).issubset(set(cache_feature_groups)):
        extra = set(desired_feature_groups) - set(cache_feature_groups)
        raise ValueError(
            f"Desired feature groups {extra} not present in cache "
            f"(cache has {cache_feature_groups})"
        )

    bfn = _get_build_feature_names()
    cache_names = bfn(curr_layer, k_list, n_list,
                      include_alpha=include_alpha,
                      feature_groups=cache_feature_groups)
    desired_names = bfn(curr_layer, k_list, n_list,
                        include_alpha=include_alpha,
                        feature_groups=desired_feature_groups)

    name_to_idx = {name: i for i, name in enumerate(cache_names)}
    indices = []
    for name in desired_names:
        if name not in name_to_idx:
            raise ValueError(
                f"Feature '{name}' expected in cache but not found. "
                f"Cache has {len(cache_names)} features."
            )
        indices.append(name_to_idx[name])

    return np.array(indices, dtype=np.int64)


def _legacy_afterdiv_removed_count(n_list: List[int]) -> int:
    """Columns removed when dropping legacy AfterDiv_* features."""
    token_indices = sorted(list(set([0] + list(n_list))))
    n_tokens = len(token_indices)
    # Legacy AfterSteering included:
    #   InterventionMag (1)
    #   AfterDiv_MagDiff: n_tokens + 4 stats
    #   AfterDiv_CosSim:  n_tokens + 4 stats
    # New schema keeps only InterventionMag.
    return 2 * (n_tokens + 4)


def resolve_feature_columns(
    cache_feature_groups: List[str],
    desired_feature_groups: List[str],
    curr_layer: int,
    k_list: List[int],
    n_list: List[int],
    cache_feature_dim: int,
    include_alpha: bool = True,
) -> Optional[np.ndarray]:
    """
    Resolve cache column indices for:
    1) strict group subsetting, and
    2) backward-compat old caches that still contain AfterDiv_* columns.
    """
    bfn = _get_build_feature_names()
    desired_names = bfn(
        curr_layer, k_list, n_list,
        include_alpha=include_alpha,
        feature_groups=desired_feature_groups,
    )
    expected_dim = len(desired_names)

    # Fast path: dimensions already match desired schema.
    if cache_feature_dim == expected_dim and set(cache_feature_groups) == set(desired_feature_groups):
        return None

    # Legacy compatibility: same groups but old AfterDiv columns still present.
    legacy_delta = _legacy_afterdiv_removed_count(n_list)
    if (
        cache_feature_dim == expected_dim + legacy_delta
        and set(cache_feature_groups) == set(desired_feature_groups)
        and "AfterSteering" in desired_feature_groups
        and include_alpha
    ):
        if cache_feature_dim <= legacy_delta + 1:
            raise ValueError(
                f"Invalid cache feature_dim={cache_feature_dim} for legacy AfterDiv trimming."
            )
        keep_prefix = cache_feature_dim - legacy_delta - 1
        keep_idx = list(range(keep_prefix)) + [cache_feature_dim - 1]  # keep trailing Alpha
        return np.array(keep_idx, dtype=np.int64)

    if set(cache_feature_groups) == set(desired_feature_groups):
        raise ValueError(
            f"Cache feature_dim={cache_feature_dim} does not match expected_dim={expected_dim} "
            "for the active feature schema. Rebuild cache or run the cache migration script."
        )

    # Generic subset path (groups differ). Requires cache schema to match current naming.
    return subset_feature_columns(
        cache_feature_groups=cache_feature_groups,
        desired_feature_groups=desired_feature_groups,
        curr_layer=curr_layer,
        k_list=k_list,
        n_list=n_list,
        include_alpha=include_alpha,
    )


def default_meta_path(npz_path: str) -> str:
    base, _ = os.path.splitext(npz_path)
    return base + "_meta.json"


def save_feature_cache(
    npz_path: str,
    meta_path: str,
    X: np.ndarray,
    y: np.ndarray,
    concept_id: np.ndarray,
    alpha: np.ndarray,
    sample_id: np.ndarray,
    meta: Dict[str, Any],
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(npz_path)) or ".", exist_ok=True)
    np.savez_compressed(
        npz_path,
        X=X.astype(np.float32),
        y=y.astype(np.int64),
        concept_id=concept_id.astype(np.int32),
        alpha=alpha.astype(np.float32),
        sample_id=sample_id.astype(np.int32),
    )
    payload = dict(meta)
    payload["format_version"] = FEATURE_CACHE_VERSION
    with open(meta_path, "w") as f:
        json.dump(payload, f, indent=2)


def load_feature_cache(
    npz_path: str,
    meta_path: Optional[str] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    if meta_path is None:
        meta_path = default_meta_path(npz_path)
    raw = np.load(npz_path)
    bundle = {k: raw[k] for k in raw.files}
    with open(meta_path) as f:
        meta = json.load(f)
    if int(meta.get("format_version", 0)) != FEATURE_CACHE_VERSION:
        raise ValueError(f"Unsupported feature cache version in {meta_path}")
    return bundle, meta


def _align_alpha(a: float, alphas: List[float]) -> Optional[float]:
    for x in alphas:
        if abs(float(x) - float(a)) < 1e-5:
            return float(x)
    return None


def setting4_ours_ranked_from_cache(
    judgments: Dict,
    alphas: List[float],
    concept_ids: List[int],
    test_prompts: Dict[int, List[int]],
    bundle: Dict[str, np.ndarray],
    model,
    feat_mean: Optional[np.ndarray] = None,
    feat_std: Optional[np.ndarray] = None,
) -> Dict:
    """
    Same contract as alphasearch.setting4_ours_ranked, using precomputed rows.
    """
    X_all = bundle["X"]
    c_all = bundle["concept_id"]
    a_all = bundle["alpha"]
    s_all = bundle["sample_id"]

    ranked: Dict = {}
    skipped = 0
    total = 0
    n_concepts = len(concept_ids)

    for ci, cid in enumerate(concept_ids):
        if cid not in judgments or not judgments[cid]:
            continue
        tps = test_prompts.get(cid, [])
        if not tps:
            continue

        ranked[cid] = {}
        prompt_alpha_probs = defaultdict(dict)

        for alpha in alphas:
            if alpha not in judgments[cid]:
                continue
            a_key = _align_alpha(alpha, list(judgments[cid].keys()))
            if a_key is None:
                a_key = alpha
            X_batch: List[np.ndarray] = []
            batch_pids: List[int] = []
            for pidx in tps:
                sel = np.where(
                    (c_all == cid)
                    & (np.abs(a_all - float(a_key)) < 1e-4)
                    & (s_all == int(pidx))
                )[0]
                if len(sel) == 0:
                    continue
                i = int(sel[0])
                X_batch.append(X_all[i])
                batch_pids.append(pidx)

            if not X_batch:
                continue
            X = np.array(X_batch, dtype=np.float32)
            n_expected = model.n_features_in_
            if X.shape[1] > n_expected:
                X = X[:, :n_expected]
            elif X.shape[1] < n_expected:
                raise ValueError(
                    f"Feature count mismatch: model expects {n_expected}, "
                    f"cache has {X.shape[1]}"
                )
            if feat_mean is not None and feat_std is not None:
                X = (X - feat_mean) / feat_std
            probs = model.predict_proba(X)
            for i, pidx in enumerate(batch_pids):
                prompt_alpha_probs[pidx][a_key] = float(probs[i, 1])

        for pidx in tps:
            total += 1
            if pidx not in prompt_alpha_probs or not prompt_alpha_probs[pidx]:
                skipped += 1
                continue
            ranked[cid][pidx] = sorted(
                prompt_alpha_probs[pidx].items(),
                key=lambda x: x[1],
                reverse=True,
            )

        if (ci + 1) % 20 == 0 or (ci + 1) == n_concepts:
            print(f"  [Ours-cache] Processed {ci + 1}/{n_concepts} concepts...")

    print(f"  [Ours-cache] Total: {total}, Skipped: {skipped}")
    return ranked


def compute_predicted_steerability_from_cache(
    concept_ids: List[int],
    alphas: List[float],
    num_prompts: int,
    bundle: Dict[str, np.ndarray],
    model,
    feat_mean: Optional[np.ndarray] = None,
    feat_std: Optional[np.ndarray] = None,
) -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Same return shape as steerability.compute_predicted_steerability, from cache.
    """
    X_all = bundle["X"]
    c_all = bundle["concept_id"]
    a_all = bundle["alpha"]
    s_all = bundle["sample_id"]

    p_success = defaultdict(lambda: defaultdict(dict))
    n_concepts = len(concept_ids)

    for ci, cid in enumerate(concept_ids):
        for alpha in alphas:
            sel = np.where(
                (c_all == cid) & (np.abs(a_all - float(alpha)) < 1e-4)
            )[0]
            if len(sel) == 0:
                continue
            X_rows = []
            pidx_list = []
            for i in sel:
                pidx = int(s_all[i])
                if pidx >= num_prompts:
                    continue
                X_rows.append(X_all[i])
                pidx_list.append(pidx)
            if not X_rows:
                continue
            X = np.array(X_rows, dtype=np.float32)
            n_expected = model.n_features_in_
            if X.shape[1] > n_expected:
                X = X[:, :n_expected]
            elif X.shape[1] < n_expected:
                raise ValueError(
                    f"Feature count mismatch: model expects {n_expected}, "
                    f"cache has {X.shape[1]}"
                )
            if feat_mean is not None and feat_std is not None:
                X = (X - feat_mean) / feat_std
            probs = model.predict_proba(X)
            for i, pidx in enumerate(pidx_list):
                p_success[cid][pidx][float(alpha)] = float(probs[i, 1])

        if (ci + 1) % 20 == 0 or (ci + 1) == n_concepts:
            print(f"  [Pred-cache] Processed {ci + 1}/{n_concepts} concepts...")

    prompt_ids = list(range(num_prompts))

    concept_max = {}
    concept_mean = {}
    for cid in concept_ids:
        if cid not in p_success or not p_success[cid]:
            continue
        per_prompt_max = []
        per_prompt_mean = []
        for pid in prompt_ids:
            if pid not in p_success[cid] or not p_success[cid][pid]:
                continue
            probs_list = list(p_success[cid][pid].values())
            per_prompt_max.append(max(probs_list))
            per_prompt_mean.append(np.mean(probs_list))
        if per_prompt_max:
            concept_max[cid] = np.mean(per_prompt_max)
            concept_mean[cid] = np.mean(per_prompt_mean)

    prompt_max = {}
    prompt_mean = {}
    for pid in prompt_ids:
        per_concept_max = []
        per_concept_mean = []
        for cid in concept_ids:
            if pid not in p_success.get(cid, {}):
                continue
            if not p_success[cid][pid]:
                continue
            probs_list = list(p_success[cid][pid].values())
            per_concept_max.append(max(probs_list))
            per_concept_mean.append(np.mean(probs_list))
        if per_concept_max:
            prompt_max[pid] = np.mean(per_concept_max)
            prompt_mean[pid] = np.mean(per_concept_mean)

    return concept_max, concept_mean, prompt_max, prompt_mean
