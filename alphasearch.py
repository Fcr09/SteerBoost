#!/usr/bin/env python3
"""
AlphaSearch: Early Prediction vs Grid Search Comparison

Compares alpha-selection strategies for activation steering using a binary
success metric: for a given prompt, did the method find an alpha that leads
to successful steering (judgment == 1)?

Baselines:
  TCGS  - Training-set Concept-level Grid Search (ID only, K=1 at test time)
  CGS   - Concept-level Grid Search Oracle (K=1 at test time)
  IGS   - Item-level Grid Search (fraction of prompts steerable by any alpha)
  IGS-A - Item-level Grid Search with Early Stop (Ascending alpha order)
  IGS-D - Item-level Grid Search with Early Stop (Descending alpha order)
  Ours  - ranked search guided by XGBoost P(success) predictions

For settings 4 and 5, we evaluate at multiple K (budget) values:
  K = how many full rollouts you are willing to spend per prompt.
  Success if ANY of the first K tried alphas gives judgment == 1.
  Cost = position of first success (1-indexed), or K if none succeed.

Usage:
  python alphasearch.py --llm Qwen3-1.7B --steering_method diffmean --layer 10
  # Defaults: alphasearch_out/<llm>/<steering_method>/results.json; plots: pareto.pdf
  # (no τ vs baselines) and, with --threshold_sweep, pareto_front.pdf (Ours-only).
  python alphasearch.py --skip_ours --output /path/to/custom.json

  # Regenerate figures from a prior results.json (no judgment / XGBoost / IO recomputation):
  python alphasearch.py --from-results alphasearch_out/<llm>/<steering_method>/results.json
  # Only plot selected budgets (must match or subset of K stored in the JSON):
  python alphasearch.py --from-results .../results.json --top_k_values 1,5,10
  # Optional: explicit figure path with --pareto_plot ...
"""

import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import importlib.util
from collections import defaultdict

root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from utils import load_all_judgments, load_judgments_from_cache_file

_xgb_utils_path = os.path.join(root_dir, "xgb", "utils.py")
_spec = importlib.util.spec_from_file_location("xgb_utils", _xgb_utils_path)
_xgb_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_xgb_utils)
get_handcrafted_features = _xgb_utils.get_handcrafted_features

_cp_path = os.path.join(root_dir, "xgb", "concept_partition.py")
_spec_cp = importlib.util.spec_from_file_location("concept_partition", _cp_path)
_cp = importlib.util.module_from_spec(_spec_cp)
_spec_cp.loader.exec_module(_cp)
load_concept_partition_from_splits_json = _cp.load_concept_partition_from_splits_json

_fc_path = os.path.join(root_dir, "xgb", "feature_cache.py")
_spec_fc = importlib.util.spec_from_file_location("xgb_feature_cache", _fc_path)
_fc = importlib.util.module_from_spec(_spec_fc)
_spec_fc.loader.exec_module(_fc)
load_feature_cache = _fc.load_feature_cache
resolve_feature_columns = _fc.resolve_feature_columns
setting4_ours_ranked_from_cache = _fc.setting4_ours_ranked_from_cache


def id_ood_display_captions(cp_meta, args):
    """Table titles for ID vs OOD blocks; uses saved partition when present."""
    if cp_meta is None:
        return (
            f"In-Distribution (concepts 0-{args.id_concept_max})",
            f"OOD (concepts {args.ood_concept_min}-{args.ood_concept_max})",
        )
    scheme = cp_meta.get("scheme")
    if scheme == "stratified_shuffle":
        sd = cp_meta.get("seed")
        return (
            f"In-Distribution (120 concepts, stratified_shuffle, seed={sd})",
            f"OOD (30 concepts, stratified_shuffle, seed={sd})",
        )
    if scheme == "static_ranges":
        return (
            f"In-Distribution (concepts 0-{cp_meta.get('id_concept_max', args.id_concept_max)})",
            f"OOD (concepts {cp_meta.get('ood_concept_min')}-"
            f"{cp_meta.get('ood_concept_max')})",
        )
    n_id = len(cp_meta.get("id_concepts") or [])
    n_ood = len(cp_meta.get("ood_concepts") or [])
    return (
        f"In-Distribution ({n_id} concepts)",
        f"OOD ({n_ood} concepts)",
    )


def check_hidden_states_data(hidden_states, n=None):
    if 'token_0' not in hidden_states:
        return False
    if 'before_steering' not in hidden_states['token_0']:
        return False
    if 'after_steering' not in hidden_states['token_0']:
        return False
    if not hidden_states['token_0']['before_steering']:
        return False
    if not hidden_states['token_0']['after_steering']:
        return False
    if n is not None:
        tk = f'token_{n}'
        if tk not in hidden_states:
            return False
        if 'before_steering' not in hidden_states[tk]:
            return False
        if 'after_steering' not in hidden_states[tk]:
            return False
        if not hidden_states[tk]['before_steering']:
            return False
        if not hidden_states[tk]['after_steering']:
            return False
    return True


# ============================================================
# Data Loading
# ============================================================

LLM_TO_TOKENIZER = {
    "Qwen3-4B": "Qwen/Qwen3-4B",
    "Qwen3-8B": "Qwen/Qwen3-8B",
    "gemma-2-2b-it": "google/gemma-2-2b-it",
    "Qwen3-1.7B": "Qwen/Qwen3-1.7B"
}


def compute_avg_response_tokens(result_dir, concept_ids, layer, llm):
    """Tokenize response_after from result files and return mean token count."""
    from transformers import AutoTokenizer
    tokenizer_name = LLM_TO_TOKENIZER.get(llm, llm)
    print(f"  Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    lengths = []
    for cid in concept_ids:
        cdir = os.path.join(result_dir, str(cid))
        if not os.path.isdir(cdir):
            continue
        for f in sorted(os.listdir(cdir)):
            if not f.endswith('.json'):
                continue
            parts = f.replace('.json', '').split('-')
            if len(parts) < 2:
                continue
            try:
                file_layer = int(parts[0])
                file_alpha = round(float(parts[1]), 2)
            except ValueError:
                continue
            if file_layer != layer:
                continue
            if file_alpha == 0.0:
                continue
            fpath = os.path.join(cdir, f)
            with open(fpath) as fp:
                result_data = json.load(fp)
            for item in result_data['results']:
                resp = item.get('response_after', '')
                if resp:
                    toks = tokenizer.encode(resp, add_special_tokens=False)
                    lengths.append(len(toks))

    if not lengths:
        print("  WARNING: no responses found, using fallback 128")
        return 128
    avg = np.mean(lengths)
    print(f"  Tokenized {len(lengths)} responses: "
          f"mean={avg:.1f}, median={np.median(lengths):.1f}, "
          f"std={np.std(lengths):.1f}")
    return avg


def load_splits_from_model(split_path, concept_ids):
    """
    Load per-concept prompt splits saved by the XGBoost training pipeline.
    Returns id_train, id_test dicts mapping concept_id -> list of prompt indices.
    """
    with open(split_path) as f:
        split_data = json.load(f)

    level = split_data.get('split_level', '')
    if level == 'prompt_per_concept':
        cs = split_data['concept_splits']
        id_train, id_test = {}, {}
        for cid in concept_ids:
            entry = cs.get(str(cid))
            if entry is None:
                continue
            id_train[cid] = entry['train'] + entry.get('val', [])
            id_test[cid] = entry['test']
        return id_train, id_test
    elif level == 'prompt':
        train = split_data['train_prompts']
        val = split_data.get('val_prompts', [])
        test = split_data['test_prompts']
        train_all = sorted(set(train) | set(val))
        id_train = {c: train_all for c in concept_ids}
        id_test = {c: test for c in concept_ids}
        return id_train, id_test
    else:
        raise ValueError(
            f"Splits file {split_path} has unsupported split_level='{level}'. "
            "Re-train with prompt-level splits first.")


def make_per_concept_splits(concept_ids, num_prompts, test_ratio, seed):
    """Fallback: generate independent per-concept prompt splits."""
    rng = np.random.RandomState(seed)
    n_test = max(1, int(num_prompts * test_ratio))
    id_train, id_test = {}, {}
    for cid in concept_ids:
        perm = np.arange(num_prompts)
        rng.shuffle(perm)
        id_test[cid] = sorted(perm[:n_test].tolist())
        id_train[cid] = sorted(perm[n_test:].tolist())
    return id_train, id_test


# ============================================================
# Upper Bound
# ============================================================

def compute_upper_bound(judgments, alphas, concept_ids, test_prompts):
    """Fraction of test prompts that have at least one alpha with judgment==1."""
    total = 0
    steerable = 0
    for cid in concept_ids:
        for pid in test_prompts.get(cid, []):
            total += 1
            for alpha in alphas:
                if judgments.get(cid, {}).get(alpha, {}).get(pid) == 1:
                    steerable += 1
                    break
    return steerable, total


# ============================================================
# Setting 1: Concept-Level GS on Training Set (single alpha)
# ============================================================

def setting1_concept_gs_train(judgments, alphas, concept_ids,
                              train_prompts, test_prompts):
    selected = {}
    concept_info = {}
    for cid in concept_ids:
        if cid not in judgments or not judgments[cid]:
            continue
        tps = train_prompts.get(cid, [])
        if not tps:
            continue
        best_alpha, best_sr = None, -1
        for alpha in alphas:
            if alpha not in judgments[cid]:
                continue
            sr = sum(1 for p in tps
                     if judgments[cid][alpha].get(p) == 1) / len(tps)
            if sr > best_sr:
                best_sr = sr
                best_alpha = alpha
        if best_alpha is None:
            continue
        concept_info[cid] = {'alpha': best_alpha, 'train_sr': best_sr}
        selected[cid] = {p: best_alpha for p in test_prompts.get(cid, [])}
    return selected, concept_info


# ============================================================
# Setting 2: Concept-Level GS on Test Set (oracle, single alpha)
# ============================================================

def setting2_concept_gs_test(judgments, alphas, concept_ids, test_prompts):
    selected = {}
    concept_info = {}
    for cid in concept_ids:
        if cid not in judgments or not judgments[cid]:
            continue
        tps = test_prompts.get(cid, [])
        if not tps:
            continue
        best_alpha, best_sr = None, -1
        for alpha in alphas:
            if alpha not in judgments[cid]:
                continue
            sr = sum(1 for p in tps
                     if judgments[cid][alpha].get(p) == 1) / len(tps)
            if sr > best_sr:
                best_sr = sr
                best_alpha = alpha
        if best_alpha is None:
            continue
        concept_info[cid] = {'alpha': best_alpha, 'test_sr': best_sr}
        selected[cid] = {p: best_alpha for p in tps}
    return selected, concept_info


# ============================================================
# Setting 4: Ours - ranked alpha list via XGBoost
# ============================================================

def setting4_ours_ranked(judgments, alphas, concept_ids, test_prompts,
                         hidden_states_dir, raw_states_dir,
                         model, curr_layer, k_list, n_list,
                         feature_groups=None,
                         feat_mean=None, feat_std=None):
    """
    Returns ranked[cid][pid] = [(alpha, p_success), ...] sorted desc by
    P(success).  The caller can then evaluate at any top-K cutoff.
    """
    import torch

    ranked = {}
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
            fname = f"{curr_layer}-{alpha}.pt"
            steered_path = os.path.join(hidden_states_dir, str(cid), fname)
            raw_path = os.path.join(raw_states_dir, str(cid), fname)
            if not os.path.exists(steered_path) or not os.path.exists(raw_path):
                continue
            try:
                steered_data = torch.load(steered_path, map_location='cpu',
                                          weights_only=False)
                raw_data = torch.load(raw_path, map_location='cpu',
                                      weights_only=False)
            except Exception:
                continue

            steered_samples = steered_data['hidden_states']
            raw_samples = raw_data['hidden_states']
            X_batch, batch_pids = [], []

            for pidx in tps:
                skey = f'sample_{pidx}'
                if skey not in steered_samples or skey not in raw_samples:
                    continue
                s_sample = steered_samples[skey]
                r_sample = raw_samples[skey]
                if not all(check_hidden_states_data(s_sample, n)
                           for n in n_list):
                    continue
                feat = get_handcrafted_features(
                    s_sample, r_sample,
                    curr_layer=curr_layer, k_list=k_list, n_list=n_list,
                    alpha=alpha, feature_groups=feature_groups)
                X_batch.append(feat)
                batch_pids.append(pidx)

            if not X_batch:
                continue
            X = np.array(X_batch)
            if feat_mean is not None and feat_std is not None:
                X = (X - feat_mean) / feat_std
            probs = model.predict_proba(X)
            for i, pidx in enumerate(batch_pids):
                prompt_alpha_probs[pidx][alpha] = probs[i, 1]

        for pidx in tps:
            total += 1
            if pidx not in prompt_alpha_probs or not prompt_alpha_probs[pidx]:
                skipped += 1
                continue
            ranked[cid][pidx] = sorted(
                prompt_alpha_probs[pidx].items(),
                key=lambda x: x[1], reverse=True)

        if (ci + 1) % 20 == 0 or (ci + 1) == n_concepts:
            print(f"  [Ours] Processed {ci + 1}/{n_concepts} concepts...")

    print(f"  [Ours] Total: {total}, Skipped: {skipped}")
    return ranked


# ============================================================
# Setting 5: Ascending-order ranked list
# ============================================================

def setting5_ascending_ranked(alphas, concept_ids, test_prompts):
    """Every prompt gets the same ordering: alphas sorted ascending."""
    ranked = {}
    asc = [(a, 0.0) for a in sorted(alphas)]
    for cid in concept_ids:
        ranked[cid] = {}
        for pid in test_prompts.get(cid, []):
            ranked[cid][pid] = asc
    return ranked


def setting5_descending_ranked(alphas, concept_ids, test_prompts):
    """Every prompt gets the same ordering: alphas sorted descending."""
    ranked = {}
    desc = [(a, 0.0) for a in sorted(alphas, reverse=True)]
    for cid in concept_ids:
        ranked[cid] = {}
        for pid in test_prompts.get(cid, []):
            ranked[cid][pid] = desc
    return ranked


# ============================================================
# Evaluation helpers
# ============================================================

def evaluate_single_alpha(selected, judgments, concept_ids, test_prompts):
    """
    Evaluate a method that picks exactly one alpha per prompt (Settings 1, 2).
    Returns (n_success, total, missing).
    """
    total = 0
    n_success = 0
    missing = 0
    for cid in concept_ids:
        for pid in test_prompts.get(cid, []):
            if cid not in selected or pid not in selected.get(cid, {}):
                missing += 1
                continue
            total += 1
            alpha = selected[cid][pid]
            if judgments.get(cid, {}).get(alpha, {}).get(pid) == 1:
                n_success += 1
    return n_success, total, missing


def evaluate_topk(ranked, judgments, concept_ids, test_prompts, k_values):
    """
    Evaluate a ranked alpha list at multiple budget levels.

    For budget K:
      - Try the first K alphas in the ranked list.
      - Success = any of them has judgment == 1.
      - Cost = position of first success (1-indexed), or K if none.

    Returns list of dicts, one per K value.
    """
    results = []
    for k in k_values:
        total = 0
        n_success = 0
        total_cost = 0
        missing = 0

        for cid in concept_ids:
            for pid in test_prompts.get(cid, []):
                if cid not in ranked or pid not in ranked.get(cid, {}):
                    missing += 1
                    continue
                total += 1
                alpha_list = ranked[cid][pid][:k]
                found = False
                for i, (alpha, _score) in enumerate(alpha_list):
                    if judgments.get(cid, {}).get(alpha, {}).get(pid) == 1:
                        n_success += 1
                        total_cost += (i + 1)
                        found = True
                        break
                if not found:
                    total_cost += len(alpha_list)

        results.append({
            'k': k,
            'n_success': n_success,
            'total': total,
            'missing': missing,
            'success_rate': n_success / total if total > 0 else 0,
            'avg_cost': total_cost / total if total > 0 else 0,
        })
    return results


# ============================================================
# Threshold sweep (confidence-based pruning for Setting 4)
# ============================================================

def evaluate_threshold_sweep(ranked, judgments, concept_ids, test_prompts,
                             k_values, thresholds):
    """
    For each (K, threshold): only try alphas whose P(success) >= threshold.
    If none pass the threshold, skip the prompt entirely (cost = 0 rollouts).
    """
    results = []
    for k in k_values:
        for thresh in thresholds:
            total = 0
            n_success = 0
            total_cost = 0
            n_skipped = 0
            missing = 0

            for cid in concept_ids:
                for pid in test_prompts.get(cid, []):
                    if cid not in ranked or pid not in ranked.get(cid, {}):
                        missing += 1
                        continue
                    total += 1
                    alpha_list = ranked[cid][pid][:k]
                    filtered = [(a, p) for a, p in alpha_list if p >= thresh]

                    if not filtered:
                        n_skipped += 1
                        continue

                    found = False
                    for i, (alpha, _score) in enumerate(filtered):
                        if judgments.get(cid, {}).get(alpha, {}).get(pid) == 1:
                            n_success += 1
                            total_cost += (i + 1)
                            found = True
                            break
                    if not found:
                        total_cost += len(filtered)

            results.append({
                'k': k,
                'threshold': round(thresh, 3),
                'total': total,
                'n_success': n_success,
                'n_skipped': n_skipped,
                'missing': missing,
                'success_rate': n_success / total if total > 0 else 0,
                'avg_cost': total_cost / total if total > 0 else 0,
            })
    return results


def print_threshold_table(sweep, full_len, few_tokens, K, title,
                          display_thresholds=None):
    print(f"\n{'=' * 88}")
    print(f"  {title}")
    print(f"{'=' * 88}")
    header = (f"  {'K':>3}  {'Thresh':>7}  {'Success%':>9}  {'Avg Cost':>9}  "
              f"{'Avg Tokens':>11}  {'Skipped%':>9}")
    print(header)
    print('  ' + '-' * 84)

    prev_k = None
    for r in sweep:
        if display_thresholds is not None:
            if r['threshold'] not in display_thresholds:
                continue
        if r['k'] != prev_k and prev_k is not None:
            print('  ' + '-' * 84)
        prev_k = r['k']

        sr = r['success_rate'] * 100
        tokens = K * few_tokens + r['avg_cost'] * full_len
        skip_pct = (r['n_skipped'] / r['total'] * 100
                    if r['total'] > 0 else 0)
        print(f"  {r['k']:>3}  {r['threshold']:>7.2f}  {sr:>8.1f}%  "
              f"{r['avg_cost']:>9.2f}  {tokens:>11.0f}  {skip_pct:>8.1f}%")


def _pareto_non_dominated(points):
    """
    points: list of (tokens, success_rate_pct, payload)
    Minimize tokens, maximize success_rate. Return list of non-dominated entries.
    """
    out = []
    for i, (ti, si, pi) in enumerate(points):
        dominated = False
        for j, (tj, sj, _) in enumerate(points):
            if i == j:
                continue
            if tj <= ti and sj >= si and (tj < ti or sj > si):
                dominated = True
                break
        if not dominated:
            out.append((ti, si, pi))
    out.sort(key=lambda x: x[0])
    return out


def build_baseline_markers(res1_id, res2_id, res2_ood, ub_id, ub_ood,
                         topk5_id, topk5_ood, topk5d_id, topk5d_ood, K, full_len):
    """Single-marker baselines for scatter plots (ID and OOD panels)."""
    sr1 = res1_id[0] / res1_id[1] * 100 if res1_id[1] > 0 else 0
    sr2_id_pct = (res2_id[0] / res2_id[1] * 100
                  if res2_id[1] > 0 else 0)
    sr2_ood_pct = (res2_ood[0] / res2_ood[1] * 100
                   if res2_ood[1] > 0 else 0)
    ub_id_pct = (ub_id[0] / ub_id[1] * 100 if ub_id[1] > 0 else 0)
    ub_ood_pct = (ub_ood[0] / ub_ood[1] * 100 if ub_ood[1] > 0 else 0)

    def _get_topk_at(topk_list, k_val):
        for r in topk_list:
            if r['k'] == k_val:
                return r
        return None

    asc_id = _get_topk_at(topk5_id, K)
    asc_ood = _get_topk_at(topk5_ood, K)
    desc_id = _get_topk_at(topk5d_id, K)
    desc_ood = _get_topk_at(topk5d_ood, K)

    baselines_id = [
        ("TCGS", sr1, 1.0 * full_len,
         'D', 'tab:orange'),
        ("CGS", sr2_id_pct,
         K * full_len, 's', 'tab:red'),
        ("IGS", ub_id_pct,
         K * full_len, '*', 'tab:green'),
    ]
    if asc_id:
        baselines_id.append((
            "IGS-A",
            asc_id['success_rate'] * 100,
            asc_id['avg_cost'] * full_len,
            '^', 'tab:blue'))
    if desc_id:
        baselines_id.append((
            "IGS-D",
            desc_id['success_rate'] * 100,
            desc_id['avg_cost'] * full_len,
            'v', 'tab:cyan'))

    baselines_ood = [
        ("CGS", sr2_ood_pct,
         K * full_len, 's', 'tab:red'),
        ("IGS", ub_ood_pct,
         K * full_len, '*', 'tab:green'),
    ]
    if asc_ood:
        baselines_ood.append((
            "IGS-A",
            asc_ood['success_rate'] * 100,
            asc_ood['avg_cost'] * full_len,
            '^', 'tab:blue'))
    if desc_ood:
        baselines_ood.append((
            "IGS-D",
            desc_ood['success_rate'] * 100,
            desc_ood['avg_cost'] * full_len,
            'v', 'tab:cyan'))

    return baselines_id, baselines_ood


def plot_no_threshold_vs_baselines(topk_id, topk_ood, full_len, few_tokens, K,
                                   baselines_id, baselines_ood, ub_id, ub_ood,
                                   save_path):
    """
    Two panels: Ours at each budget K as a single point (no τ sweep), plus
    baselines and upper-bound line.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    k_vals = sorted({r['k'] for r in topk_id} | {r['k'] for r in topk_ood})
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, max(len(k_vals), 1)))
    k_to_color = {k: c for k, c in zip(k_vals, cmap)}

    for ax, topk, baselines, ub, panel_title in [
        (axes[0], topk_id, baselines_id, ub_id, "In-Distribution"),
        (axes[1], topk_ood, baselines_ood, ub_ood, "OOD"),
    ]:
        by_k = {r['k']: r for r in topk}
        for k_val in k_vals:
            r = by_k.get(k_val)
            if r is None:
                continue
            early = K * few_tokens
            tokens = early + r['avg_cost'] * full_len
            sr = r['success_rate'] * 100
            ax.scatter([tokens], [sr], color=k_to_color[k_val], s=55,
                       edgecolors='white', linewidths=0.6, zorder=4,
                       label=f'Ours K={k_val}')

        ub_pct = ub[0] / ub[1] * 100 if ub[1] > 0 else 0
        ax.axhline(y=ub_pct, color='grey', linestyle=':', alpha=0.5)

        for name, sr_pct, token_cost, marker, mcolor in baselines:
            ax.plot(token_cost, sr_pct, marker, color=mcolor,
                    markersize=8, alpha=0.85, label=name, zorder=5)

        ax.set_xlabel('Avg Decoded Tokens per Prompt')
        ax.set_ylabel('Success Rate (%)')
        ax.set_title(panel_title, fontsize=13, fontweight='bold')
        ax.legend(
            fontsize=7, loc='lower center', bbox_to_anchor=(0.5, 0),
            ncol=4, columnspacing=0.8, handletextpad=0.4)
        ax.grid(True, alpha=0.25)
        ax.set_ylim(bottom=50, top=90)
        ax.set_xlim(left=0)

    fig.tight_layout(rect=[0, 0.14, 1, 1])
    out_dir = os.path.dirname(os.path.abspath(save_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  No-threshold vs baselines plot saved: {save_path}")


def plot_ours_pareto_front_only(sweep_id, sweep_ood, full_len, few_tokens, K,
                                save_path):
    """
    Two panels: non-dominated frontier over all (K, τ) sweep points for Ours
    only — no baseline markers.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    k_set = {r['k'] for r in sweep_id}
    if sweep_ood:
        k_set |= {r['k'] for r in sweep_ood}
    k_vals = sorted(k_set)
    n_k = max(len(k_vals), 1)
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, n_k))
    k_to_color = {k: c for k, c in zip(k_vals, cmap)}

    def _panel(ax, sweep, title):
        if not sweep:
            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.set_xlabel('Avg Decoded Tokens per Prompt')
            ax.set_ylabel('Success Rate (%)')
            ax.grid(True, alpha=0.25)
            ax.text(0.5, 0.5, '(no sweep)', transform=ax.transAxes,
                    ha='center', va='center', fontsize=10, color='0.45')
            ax.set_ylim(bottom=0)
            ax.set_xlim(left=0)
            return
        points = []
        for r in sweep:
            t = K * few_tokens + r['avg_cost'] * full_len
            s = r['success_rate'] * 100
            points.append((t, s, {'k': r['k'], 'thresh': r['threshold']}))
        front = _pareto_non_dominated(points)
        if not front:
            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.set_xlabel('Avg Decoded Tokens per Prompt')
            ax.set_ylabel('Success Rate (%)')
            ax.grid(True, alpha=0.25)
            ax.set_ylim(bottom=0)
            ax.set_xlim(left=0)
            return
        ftoks = [p[0] for p in front]
        fsr = [p[1] for p in front]
        fcols = [k_to_color[p[2]['k']] for p in front]
        ax.plot(ftoks, fsr, '-', color='0.35', linewidth=1.2, alpha=0.85,
                zorder=2)
        ax.scatter(ftoks, fsr, c=fcols, s=42, edgecolors='white',
                   linewidths=0.5, zorder=3)
        # Legend: one entry per K (color), not every frontier point
        for k_val in k_vals:
            ax.scatter([], [], color=k_to_color[k_val], s=42,
                       edgecolors='white', linewidths=0.5,
                       label=f'Ours K={k_val}')
        ax.set_xlabel('Avg Decoded Tokens per Prompt')
        ax.set_ylabel('Success Rate (%)')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(fontsize=7, loc='lower right')
        ax.grid(True, alpha=0.25)
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=0)

    _panel(axes[0], sweep_id, "In-Distribution")
    _panel(axes[1], sweep_ood, "OOD")

    plt.tight_layout()
    out_dir = os.path.dirname(os.path.abspath(save_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Ours Pareto-front plot saved: {save_path}")


# ============================================================
# Display
# ============================================================

def print_comparison_table(title, ub_steerable, ub_total, single_results,
                           topk_sections, full_len, few_tokens, K):
    """
    Args:
        single_results: [(name, n_succ, total, missing, cost_in_rollouts)]
        topk_sections:  [(method_name, topk_results, uses_early_pred)]
            uses_early_pred: if True, adds K * few_tokens to token cost
    """
    ub_pct = ub_steerable / ub_total * 100 if ub_total > 0 else 0
    ub_tokens = K * full_len
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")

    header = (f"  {'Method':<38} {'K':>3}  {'Success%':>9}  "
              f"{'Avg Cost':>9}  {'Avg Tokens':>11}")
    print(header)
    print('  ' + '-' * 76)

    print(f"  {'IGS (Item-level GS)':<38} {K:>3}  {ub_pct:>8.1f}%  "
          f"{float(K):>9.2f}  {ub_tokens:>11.0f}")

    for name, n_succ, total, _miss, cost in single_results:
        sr = n_succ / total * 100 if total > 0 else 0
        tokens = cost * full_len
        print(f"  {name:<38} {'1':>3}  {sr:>8.1f}%  {cost:>9.2f}  "
              f"{tokens:>11.0f}")

    for method_name, topk_results, uses_early in topk_sections:
        print('  ' + '-' * 76)
        for r in topk_results:
            sr = r['success_rate'] * 100
            early_cost = K * few_tokens if uses_early else 0
            tokens = early_cost + r['avg_cost'] * full_len
            print(f"  {method_name:<38} {r['k']:>3}  {sr:>8.1f}%  "
                  f"{r['avg_cost']:>9.2f}  {tokens:>11.0f}")


# ============================================================
# Plot from saved results (no recomputation)
# ============================================================

def _setting_to_tuple3(d):
    """Reconstruct evaluate_single_alpha-style (n_success, total, missing)."""
    return (d["n_success"], d["total"], d["missing"])


def _filter_rows_by_k(rows, k_allowed):
    """
    rows: list of dicts with a 'k' key (topk or threshold_sweep).
    k_allowed: set or sequence of int — only these budget K values are kept.
    """
    if rows is None:
        return None
    allowed = set(k_allowed)
    return [r for r in rows if r.get("k") in allowed]


def run_figures_from_saved_json(results_path, pareto_plot, k_filter=None):
    """
    Load a results.json written by this script and redraw pareto figures.

    Expects the same structure as the `out` dict in main(): config with
    full_rollout_tokens and few_tokens, alphas, id/ood blocks with settings
    and optional setting4_topk / threshold_sweep.

    If k_filter is a non-empty list of int, only rows whose 'k' is in that
    list (and <= len(alphas)) are used for Ours / threshold plots.
    k_filter None keeps all K present in the file.
    """
    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)

    alphas = data.get("alphas") or []
    K = len(alphas)
    if K == 0:
        print(f"Error: no 'alphas' in {results_path}; cannot plot.")
        sys.exit(1)

    cfg = data.get("config") or {}
    if cfg.get("full_rollout_tokens") is None:
        print("Error: results.json must include config.full_rollout_tokens.")
        sys.exit(1)
    full_len = float(cfg["full_rollout_tokens"])
    few_tok = int(cfg.get("few_tokens", 5))

    id_b = data.get("id")
    ood_b = data.get("ood")
    if not id_b or not ood_b:
        print("Error: results.json must include 'id' and 'ood' sections.")
        sys.exit(1)

    ub_id = (id_b["upper_bound"]["steerable"], id_b["upper_bound"]["total"])
    ub_ood = (ood_b["upper_bound"]["steerable"], ood_b["upper_bound"]["total"])
    res1_id = _setting_to_tuple3(id_b["setting1"])
    res2_id = _setting_to_tuple3(id_b["setting2"])
    res2_ood = _setting_to_tuple3(ood_b["setting2"])

    topk5_id = id_b["setting5_asc_topk"]
    topk5_ood = ood_b["setting5_asc_topk"]
    topk5d_id = id_b["setting5_desc_topk"]
    topk5d_ood = ood_b["setting5_desc_topk"]

    topk4_id = id_b.get("setting4_topk")
    topk4_ood = ood_b.get("setting4_topk")

    sweep_id = id_b.get("threshold_sweep")
    sweep_ood = ood_b.get("threshold_sweep")

    if k_filter is not None and k_filter:
        k_allowed = sorted({k for k in k_filter if k <= K})
        if not k_allowed:
            print("Error: no value in --top_k_values is <= len(alphas).")
            sys.exit(1)
        print(f"  Using only K in {k_allowed} (from --top_k_values).")
        if topk4_id is not None:
            topk4_id = _filter_rows_by_k(topk4_id, k_allowed)
        if topk4_ood is not None:
            topk4_ood = _filter_rows_by_k(topk4_ood, k_allowed)
        if sweep_id is not None:
            sweep_id = _filter_rows_by_k(sweep_id, k_allowed)
        if sweep_ood is not None:
            sweep_ood = _filter_rows_by_k(sweep_ood, k_allowed)

    # Threshold-sweep Pareto (Ours only) — mirror main() branching
    if sweep_id is not None or sweep_ood is not None:
        p = Path(pareto_plot)
        front_path = str(p.with_name(p.stem + "_front" + p.suffix))
        if sweep_id is not None and sweep_ood is not None:
            plot_ours_pareto_front_only(
                sweep_id, sweep_ood, full_len, few_tok, K, front_path)
        elif sweep_id is not None:
            plot_ours_pareto_front_only(
                sweep_id, [], full_len, few_tok, K, front_path)
    else:
        print("  No threshold_sweep in results; skipping Ours-only Pareto front plot.")

    if topk4_id is not None and topk4_ood is not None:
        baselines_id, baselines_ood = build_baseline_markers(
            res1_id, res2_id, res2_ood, ub_id, ub_ood,
            topk5_id, topk5_ood, topk5d_id, topk5d_ood, K, full_len)
        plot_no_threshold_vs_baselines(
            topk4_id, topk4_ood, full_len, few_tok, K,
            baselines_id, baselines_ood, ub_id, ub_ood,
            pareto_plot)
    else:
        print("  No setting4_topk (Ours) in results; skipping pareto vs baselines plot.")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='AlphaSearch: Early Prediction vs Grid Search')
    parser.add_argument('--llm', type=str, default='Qwen3-4B')
    parser.add_argument('--steering_method', type=str, default='diffmean')
    parser.add_argument('--layer', type=int, default=10)
    parser.add_argument('--k_list', type=str, default='1,2,3,5,15')
    parser.add_argument('--target_token_n_list', type=str, default='1,2,3')
    parser.add_argument('--id_concept_max', type=int, default=119)
    parser.add_argument('--ood_concept_min', type=int, default=120)
    parser.add_argument('--ood_concept_max', type=int, default=149)
    parser.add_argument('--num_prompts', type=int, default=50)
    parser.add_argument('--test_ratio', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to saved XGBoost model (.json).')
    parser.add_argument('--skip_ours', action='store_true',
                        help='Skip Setting 4 (Ours).')
    parser.add_argument('--model_tag', type=str, default='weighted_search',
                        help='Model tag used in saved model filename.')
    parser.add_argument('--top_k_values', type=str, default='5,10,15,20',
                        help='Comma-separated K budgets for ranked search.')
    parser.add_argument('--full_rollout_tokens', type=int, default=None,
                        help='Avg tokens per full rollout. '
                             'If not set, computed from data via tokenizer.')
    parser.add_argument('--few_tokens', type=int, default=6,
                        help='Tokens decoded per early-exit prediction.')
    parser.add_argument('--threshold_sweep', action='store_true',
                        help='Run confidence-threshold sweep for Setting 4.')
    parser.add_argument('--display_thresholds', type=str,
                        default='0.1,0.3,0.5,0.7,0.9',
                        help='Thresholds to display in the sweep table.')
    parser.add_argument('--pareto_plot', type=str, default=None,
                        help=('Path to save no-threshold vs baselines plot '
                              '(default: alphasearch_out/<llm>/<steering_method>/pareto.pdf). '
                              'With --threshold_sweep, also writes '
                              '<stem>_front<suffix> for the Ours-only '
                              'Pareto frontier.'))
    parser.add_argument('--output', type=str, default=None,
                        help=('Path to save results JSON (default: '
                              'alphasearch_out/<llm>/<steering_method>/results.json).'))
    parser.add_argument(
        '--from-results', type=str, default=None, metavar='JSON',
        help=('Load a previously saved results.json and only regenerate '
              'figures (no recomputation). Default --pareto_plot: '
              '<json_dir>/pareto.png. With --from-results, --top_k_values '
              'selects which budget K to plot for Ours and threshold_sweep.'))
    parser.add_argument(
        '--judgments_cache', type=str, default=None,
        help='JSON from script/cache_judgments.py; skips scanning result JSONs.')
    parser.add_argument(
        '--feature_cache', type=str, default=None,
        help='Feature .npz from script/build_feature_cache.py; skips torch.load in Setting 4.')
    args = parser.parse_args()

    if args.from_results:
        if not os.path.isfile(args.from_results):
            print(f"Error: --from-results file not found: {args.from_results}")
            sys.exit(1)
        if args.pareto_plot is None:
            args.pareto_plot = str(
                Path(args.from_results).resolve().parent / "pareto.pdf")
        print(f"Plot-only mode: loading {args.from_results}")
        print(f"  Figures -> pareto: {args.pareto_plot}")
        top_k_for_plot = [
            int(x.strip()) for x in args.top_k_values.split(',') if x.strip()]
        if not top_k_for_plot:
            print("Error: --from-results needs at least one K in --top_k_values.")
            sys.exit(1)
        run_figures_from_saved_json(
            args.from_results, args.pareto_plot, k_filter=top_k_for_plot)
        return

    if args.output is None:
        args.output = os.path.join(
            root_dir, 'alphasearch_out', args.llm, args.steering_method,
            'results.json')
    if args.pareto_plot is None:
        args.pareto_plot = os.path.join(
            root_dir, 'alphasearch_out', args.llm, args.steering_method,
            'pareto.pdf')

    k_list = [int(x.strip()) for x in args.k_list.split(',')]
    n_list = [int(x.strip()) for x in args.target_token_n_list.split(',')]
    top_k_values = [int(x.strip()) for x in args.top_k_values.split(',')]

    data_dir = os.path.join(root_dir, 'data')
    result_dir = os.path.join(data_dir, 'result', args.steering_method,
                              args.llm)
    hidden_states_dir = os.path.join(data_dir, 'hidden_states',
                                     args.steering_method, args.llm)
    raw_states_dir = os.path.join(data_dir, 'hidden_states_raw',
                                  args.steering_method, args.llm)

    if args.model_path is None:
        _model_dir_for_splits = os.path.join(
            root_dir, 'saved_predictors_xgb',
            args.llm, args.steering_method)
        split_path = os.path.join(_model_dir_for_splits, 'splits.json')
    else:
        split_path = os.path.join(
            os.path.dirname(args.model_path), 'splits.json')

    cp_meta = None
    loaded_part = load_concept_partition_from_splits_json(split_path)
    if loaded_part is not None:
        id_cids, ood_cids, cp_meta = loaded_part
        print(f"\nUsing ID/OOD concept lists from {split_path} (concept_partition).")
    else:
        id_cids = list(range(0, args.id_concept_max + 1))
        ood_cids = list(range(args.ood_concept_min, args.ood_concept_max + 1))

    all_cids = id_cids + ood_cids
    id_caption, ood_caption = id_ood_display_captions(cp_meta, args)

    # ========== Load Judgments ==========
    print("Loading judgments...")
    jextras = {}
    if args.judgments_cache and os.path.isfile(args.judgments_cache):
        print(f"  Using judgment cache: {args.judgments_cache}")
        judgments, alphas, jextras = load_judgments_from_cache_file(
            args.judgments_cache, all_cids, args.layer,
            steering_method=args.steering_method, llm=args.llm)
    else:
        if args.judgments_cache:
            print(f"  WARNING: judgments_cache not found ({args.judgments_cache}), "
                  f"scanning result JSONs.")
        judgments, alphas = load_all_judgments(result_dir, all_cids, args.layer)
    n_loaded = sum(1 for c in judgments if judgments[c])
    print(f"Loaded {n_loaded} concepts, {len(alphas)} alphas: {alphas}")

    top_k_values = [k for k in top_k_values if k <= len(alphas)]

    # ========== Compute Avg Response Length ==========
    if args.full_rollout_tokens is None:
        tok = jextras.get('avg_response_tokens') if jextras else None
        if tok is not None:
            print(f"\nUsing avg response length from judgment cache: {float(tok):.1f} tokens")
            args.full_rollout_tokens = float(tok)
        else:
            print("\nComputing avg response length from data...")
            args.full_rollout_tokens = compute_avg_response_tokens(
                result_dir, all_cids, args.layer, args.llm)
    else:
        print(f"\nUsing provided full_rollout_tokens={args.full_rollout_tokens}")

    # ========== Prompt Split ==========
    # Prefer loading per-concept splits from XGBoost model for consistency
    if not args.skip_ours and os.path.exists(split_path):
        try:
            id_train, id_test = load_splits_from_model(split_path, id_cids)
            n_test_per = [len(id_test[c]) for c in id_cids if c in id_test]
            print(f"\nLoaded per-concept splits from {split_path}")
            print(f"  {len(id_test)} concepts, "
                  f"{n_test_per[0]} test prompts each")
        except ValueError as e:
            print(f"\n  WARNING: {e}")
            print("  Falling back to generated per-concept splits.")
            id_train, id_test = make_per_concept_splits(
                id_cids, args.num_prompts, args.test_ratio, args.seed)
    else:
        id_train, id_test = make_per_concept_splits(
            id_cids, args.num_prompts, args.test_ratio, args.seed)
        print(f"\nGenerated per-concept splits (seed={args.seed})")

    ood_test = {c: list(range(args.num_prompts)) for c in ood_cids}

    # ========== Upper Bounds ==========
    ub_id = compute_upper_bound(judgments, alphas, id_cids, id_test)
    ub_ood = compute_upper_bound(judgments, alphas, ood_cids, ood_test)

    # ========== Setting 1 (ID only) ==========
    print("\nRunning TCGS (Training-set Concept-level GS)...")
    sel1, info1 = setting1_concept_gs_train(
        judgments, alphas, id_cids, id_train, id_test)
    res1_id = evaluate_single_alpha(sel1, judgments, id_cids, id_test)

    # ========== Setting 2 ==========
    print("Running CGS (Concept-level GS Oracle)...")
    sel2_id, info2_id = setting2_concept_gs_test(
        judgments, alphas, id_cids, id_test)
    res2_id = evaluate_single_alpha(sel2_id, judgments, id_cids, id_test)

    sel2_ood, info2_ood = setting2_concept_gs_test(
        judgments, alphas, ood_cids, ood_test)
    res2_ood = evaluate_single_alpha(sel2_ood, judgments, ood_cids, ood_test)

    # ========== Setting 5: Ascending Order ==========
    print("Running IGS-A (Item-level GS, Ascending)...")
    ranked5_id = setting5_ascending_ranked(alphas, id_cids, id_test)
    topk5_id = evaluate_topk(ranked5_id, judgments, id_cids, id_test,
                             [len(alphas)])

    ranked5_ood = setting5_ascending_ranked(alphas, ood_cids, ood_test)
    topk5_ood = evaluate_topk(ranked5_ood, judgments, ood_cids, ood_test,
                              [len(alphas)])

    # ========== Setting 5b: Descending Order ==========
    print("Running IGS-D (Item-level GS, Descending)...")
    ranked5d_id = setting5_descending_ranked(alphas, id_cids, id_test)
    topk5d_id = evaluate_topk(ranked5d_id, judgments, id_cids, id_test,
                              [len(alphas)])

    ranked5d_ood = setting5_descending_ranked(alphas, ood_cids, ood_test)
    topk5d_ood = evaluate_topk(ranked5d_ood, judgments, ood_cids, ood_test,
                               [len(alphas)])

    # ========== Setting 4: Ours ==========
    topk4_id = None
    topk4_ood = None
    ranked4_id = None
    ranked4_ood = None

    if not args.skip_ours:
        print("\nRunning Ours (Early Prediction)...")
        if args.model_path is None:
            model_dir = os.path.join(
                root_dir, 'saved_predictors_xgb',
                args.llm, args.steering_method)
            model_path = os.path.join(model_dir, 'model.json')
        else:
            model_path = args.model_path
            model_dir = os.path.dirname(model_path)

        if not os.path.exists(model_path):
            print(f"  WARNING: Model not found: {model_path}")
        else:
            import xgboost as xgb
            xgb_model = xgb.XGBClassifier()
            xgb_model.load_model(model_path)
            print(f"  Model loaded: {model_path}")

            result_meta_path = os.path.join(model_dir, 'result.json')
            model_k_list = k_list
            model_n_list = n_list
            model_feature_groups = None
            model_feat_mean = None
            model_feat_std = None
            if os.path.exists(result_meta_path):
                with open(result_meta_path) as _mf:
                    model_meta = json.load(_mf)
                train_args = model_meta.get('args', {})
                if 'k_list' in train_args:
                    model_k_list = [int(x) for x in
                                    train_args['k_list'].split(',')]
                if 'target_token_n_list' in train_args:
                    model_n_list = [int(x) for x in
                                    train_args['target_token_n_list'].split(',')]
                if model_k_list != k_list or model_n_list != n_list:
                    print(f"  Using model's training config: "
                          f"k_list={model_k_list}, n_list={model_n_list}")
                fg_str = model_meta.get('feature_groups')
                if fg_str:
                    model_feature_groups = [g.strip() for g in fg_str.split(',')]
                    print(f"  Feature groups from model: {model_feature_groups}")
                if model_meta.get('normalize_features', False):
                    norm_path = os.path.join(model_dir, 'norm.npz')
                    if os.path.exists(norm_path):
                        norm_data = np.load(norm_path)
                        model_feat_mean = norm_data['mean']
                        model_feat_std = norm_data['std']
                        print(f"  Feature normalization loaded: {norm_path}")

            use_fc = args.feature_cache and os.path.isfile(args.feature_cache)
            if use_fc:
                print(f"  Using feature cache: {args.feature_cache}")
                bundle, _fc_meta = load_feature_cache(args.feature_cache)
                cache_fg = list(_fc_meta.get("feature_groups") or [])
                desired_fg = list(model_feature_groups) if model_feature_groups else cache_fg
                fc_col_idx = resolve_feature_columns(
                    cache_feature_groups=cache_fg,
                    desired_feature_groups=desired_fg,
                    curr_layer=args.layer,
                    k_list=model_k_list,
                    n_list=model_n_list,
                    cache_feature_dim=int(_fc_meta.get("feature_dim", bundle["X"].shape[1])),
                    include_alpha=True,
                )
                if fc_col_idx is not None:
                    bundle["X"] = bundle["X"][:, fc_col_idx]
                    print(f"  Subset features: {len(cache_fg)} groups -> "
                          f"{len(desired_fg)} groups ({bundle['X'].shape[1]} cols)")
            else:
                if args.feature_cache:
                    print(f"  WARNING: feature_cache not found ({args.feature_cache}), "
                          f"loading .pt files.")

            print("  ID test set...")
            if use_fc:
                ranked4_id = setting4_ours_ranked_from_cache(
                    judgments, alphas, id_cids, id_test,
                    bundle, xgb_model,
                    feat_mean=model_feat_mean, feat_std=model_feat_std)
            else:
                ranked4_id = setting4_ours_ranked(
                    judgments, alphas, id_cids, id_test,
                    hidden_states_dir, raw_states_dir,
                    xgb_model, args.layer, model_k_list, model_n_list,
                    feature_groups=model_feature_groups,
                    feat_mean=model_feat_mean, feat_std=model_feat_std)
            topk4_id = evaluate_topk(ranked4_id, judgments, id_cids, id_test,
                                     top_k_values)

            print("  OOD test set...")
            if use_fc:
                ranked4_ood = setting4_ours_ranked_from_cache(
                    judgments, alphas, ood_cids, ood_test,
                    bundle, xgb_model,
                    feat_mean=model_feat_mean, feat_std=model_feat_std)
            else:
                ranked4_ood = setting4_ours_ranked(
                    judgments, alphas, ood_cids, ood_test,
                    hidden_states_dir, raw_states_dir,
                    xgb_model, args.layer, model_k_list, model_n_list,
                    feature_groups=model_feature_groups,
                    feat_mean=model_feat_mean, feat_std=model_feat_std)
            topk4_ood = evaluate_topk(ranked4_ood, judgments, ood_cids,
                                      ood_test, top_k_values)

    # ========== Display: ID ==========
    K = len(alphas)
    full_len = args.full_rollout_tokens
    few_tok = args.few_tokens

    id_single = [
        ("TCGS", *res1_id, 1.0),
        ("CGS", *res2_id, float(K)),
    ]
    id_topk = []
    if topk4_id is not None:
        id_topk.append(("Ours", topk4_id, True))
    id_topk.append(("IGS-A", topk5_id, False))
    id_topk.append(("IGS-D", topk5d_id, False))

    print_comparison_table(
        id_caption,
        *ub_id, id_single, id_topk, full_len, few_tok, K)

    # ========== Display: OOD ==========
    ood_single = [
        ("CGS", *res2_ood, float(K)),
    ]
    ood_topk = []
    if topk4_ood is not None:
        ood_topk.append(("Ours", topk4_ood, True))
    ood_topk.append(("IGS-A", topk5_ood, False))
    ood_topk.append(("IGS-D", topk5d_ood, False))

    print_comparison_table(
        ood_caption,
        *ub_ood, ood_single, ood_topk, full_len, few_tok, K)

    # ========== Threshold Sweep ==========
    sweep_id = None
    sweep_ood = None

    if args.threshold_sweep and not args.skip_ours:
        if ranked4_id is None and ranked4_ood is None:
            print("\n  WARNING: No ranked data available; skipping threshold sweep.")
        else:
            fine_thresholds = list(np.arange(0.0, 0.92, 0.02))
            disp_thresholds = set(
                round(float(x), 3)
                for x in args.display_thresholds.split(','))

            if ranked4_id is not None:
                print("\nRunning threshold sweep (ID)...")
                sweep_id = evaluate_threshold_sweep(
                    ranked4_id, judgments, id_cids, id_test,
                    top_k_values, fine_thresholds)
                print_threshold_table(
                    sweep_id, full_len, few_tok, K,
                    f"Threshold Sweep — Ours (ID, {id_caption})",
                    display_thresholds=disp_thresholds)

            if ranked4_ood is not None:
                print("\nRunning threshold sweep (OOD)...")
                sweep_ood = evaluate_threshold_sweep(
                    ranked4_ood, judgments, ood_cids, ood_test,
                    top_k_values, fine_thresholds)
                print_threshold_table(
                    sweep_ood, full_len, few_tok, K,
                    f"Threshold Sweep — Ours (OOD, {ood_caption})",
                    display_thresholds=disp_thresholds)

            p = Path(args.pareto_plot)
            front_path = str(p.with_name(p.stem + '_front' + p.suffix))
            if sweep_id is not None and sweep_ood is not None:
                plot_ours_pareto_front_only(
                    sweep_id, sweep_ood, full_len, few_tok, K, front_path)
            elif sweep_id is not None:
                plot_ours_pareto_front_only(
                    sweep_id, [], full_len, few_tok, K, front_path)

    if (args.pareto_plot and not args.skip_ours
            and topk4_id is not None and topk4_ood is not None):
        baselines_id, baselines_ood = build_baseline_markers(
            res1_id, res2_id, res2_ood, ub_id, ub_ood,
            topk5_id, topk5_ood, topk5d_id, topk5d_ood, K, full_len)
        plot_no_threshold_vs_baselines(
            topk4_id, topk4_ood, full_len, few_tok, K,
            baselines_id, baselines_ood, ub_id, ub_ood,
            args.pareto_plot)

    # ========== Cost Summary ==========
    print(f"\n{'=' * 80}")
    print("  Cost Notes")
    print(f"{'=' * 80}")
    print(f"  K = {K} alpha candidates, "
          f"full rollout = {full_len:.0f} tokens, "
          f"early exit = {few_tok} tokens")
    print(f"  Avg Tokens = (early-exit cost) + (Avg Cost) x {full_len:.0f}")
    n_train_avg = int(np.mean([len(id_train[c]) for c in id_cids
                               if c in id_train]))
    print(f"  Setting 1: ~{K}x{n_train_avg} = "
          f"~{K * n_train_avg} training rollouts per concept (amortized)")
    print(f"  Setting 4: early-exit cost = "
          f"{K} x {few_tok} = {K * few_tok} tokens (included in Avg Tokens)")

    # ========== Save ==========
    if args.output:
        out = {
            'config': vars(args),
            'alphas': alphas,
            'split_source': split_path if os.path.exists(split_path) else 'generated',
            'concept_partition': cp_meta,
            'id': {
                'upper_bound': {'steerable': ub_id[0], 'total': ub_id[1]},
                'setting1': {'n_success': res1_id[0], 'total': res1_id[1],
                             'missing': res1_id[2]},
                'setting2': {'n_success': res2_id[0], 'total': res2_id[1],
                             'missing': res2_id[2]},
                'setting5_asc_topk': topk5_id,
                'setting5_desc_topk': topk5d_id,
            },
            'ood': {
                'upper_bound': {'steerable': ub_ood[0], 'total': ub_ood[1]},
                'setting2': {'n_success': res2_ood[0], 'total': res2_ood[1],
                             'missing': res2_ood[2]},
                'setting5_asc_topk': topk5_ood,
                'setting5_desc_topk': topk5d_ood,
            },
            'concept_best_alphas': {
                'setting1_id': {str(k): v for k, v in info1.items()},
                'setting2_id': {str(k): v for k, v in info2_id.items()},
                'setting2_ood': {str(k): v for k, v in info2_ood.items()},
            },
        }
        if topk4_id is not None:
            out['id']['setting4_topk'] = topk4_id
        if topk4_ood is not None:
            out['ood']['setting4_topk'] = topk4_ood
        if sweep_id is not None:
            out['id']['threshold_sweep'] = sweep_id
        if sweep_ood is not None:
            out['ood']['threshold_sweep'] = sweep_ood

        os.makedirs(os.path.dirname(os.path.abspath(args.output)),
                    exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
