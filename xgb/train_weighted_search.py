import argparse
import importlib.util
import json
import os
import pickle
import sys
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def _load_module(path: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


root_utils = _load_module(os.path.join(ROOT_DIR, "utils.py"), "root_utils")
train_utils = _load_module(os.path.join(os.path.dirname(__file__), "train.py"), "xgb_train")

load_paired_data = root_utils.load_paired_data
build_label_map = train_utils.build_label_map
extract_features = train_utils.extract_features
print_feature_importances = train_utils.print_feature_importances

xgb_utils = _load_module(os.path.join(os.path.dirname(__file__), "utils.py"), "xgb_utils")
ALL_FEATURE_GROUPS = xgb_utils.ALL_FEATURE_GROUPS

feature_cache_mod = _load_module(
    os.path.join(os.path.dirname(__file__), "feature_cache.py"), "xfc")
load_feature_cache = feature_cache_mod.load_feature_cache
resolve_feature_columns = feature_cache_mod.resolve_feature_columns

_cp = _load_module(os.path.join(os.path.dirname(__file__), "concept_partition.py"), "concept_partition")
build_static_partition = _cp.build_static_partition
build_stratified_shuffle_partition = _cp.build_stratified_shuffle_partition
concept_to_level_map = _cp.concept_to_level_map


def print_metrics(y_true: np.ndarray, y_pred: np.ndarray, title: str) -> Dict:
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    print(f"Samples: {len(y_true)}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))
    print("Confusion Matrix:")
    print(cm)

    labels = sorted(set(y_true) | set(y_pred))
    cm_labeled = {
        "labels": [CLASS_NAMES.get(int(l), str(l)) for l in labels],
        "matrix": cm.tolist(),
    }
    per_class = {}
    for k, v in report.items():
        if isinstance(v, dict):
            label_name = CLASS_NAMES.get(int(k), str(k)) if k.lstrip("-").isdigit() else k
            per_class[label_name] = v

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "confusion_matrix": cm_labeled,
        "per_class": per_class,
    }


def make_class_weights(y: np.ndarray) -> Dict[int, float]:
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    n_classes = len(classes)
    weights = {int(c): total / (n_classes * cnt) for c, cnt in zip(classes, counts)}
    return weights


def make_sample_weights(y: np.ndarray, class_weights: Dict[int, float]) -> np.ndarray:
    return np.array([class_weights[int(v)] for v in y], dtype=np.float32)


def sample_param_sets(rng: np.random.Generator, n_trials: int, num_class: int, seed: int) -> List[Dict]:
    param_sets = []
    for _ in range(n_trials):
        p = {
            "objective": "multi:softprob" if num_class > 2 else "binary:logistic",
            "num_class": num_class if num_class > 2 else None,
            "n_estimators": int(rng.integers(500, 750)),
            "learning_rate": float(rng.choice([0.02, 0.03, 0.05, 0.08])),
            "max_depth": int(rng.integers(6, 10)),
            "min_child_weight": float(rng.choice([2, 4, 6, 8, 10, 12])),
            "subsample": float(rng.choice([0.65, 0.75, 0.85, 0.95, 1.0])),
            "colsample_bytree": float(rng.choice([0.65, 0.75, 0.85, 0.95, 1.0])),
            "reg_alpha": float(rng.choice([0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0])),
            "reg_lambda": float(rng.choice([1.0, 2.0, 5.0, 10.0, 20.0])),
            "gamma": float(rng.choice([0.0, 0.05, 0.1, 0.2, 0.4, 0.8])),
            "max_delta_step": int(rng.choice([0, 1, 2])),
            "eval_metric": "mlogloss" if num_class > 2 else "logloss",
            "random_state": seed,
            "use_label_encoder": False,
            "n_jobs": -1,
        }
        if p["num_class"] is None:
            p.pop("num_class")
        param_sets.append(p)
    return param_sets


def load_baseline_success_set(result_dir, concept_ids, layer):
    """Return set of (concept_id, sample_id) where baseline (alpha=0) judgment == 1."""
    exclude = set()
    for cid in concept_ids:
        path = os.path.join(result_dir, str(cid), f"{layer}-0.0.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        for item in data.get("results", []):
            if item.get("judgment") == 1:
                exclude.add((cid, item["sample_id"]))
    return exclude


def _filter_baseline_success(pairs, labels, meta, baseline_set):
    """Remove samples whose (concept_id, sample_id) is in baseline_set."""
    keep = [i for i, (cid, _alpha, sid) in enumerate(meta)
            if (cid, sid) not in baseline_set]
    return (
        [pairs[i] for i in keep],
        [labels[i] for i in keep],
        [meta[i] for i in keep],
    )


def _label_map_from_feature_meta(fmeta: Dict) -> Dict:
    out = {}
    for k, v in fmeta["label_map"].items():
        try:
            ik = int(k)
        except ValueError:
            ik = k
        out[ik] = int(v)
    return out


def _validate_feature_cache_meta(fmeta, args, k_list, n_list, feature_groups):
    """
    Validate that a feature cache is compatible with the current training run.

    Returns:
        col_idx  (np.ndarray | None) -- column indices to select from cache X
                 when feature_groups is a *strict subset* of the cache's groups.
                 None when the groups match exactly (no subsetting needed).
    """
    if fmeta.get("steering_method") != args.steering_method:
        raise ValueError(
            f"features_cache steering_method {fmeta.get('steering_method')} != "
            f"{args.steering_method}"
        )
    if fmeta.get("llm") != args.llm:
        raise ValueError(
            f"features_cache llm {fmeta.get('llm')} != {args.llm}"
        )
    if int(fmeta.get("curr_layer")) != int(args.curr_layer):
        raise ValueError(
            f"features_cache curr_layer {fmeta.get('curr_layer')} != {args.curr_layer}"
        )
    if list(fmeta.get("k_list") or []) != k_list:
        raise ValueError("features_cache k_list mismatch with training args")
    if list(fmeta.get("target_token_n_list") or []) != n_list:
        raise ValueError("features_cache target_token_n_list mismatch")
    if fmeta.get("alpha_min") != args.alpha_min or fmeta.get("alpha_max") != args.alpha_max:
        raise ValueError(
            f"features_cache alpha bounds ({fmeta.get('alpha_min')}, {fmeta.get('alpha_max')}) "
            f"!= ({args.alpha_min}, {args.alpha_max})"
        )
    if fmeta.get("id_ood_partition") != args.id_ood_partition:
        raise ValueError("features_cache id_ood_partition mismatch")

    desired_fg = list(feature_groups) if feature_groups else list(ALL_FEATURE_GROUPS)
    cache_fg = list(fmeta.get("feature_groups") or [])

    include_alpha = True
    col_idx = resolve_feature_columns(
        cache_feature_groups=cache_fg,
        desired_feature_groups=desired_fg,
        curr_layer=args.curr_layer,
        k_list=k_list,
        n_list=n_list,
        cache_feature_dim=int(fmeta.get("feature_dim", -1)),
        include_alpha=include_alpha,
    )
    if col_idx is not None:
        print(f"  Feature subset: cache has {len(cache_fg)} groups, "
              f"selecting {len(desired_fg)} -> {len(col_idx)} / "
              f"{fmeta.get('feature_dim', '?')} columns")
    return col_idx


CLASS_NAMES = {0: "Understeer", 1: "Successful Steer", 2: "Oversteer"}
EPS = 1e-8


def plot_top_features_heatmap(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    output_path: str,
    top_n: int = 15,
    title: str = "",
):
    """Plot a heatmap of the top-N important features averaged per class.

    Rows = classes (understeer / successful / oversteer).
    Columns = top features ranked by XGBoost importance.
    Values are z-score-normalized per feature (column) across the 3 classes.
    """
    importances = model.feature_importances_
    n_avail = min(int(importances.shape[0]), int(X.shape[1]))
    if n_avail < 1:
        print("Skipping top-feature heatmap: no features in model.")
        return
    n_cols = min(top_n, n_avail)
    top_indices = np.argsort(importances)[::-1][:n_cols]

    classes = sorted(CLASS_NAMES.keys())
    avg_matrix = np.zeros((len(classes), n_cols))
    for ri, cls in enumerate(classes):
        mask = y == cls
        if mask.sum() > 0:
            avg_matrix[ri] = X[mask][:, top_indices].mean(axis=0)

    col_mean = avg_matrix.mean(axis=0)
    col_std = avg_matrix.std(axis=0)
    col_std[col_std < EPS] = 1.0
    norm_matrix = (avg_matrix - col_mean) / col_std

    top_names = [feature_names[i] for i in top_indices]
    row_labels = [CLASS_NAMES[c] for c in classes]

    fig, ax = plt.subplots(figsize=(max(10, 0.8 * n_cols), 3.5))
    abs_max = max(np.abs(norm_matrix).max(), EPS)
    im = ax.imshow(
        norm_matrix, aspect="auto", cmap="RdBu_r",
        vmin=-abs_max, vmax=abs_max,
    )

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(top_names, rotation=55, ha="right", fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=10)

    for ri in range(norm_matrix.shape[0]):
        for ci in range(norm_matrix.shape[1]):
            ax.text(ci, ri, f"{norm_matrix[ri, ci]:.2f}",
                    ha="center", va="center", fontsize=7,
                    color="white" if abs(norm_matrix[ri, ci]) > abs_max * 0.6 else "black")

    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04, label="z-score")
    ax.set_title(title or "Top Features by Class (z-normed per feature)", fontsize=12, pad=10)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved top-feature heatmap → {output_path}")


def prepare_training_data(args, k_list, n_list, feature_groups):
    """Shared data-prep pipeline used by both hyperparameter search and
    fixed-param reruns.

    Loads (or generates) the feature matrix, builds ID/OOD partitions,
    per-concept prompt splits, ID sample train/val/test assignment,
    optional normalization, and class/sample weights.

    Returns a dict with every array / metadata item downstream steps need.
    """
    if args.id_ood_partition == "stratified_shuffle":
        id_concepts, ood_concepts, concept_partition_meta = build_stratified_shuffle_partition(
            args.seed)
    else:
        id_concepts, ood_concepts, concept_partition_meta = build_static_partition(
            args.id_concept_max, args.ood_concept_min, args.ood_concept_max)
    concept_partition_meta = dict(concept_partition_meta)
    concept_partition_meta["concept_id_to_level"] = {
        str(k): v for k, v in concept_to_level_map().items()
    }

    n_test = max(1, int(args.num_prompts * args.test_size))
    n_val = max(1, int((args.num_prompts - n_test) * args.val_size))
    rng_split = np.random.RandomState(args.seed)

    concept_splits = {}
    for cid in id_concepts:
        perm = np.arange(args.num_prompts)
        rng_split.shuffle(perm)
        c_test = sorted(perm[:n_test].tolist())
        c_val = sorted(perm[n_test:n_test + n_val].tolist())
        c_train = sorted(perm[n_test + n_val:].tolist())
        concept_splits[cid] = {
            'train': c_train, 'val': c_val, 'test': c_test}

    print(f"\nPer-concept prompt split (seed={args.seed}): "
          f"{n_test} test, {n_val} val, {args.num_prompts - n_test - n_val} train per concept")
    _ex_cid = id_concepts[0]
    print(f"  Example concept {_ex_cid}: {concept_splits[_ex_cid]}")

    id_set = set(id_concepts)
    ood_set = set(ood_concepts)
    all_concept_list = list(id_concepts) + list(ood_concepts)

    use_cache = bool(args.features_cache and os.path.isfile(args.features_cache))

    X_all = y_all = c_all = a_all = s_all = None
    pairs = labels = meta = None

    if use_cache:
        print(f"\nLoading features from cache: {args.features_cache}")
        bundle, fmeta = load_feature_cache(args.features_cache)
        col_idx = _validate_feature_cache_meta(fmeta, args, k_list, n_list, feature_groups)
        X_all = bundle["X"]
        if col_idx is not None:
            X_all = X_all[:, col_idx]
        y_all = bundle["y"].astype(np.int64)
        c_all = bundle["concept_id"]
        a_all = bundle["alpha"]
        s_all = bundle["sample_id"]
        label_map = _label_map_from_feature_meta(fmeta)

        if args.exclude_baseline_success:
            result_dir = os.path.join(ROOT_DIR, "data", "result",
                                      args.steering_method, args.llm)
            baseline_set = load_baseline_success_set(
                result_dir, all_concept_list, args.curr_layer)
            keep = np.array([
                (int(c_all[i]), int(s_all[i])) not in baseline_set
                for i in range(len(c_all))
            ], dtype=bool)
            n_before = len(c_all)
            X_all = X_all[keep]
            y_all = y_all[keep]
            c_all = c_all[keep]
            a_all = a_all[keep]
            s_all = s_all[keep]
            print(f"Excluded {n_before - len(X_all)} baseline-success samples "
                  f"(kept {len(X_all)}/{n_before})")
            if len(X_all) == 0:
                raise RuntimeError("No data left after baseline exclusion.")

        # Drop the trailing alpha column so non-Condition ablations can
        # exclude SteeringStrength as a feature.
        if args.no_alpha_feature:
            if X_all.shape[1] < 1:
                raise RuntimeError("Cannot drop alpha: cache X has no columns.")
            X_all = X_all[:, :-1]
            print(f"Dropped trailing alpha column (no_alpha_feature). "
                  f"X_all shape -> {X_all.shape}")

        id_mask = np.isin(c_all, list(id_set))
        if not np.any(id_mask):
            raise RuntimeError("No ID data found in cache.")
        X_id = X_all[id_mask]
        y_id = y_all[id_mask]
        id_meta = [
            (int(c_all[i]), float(a_all[i]), int(s_all[i]))
            for i in np.where(id_mask)[0]
        ]
        print(f"ID matrix (cache): X={X_id.shape}, y={y_id.shape}")
    else:
        if args.features_cache:
            print(f"  WARNING: features_cache not found ({args.features_cache}), "
                  f"loading from disk.")

        print(f"\nLoading data: {len(id_concepts)} ID + {len(ood_concepts)} OOD concepts "
              f"(single pass, id_ood_partition={args.id_ood_partition})")
        print(f"  ID concept ids (first 12): {id_concepts[:12]}")
        pairs, labels, meta = load_paired_data(
            method=args.steering_method,
            model=args.llm,
            layer=args.curr_layer,
            n_list=n_list,
            k_list=k_list,
            alpha_min=args.alpha_min,
            alpha_max=args.alpha_max,
            concept_ids=all_concept_list,
            force_regenerate_raw=args.force_regenerate_raw,
            return_metadata=True,
            batch_size_raw=args.batch_size_raw,
        )
        if not pairs:
            raise RuntimeError("No data found.")

        if args.exclude_baseline_success:
            result_dir = os.path.join(ROOT_DIR, "data", "result",
                                      args.steering_method, args.llm)
            baseline_set = load_baseline_success_set(
                result_dir, all_concept_list, args.curr_layer)
            n_before = len(pairs)
            pairs, labels, meta = _filter_baseline_success(
                pairs, labels, meta, baseline_set)
            print(f"Excluded {n_before - len(pairs)} samples with successful "
                  f"baseline (kept {len(pairs)}/{n_before})")
            if not pairs:
                raise RuntimeError("No data left after baseline exclusion.")

        id_ix = [i for i in range(len(meta)) if meta[i][0] in id_set]
        if not id_ix:
            raise RuntimeError("No ID data found.")
        id_pairs = [pairs[i] for i in id_ix]
        id_labels = [labels[i] for i in id_ix]
        id_meta = [meta[i] for i in id_ix]

        label_map = build_label_map(id_labels)
        X_id, y_id = extract_features(
            id_pairs, id_labels, label_map, args.curr_layer, k_list, n_list,
            metadata=id_meta, feature_groups=feature_groups)
        if args.no_alpha_feature:
            if X_id.shape[1] < 1:
                raise RuntimeError("Cannot drop alpha: extracted X has no columns.")
            X_id = X_id[:, :-1]
            print(f"Dropped trailing alpha column (no_alpha_feature). "
                  f"X_id shape -> {X_id.shape}")
        print(f"ID matrix: X={X_id.shape}, y={y_id.shape}")

    idx_train, idx_val, idx_test_id = [], [], []
    for i, (cid, _alpha, sample_id) in enumerate(id_meta):
        cs = concept_splits.get(cid)
        if cs is None:
            idx_train.append(i)
            continue
        if sample_id in cs['test']:
            idx_test_id.append(i)
        elif sample_id in cs['val']:
            idx_val.append(i)
        else:
            idx_train.append(i)
    idx_train = np.array(idx_train)
    idx_val = np.array(idx_val)
    idx_test_id = np.array(idx_test_id)
    print(f"Sample split: train={len(idx_train)}, val={len(idx_val)}, test={len(idx_test_id)}")

    X_train, y_train = X_id[idx_train], y_id[idx_train]
    X_val, y_val = X_id[idx_val], y_id[idx_val]
    X_test_id, y_test_id = X_id[idx_test_id], y_id[idx_test_id]

    feat_mean, feat_std = None, None
    if args.normalize_features:
        feat_mean = X_train.mean(axis=0)
        feat_std = X_train.std(axis=0)
        feat_std[feat_std < EPS] = 1.0
        X_train = (X_train - feat_mean) / feat_std
        X_val = (X_val - feat_mean) / feat_std
        X_test_id = (X_test_id - feat_mean) / feat_std
        X_id = (X_id - feat_mean) / feat_std
        print(f"Feature normalization: applied (train mean/std, {len(feat_mean)} features)")

    class_weights = make_class_weights(y_train)
    sample_weights = make_sample_weights(y_train, class_weights)
    print(f"Train label distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"Class weights: {class_weights}")

    num_class = len(np.unique(y_id))

    return {
        "id_concepts": id_concepts,
        "ood_concepts": ood_concepts,
        "id_set": id_set,
        "ood_set": ood_set,
        "concept_partition_meta": concept_partition_meta,
        "concept_splits": concept_splits,
        "use_cache": use_cache,
        "X_all": X_all, "y_all": y_all, "c_all": c_all, "a_all": a_all, "s_all": s_all,
        "pairs": pairs, "labels": labels, "meta": meta,
        "X_id": X_id, "y_id": y_id, "id_meta": id_meta, "label_map": label_map,
        "idx_train": idx_train, "idx_val": idx_val, "idx_test_id": idx_test_id,
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test_id": X_test_id, "y_test_id": y_test_id,
        "feat_mean": feat_mean, "feat_std": feat_std,
        "class_weights": class_weights, "sample_weights": sample_weights,
        "num_class": num_class,
    }


def evaluate_ood(data, best_model, args, k_list, n_list, feature_groups):
    """Run OOD evaluation using the trained model and shared data bundle."""
    ood_concepts = data["ood_concepts"]
    ood_set = data["ood_set"]
    feat_mean = data["feat_mean"]
    feat_std = data["feat_std"]
    label_map = data["label_map"]

    if data["use_cache"]:
        print(f"\nOOD data from cache: {len(ood_concepts)} concepts")
        c_all = data["c_all"]
        ood_mask = np.isin(c_all, list(ood_set))
        if not np.any(ood_mask):
            print("No OOD rows in cache.")
            return None
        X_ood = data["X_all"][ood_mask]
        y_ood = data["y_all"][ood_mask]
        if feat_mean is not None:
            X_ood = (X_ood - feat_mean) / feat_std
        y_ood_pred = best_model.predict(X_ood)
        return print_metrics(y_ood, y_ood_pred, "OOD Test Set")

    print(f"\nLoading OOD data: {len(ood_concepts)} concepts")
    print(f"  OOD concept ids (first 12): {ood_concepts[:12]}")
    meta = data["meta"]
    pairs = data["pairs"]
    labels = data["labels"]
    ood_ix = [i for i in range(len(meta)) if meta[i][0] in ood_set]
    ood_pairs = [pairs[i] for i in ood_ix]
    ood_labels = [labels[i] for i in ood_ix]
    ood_meta = [meta[i] for i in ood_ix]

    if not ood_pairs:
        print("No OOD data found.")
        return None

    if label_map:
        for lbl in sorted(set(ood_labels) - set(label_map.keys())):
            label_map[lbl] = max(label_map.values()) + 1
    X_ood, y_ood = extract_features(
        ood_pairs, ood_labels, label_map, args.curr_layer, k_list, n_list,
        metadata=ood_meta, feature_groups=feature_groups)
    if args.no_alpha_feature:
        if X_ood.shape[1] < 1:
            raise RuntimeError("Cannot drop alpha: extracted OOD X has no columns.")
        X_ood = X_ood[:, :-1]
    if feat_mean is not None:
        X_ood = (X_ood - feat_mean) / feat_std
    y_ood_pred = best_model.predict(X_ood)
    return print_metrics(y_ood, y_ood_pred, "OOD Test Set")


def save_run_outputs(
    save_dir: str,
    suffix: str,
    best_model,
    best_params: Dict,
    best_score: float,
    data: Dict,
    id_metrics: Dict,
    ood_metrics,
    feature_groups,
    args,
    k_list: List[int],
    n_list: List[int],
    save_plots: bool = True,
):
    """Write model / result / labelmap / splits / norm / feature_importance
    (and optionally top_features heatmap) to `save_dir`, with filenames like
    `model{suffix}.json`. Pass suffix="" for the canonical (search) outputs.
    """
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, f"model{suffix}.json")
    result_path = os.path.join(save_dir, f"result{suffix}.json")
    label_map_path = os.path.join(save_dir, f"labelmap{suffix}.pkl")
    split_path = os.path.join(save_dir, f"splits{suffix}.json")
    norm_path = os.path.join(save_dir, f"norm{suffix}.npz")
    heatmap_path = os.path.join(save_dir, f"top_features{suffix}.png")
    importance_path = os.path.join(save_dir, f"feature_importance{suffix}.json")

    best_model.save_model(model_path)
    with open(label_map_path, "wb") as f:
        pickle.dump(data["label_map"], f)

    feat_mean = data["feat_mean"]
    feat_std = data["feat_std"]
    if feat_mean is not None:
        np.savez(norm_path, mean=feat_mean, std=feat_std)
        print(f"Saved normalization stats: {norm_path}")

    split_payload = {
        "split_level": "prompt_per_concept",
        "num_prompts": args.num_prompts,
        "num_samples": int(len(data["y_id"])),
        "concept_splits": {str(k): v for k, v in data["concept_splits"].items()},
        "concept_partition": data["concept_partition_meta"],
        "idx_train": data["idx_train"].tolist(),
        "idx_val": data["idx_val"].tolist(),
        "idx_test_id": data["idx_test_id"].tolist(),
        "args": {
            "test_size": args.test_size,
            "val_size": args.val_size,
            "seed": args.seed,
            "llm": args.llm,
            "steering_method": args.steering_method,
            "curr_layer": args.curr_layer,
            "k_list": args.k_list,
            "target_token_n_list": args.target_token_n_list,
            "alpha_min": args.alpha_min,
            "alpha_max": args.alpha_max,
            "id_ood_partition": args.id_ood_partition,
            "id_concept_max": args.id_concept_max,
            "ood_concept_min": args.ood_concept_min,
            "ood_concept_max": args.ood_concept_max,
        },
    }
    with open(split_path, "w") as f:
        json.dump(split_payload, f, indent=2)

    result_payload = {
        "best_validation_macro_f1": best_score,
        "best_params": best_params,
        "class_weights": {str(k): v for k, v in data["class_weights"].items()},
        "id_metrics": id_metrics,
        "ood_metrics": ood_metrics,
        "feature_groups": ",".join(feature_groups) if feature_groups else ",".join(ALL_FEATURE_GROUPS),
        "no_alpha_feature": bool(getattr(args, "no_alpha_feature", False)),
        "normalize_features": args.normalize_features,
        "concept_partition": data["concept_partition_meta"],
        "args": vars(args),
    }
    with open(result_path, "w") as f:
        json.dump(result_payload, f, indent=2)

    print(f"\nSaved to {save_dir}/")
    print(f"  model{suffix}.json              — XGBoost model")
    print(f"  result{suffix}.json             — metrics, confusion matrices & config")
    print(f"  labelmap{suffix}.pkl            — label mapping")
    print(f"  splits{suffix}.json             — train/val/test splits")
    if feat_mean is not None:
        print(f"  norm{suffix}.npz                — feature normalization stats")

    include_alpha_names = (data["id_meta"] is not None) and not getattr(args, "no_alpha_feature", False)
    print_feature_importances(
        best_model, args.curr_layer, k_list, n_list,
        include_alpha=include_alpha_names,
        feature_groups=feature_groups,
    )

    feature_names = train_utils.build_feature_names(
        args.curr_layer, k_list, n_list,
        include_alpha=include_alpha_names,
        feature_groups=feature_groups,
    )
    if save_plots:
        plot_top_features_heatmap(
            best_model, data["X_id"], data["y_id"], feature_names,
            output_path=heatmap_path,
            top_n=15,
            title=f"Top 15 Features — {args.llm} / {args.steering_method} / Layer {args.curr_layer}",
        )

    importances = best_model.feature_importances_
    ranked_idx = np.argsort(importances)[::-1]
    importance_payload = {
        "ranked_features": [
            {"rank": r + 1, "name": feature_names[i], "importance": float(importances[i])}
            for r, i in enumerate(ranked_idx)
        ],
    }
    group_imp: Dict[str, List[float]] = {}
    for i, name in enumerate(feature_names):
        prefix = name.split("_")[0]
        group_imp.setdefault(prefix, []).append(float(importances[i]))
    importance_payload["group_summary"] = {
        g: {"count": len(v), "sum": sum(v), "mean": sum(v) / len(v), "max": max(v)}
        for g, v in sorted(group_imp.items(), key=lambda kv: -sum(kv[1]))
    }
    with open(importance_path, "w") as f:
        json.dump(importance_payload, f, indent=2)
    print(f"  feature_importance{suffix}.json — ranked features & group summary")


def main():
    parser = argparse.ArgumentParser(description="Weighted XGBoost hyperparameter search with ID/OOD evaluation.")
    parser.add_argument("--llm", type=str, default="Qwen3-4B")
    parser.add_argument("--steering_method", type=str, default="diffmean")
    parser.add_argument("--curr_layer", type=int, default=10)
    parser.add_argument("--k_list", type=str, default="1,2,3,5,10,15")
    parser.add_argument("--target_token_n_list", type=str, default="1,3,5")
    parser.add_argument("--alpha_min", type=float, default=None)
    parser.add_argument("--alpha_max", type=float, default=None)
    parser.add_argument("--id_concept_max", type=int, default=119)
    parser.add_argument("--ood_concept_min", type=int, default=120)
    parser.add_argument("--ood_concept_max", type=int, default=149)
    parser.add_argument(
        "--id_ood_partition",
        type=str,
        choices=("stratified_shuffle", "static_ranges"),
        default="stratified_shuffle",
        help="ID/OOD concept assignment. stratified_shuffle: per level (low/mid/high), "
             "40 train + 10 test concepts each, shuffled (seed --seed). "
             "static_ranges: legacy contiguous ID 0..id_concept_max and OOD ood_min..ood_max.",
    )
    parser.add_argument("--test_size", type=float, default=0.3,
                        help="Fraction of prompts used for test.")
    parser.add_argument("--val_size", type=float, default=0.2,
                        help="Fraction of remaining prompts used for validation.")
    parser.add_argument("--num_prompts", type=int, default=50,
                        help="Total number of prompts per concept.")
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="saved_predictors_xgb")
    parser.add_argument("--model_tag", type=str, default="weighted_search")
    parser.add_argument(
        "--force_regenerate_raw",
        action="store_true",
        help="Regenerate raw hidden states even if cached files already exist.",
    )
    parser.add_argument(
        "--batch_size_raw", type=int, default=8,
        help="Batch size for raw hidden-state extraction (used with --force_regenerate_raw).",
    )
    parser.add_argument(
        "--feature_groups", type=str, default=None,
        help="Comma-separated feature groups. Default: all ("
             + ",".join(ALL_FEATURE_GROUPS) + ").",
    )
    parser.add_argument(
        "--normalize_features",
        action="store_true",
        help="Z-normalize each feature (per-column) using training-set stats before training. "
             "Saves mean/std to <stem>_norm.npz for inference.",
    )
    parser.add_argument(
        "--no_alpha_feature",
        action="store_true",
        help="Drop the trailing alpha (steering strength) column from the feature "
             "vector. Useful for clean ablations of paper feature groups, since alpha "
             "belongs to the Steering Condition group but is otherwise appended "
             "unconditionally inside get_handcrafted_features.",
    )
    parser.add_argument(
        "--exclude_baseline_success",
        action="store_true",
        help="Exclude samples where the unsteered (alpha=0) response already "
             "received a successful-steering judgment. Requires baseline eval "
             "(before_steering_eval.py) to have been processed.",
    )
    parser.add_argument(
        "--features_cache",
        type=str,
        default=None,
        help="Prebuilt .npz (+ _meta.json) from script/build_feature_cache.py; "
             "skips load_paired_data and extract_features.",
    )
    args = parser.parse_args()

    k_list = [int(x.strip()) for x in args.k_list.split(",")]
    n_list = [int(x.strip()) for x in args.target_token_n_list.split(",")]
    feature_groups = (
        [g.strip() for g in args.feature_groups.split(",")]
        if args.feature_groups else None
    )

    data = prepare_training_data(args, k_list, n_list, feature_groups)

    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test_id, y_test_id = data["X_test_id"], data["y_test_id"]
    sample_weights = data["sample_weights"]
    num_class = data["num_class"]

    rng = np.random.default_rng(args.seed)
    trial_params = sample_param_sets(rng, args.n_trials, num_class=num_class, seed=args.seed)

    best_score = -1.0
    best_params = None
    best_model = None

    print(f"\nRunning {len(trial_params)} trials...")
    for i, params in enumerate(trial_params, start=1):
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, sample_weight=sample_weights)
        y_val_pred = model.predict(X_val)
        val_macro_f1 = f1_score(y_val, y_val_pred, average="macro", zero_division=0)
        val_acc = accuracy_score(y_val, y_val_pred)
        print(f"Trial {i:02d}/{len(trial_params)}: macro_f1={val_macro_f1:.4f}, acc={val_acc:.4f}")
        if val_macro_f1 > best_score:
            best_score = val_macro_f1
            best_params = params
            best_model = model

    print(f"\nBest validation macro-F1: {best_score:.4f}")
    print(f"Best params: {best_params}")

    y_test_id_pred = best_model.predict(X_test_id)
    id_metrics = print_metrics(y_test_id, y_test_id_pred, "ID Test Set")

    ood_metrics = evaluate_ood(data, best_model, args, k_list, n_list, feature_groups)

    save_dir = os.path.join(args.save_dir, args.llm, args.steering_method)
    save_run_outputs(
        save_dir=save_dir,
        suffix="",
        best_model=best_model,
        best_params=best_params,
        best_score=best_score,
        data=data,
        id_metrics=id_metrics,
        ood_metrics=ood_metrics,
        feature_groups=feature_groups,
        args=args,
        k_list=k_list,
        n_list=n_list,
        save_plots=True,
    )


if __name__ == "__main__":
    main()
