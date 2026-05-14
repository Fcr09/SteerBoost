import argparse
import numpy as np
import sys
import os
import importlib.util
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import resample
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

root_utils_path = os.path.join(root_dir, "utils.py")
spec_root = importlib.util.spec_from_file_location("root_utils", root_utils_path)
root_utils = importlib.util.module_from_spec(spec_root)
spec_root.loader.exec_module(root_utils)
load_paired_data = root_utils.load_paired_data

local_dir = os.path.dirname(os.path.abspath(__file__))
local_utils_path = os.path.join(local_dir, "utils.py")
spec_local = importlib.util.spec_from_file_location("local_utils", local_utils_path)
local_utils = importlib.util.module_from_spec(spec_local)
spec_local.loader.exec_module(local_utils)
get_handcrafted_features = local_utils.get_handcrafted_features
ALL_FEATURE_GROUPS = local_utils.ALL_FEATURE_GROUPS
DEFAULT_FEATURE_GROUPS = local_utils.DEFAULT_FEATURE_GROUPS


def build_label_map(labels):
    unique_labels = set(labels)
    label_map = {}
    if any(isinstance(l, str) for l in unique_labels):
        mapping_logic = {
            'understeer': 0,
            'success': 1,
            'over steer': 2,
            'oversteer': 2
        }
        sorted_labels = sorted(list(unique_labels))
        next_id = 3
        for lbl in sorted_labels:
            if lbl in mapping_logic:
                label_map[lbl] = mapping_logic[lbl]
            else:
                label_map[lbl] = next_id
                next_id += 1
    return label_map


def extract_features(pairs, labels, label_map, curr_layer, k_list, n_list,
                     metadata=None, feature_groups=None,
                     return_valid_indices=False):
    X, y = [], []
    for idx, ((s_sample, r_sample), label) in enumerate(zip(pairs, labels)):
        alpha = None
        if metadata is not None:
            _, alpha, _ = metadata[idx]
        feat = get_handcrafted_features(
            s_sample, r_sample,
            curr_layer=curr_layer,
            k_list=k_list,
            n_list=n_list,
            alpha=alpha,
            feature_groups=feature_groups,
        )
        X.append(feat)
        if label_map:
            y.append(label_map.get(label, -1))
        else:
            y.append(int(label))
    X = np.array(X)
    y = np.array(y)
    valid_mask = y != -1
    X_out = X[valid_mask]
    y_out = y[valid_mask]
    if return_valid_indices:
        idx_kept = np.flatnonzero(valid_mask)
        return X_out, y_out, idx_kept
    return X_out, y_out


def oversample(X_train, y_train, seed):
    X_resampled, y_resampled = [], []
    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    max_count = np.max(class_counts)
    for cls in unique_classes:
        cls_idx = np.where(y_train == cls)[0]
        X_cls, y_cls = X_train[cls_idx], y_train[cls_idx]
        if len(X_cls) < max_count * 0.8:
            X_cls, y_cls = resample(
                X_cls, y_cls,
                replace=True,
                n_samples=int(max_count * 0.8),
                random_state=seed
            )
        X_resampled.append(X_cls)
        y_resampled.append(y_cls)
    return np.vstack(X_resampled), np.hstack(y_resampled)


def evaluate_and_print(model, X, y, dataset_name):
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f"\n{'='*60}")
    print(f"  {dataset_name}")
    print(f"{'='*60}")
    print(f"Samples: {len(y)}")
    print(f"Accuracy: {acc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y, y_pred, zero_division=0))
    print(f"Confusion Matrix:")
    print(confusion_matrix(y, y_pred))
    return acc


def build_feature_names(curr_layer, k_list, n_list, include_alpha=False,
                        feature_groups=None):
    """Build human-readable names for every feature in the feature vector.

    Covers: before-steering matrices, cross-token propagation,
    after-steering intervention magnitude, SteerProj, and optional alpha.
    """
    if feature_groups is None:
        feature_groups = DEFAULT_FEATURE_GROUPS
    active = set(feature_groups)

    stat_suffixes = ["Mean", "Std", "Max", "Min"]

    sorted_k_list = sorted(list(set(k_list)))
    sorted_n_list = sorted(list(set(n_list)))

    token_indices = sorted(list(set([0] + sorted_n_list)))
    token_names = [f'T{n}' for n in token_indices]
    non_zero_token_names = [f'T{n}' for n in token_indices if n != 0]

    layer_indices = sorted(list(set([curr_layer] + [curr_layer + k for k in sorted_k_list])))
    layer_names = [f'L{l}' for l in layer_indices]

    l0_idx = layer_indices.index(curr_layer)
    layer_names_no_l0 = [n for i, n in enumerate(layer_names) if i != l0_idx]

    feature_names = []

    def _matrix_names(metric, t_names, l_names):
        names = []
        for t_name in t_names:
            for l_name in l_names:
                names.append(f"{metric}_{t_name}_{l_name}")
        for stat in stat_suffixes:
            names.append(f"{metric}_{stat}_Global")
        for t_name in t_names:
            names.append(f"{metric}_Mean_{t_name}_AllLayers")
            names.append(f"{metric}_Std_{t_name}_AllLayers")
            names.append(f"{metric}_Max_{t_name}_AllLayers")
        for l_name in l_names:
            names.append(f"{metric}_Mean_{l_name}_AllTokens")
            names.append(f"{metric}_Std_{l_name}_AllTokens")
            names.append(f"{metric}_Max_{l_name}_AllTokens")
        return names

    # ---- Base matrix features (canonical order) ----
    if "MagDiff" in active:
        feature_names.extend(_matrix_names("MagDiff", token_names, layer_names))
    if "CosSim" in active:
        feature_names.extend(_matrix_names("CosSim", token_names, layer_names))
    if "SteerAlign" in active:
        feature_names.extend(_matrix_names("SteerAlign", token_names, layer_names_no_l0))
    if "SteerProj" in active:
        feature_names.extend(_matrix_names("SteerProj", token_names, layer_names))

    # ---- Cross-token propagation features ----
    if len(non_zero_token_names) > 0:
        def _prop_names(metric, l_names):
            names = []
            for t_name in non_zero_token_names:
                for l_name in l_names:
                    names.append(f"{metric}_{t_name}_{l_name}")
            for stat in stat_suffixes:
                names.append(f"{metric}_{stat}_Global")
            for t_name in non_zero_token_names:
                names.append(f"{metric}_Mean_{t_name}_AllLayers")
                names.append(f"{metric}_Std_{t_name}_AllLayers")
                names.append(f"{metric}_Max_{t_name}_AllLayers")
            for l_name in l_names:
                names.append(f"{metric}_Mean_{l_name}_AllNonZeroTokens")
                names.append(f"{metric}_Std_{l_name}_AllNonZeroTokens")
                names.append(f"{metric}_Max_{l_name}_AllNonZeroTokens")
            return names

        if "MagDiffRatio" in active:
            feature_names.extend(_prop_names("MagDiffRatio", layer_names))
        if "CosSimDelta" in active:
            feature_names.extend(_prop_names("CosSimDelta", layer_names))
        if "SteerAlignDelta" in active:
            feature_names.extend(_prop_names("SteerAlignDelta", layer_names_no_l0))

    # ---- After-steering features ----
    if "AfterSteering" in active:
        feature_names.append("InterventionMag")

    # ---- Alpha ----
    if include_alpha:
        feature_names.append("Alpha")

    return feature_names


def print_feature_importances(model, curr_layer, k_list, n_list,
                              include_alpha=False, top_n=30, bottom_n=30,
                              feature_groups=None):
    """Print ranked feature importances plus group-level summaries.

    Shows the top-N and bottom-N individual features, then aggregates
    importance by feature group so you can decide which groups to keep or drop.
    """
    feature_names = build_feature_names(curr_layer, k_list, n_list,
                                        include_alpha=include_alpha,
                                        feature_groups=feature_groups)

    if not hasattr(model, 'feature_importances_'):
        return

    importances = model.feature_importances_
    if len(importances) != len(feature_names):
        print(f"Feature importance count mismatch. "
              f"Expected {len(feature_names)}, got {len(importances)}")
        return

    indices = np.argsort(importances)[::-1]
    n_feats = len(feature_names)
    top_n = min(top_n, n_feats)
    bottom_n = min(bottom_n, n_feats)

    print(f"\n{'=' * 60}")
    print(f"  Top {top_n} Features (most important)")
    print(f"{'=' * 60}")
    for rank in range(top_n):
        idx = indices[rank]
        print(f"  {rank+1:3d}. {feature_names[idx]:50s}  {importances[idx]:.6f}")

    print(f"\n{'=' * 60}")
    print(f"  Bottom {bottom_n} Features (least important)")
    print(f"{'=' * 60}")
    for rank in range(bottom_n):
        idx = indices[n_feats - bottom_n + rank]
        print(f"  {rank+1:3d}. {feature_names[idx]:50s}  {importances[idx]:.6f}")

    # ---- Group-level summary ----
    group_importance = {}  # group_name -> list of importances
    for idx, name in enumerate(feature_names):
        prefix = name.split("_")[0]
        if prefix == "InterventionMag":
            parts = name.split("_")
            prefix = parts[0]
        group_importance.setdefault(prefix, []).append(importances[idx])

    print(f"\n{'=' * 60}")
    print(f"  Feature Group Summary (total {n_feats} features)")
    print(f"{'=' * 60}")
    print(f"  {'Group':<25s} {'Count':>5s}  {'Sum':>8s}  {'Mean':>8s}  {'Max':>8s}")
    print(f"  {'-'*25} {'-'*5}  {'-'*8}  {'-'*8}  {'-'*8}")
    group_rows = []
    for group, imps in group_importance.items():
        arr = np.array(imps)
        group_rows.append((group, len(arr), np.sum(arr), np.mean(arr), np.max(arr)))
    group_rows.sort(key=lambda r: r[2], reverse=True)
    for group, cnt, s, m, mx in group_rows:
        print(f"  {group:<25s} {cnt:5d}  {s:8.4f}  {m:8.6f}  {mx:8.6f}")

    # ---- Per-layer breakdown ----
    import re
    layer_indices = sorted(list(set(
        [curr_layer] + [curr_layer + k for k in k_list]
    )))
    layer_tags = [f'L{l}' for l in layer_indices]

    layer_importance = {tag: [] for tag in layer_tags}
    layer_importance['Global/Other'] = []

    for idx, name in enumerate(feature_names):
        matched = False
        for tag in layer_tags:
            if f'_{tag}_' in name or name.endswith(f'_{tag}'):
                layer_importance[tag].append(importances[idx])
                matched = True
                break
        if not matched:
            if re.search(r'_L\d+', name):
                layer_importance['Global/Other'].append(importances[idx])
            elif 'AllLayers' in name or 'Global' in name:
                layer_importance['Global/Other'].append(importances[idx])

    print(f"\n{'=' * 60}")
    print(f"  Importance by Layer")
    print(f"{'=' * 60}")
    print(f"  {'Layer':<15s} {'Count':>5s}  {'Sum':>8s}  {'Mean':>8s}  {'Max':>8s}")
    print(f"  {'-'*15} {'-'*5}  {'-'*8}  {'-'*8}  {'-'*8}")
    layer_rows = []
    for tag in layer_tags:
        imps = layer_importance[tag]
        if imps:
            arr = np.array(imps)
            layer_rows.append((tag, len(arr), np.sum(arr), np.mean(arr), np.max(arr)))
    layer_rows.sort(key=lambda r: r[2], reverse=True)
    for tag, cnt, s, m, mx in layer_rows:
        print(f"  {tag:<15s} {cnt:5d}  {s:8.4f}  {m:8.6f}  {mx:8.6f}")
    if layer_importance['Global/Other']:
        arr = np.array(layer_importance['Global/Other'])
        print(f"  {'Global/Other':<15s} {len(arr):5d}  {np.sum(arr):8.4f}  "
              f"{np.mean(arr):8.6f}  {np.max(arr):8.6f}")

    # ---- Per-token breakdown ----
    token_indices = sorted(list(set([0] + list(n_list))))
    token_tags = [f'T{n}' for n in token_indices]

    token_importance = {tag: [] for tag in token_tags}
    token_importance['AllTokens/Other'] = []

    for idx, name in enumerate(feature_names):
        matched = False
        for tag in token_tags:
            if f'_{tag}_' in name or name.endswith(f'_{tag}'):
                token_importance[tag].append(importances[idx])
                matched = True
                break
        if not matched:
            if re.search(r'_T\d+', name):
                token_importance['AllTokens/Other'].append(importances[idx])
            elif 'AllTokens' in name or 'AllNonZeroTokens' in name or 'Global' in name:
                token_importance['AllTokens/Other'].append(importances[idx])

    print(f"\n{'=' * 60}")
    print(f"  Importance by Token Position")
    print(f"{'=' * 60}")
    print(f"  {'Token':<20s} {'Count':>5s}  {'Sum':>8s}  {'Mean':>8s}  {'Max':>8s}")
    print(f"  {'-'*20} {'-'*5}  {'-'*8}  {'-'*8}  {'-'*8}")
    token_rows = []
    for tag in token_tags:
        imps = token_importance[tag]
        if imps:
            arr = np.array(imps)
            token_rows.append((tag, len(arr), np.sum(arr), np.mean(arr), np.max(arr)))
    token_rows.sort(key=lambda r: r[2], reverse=True)
    for tag, cnt, s, m, mx in token_rows:
        print(f"  {tag:<20s} {cnt:5d}  {s:8.4f}  {m:8.6f}  {mx:8.6f}")
    if token_importance['AllTokens/Other']:
        arr = np.array(token_importance['AllTokens/Other'])
        print(f"  {'AllTokens/Other':<20s} {len(arr):5d}  {np.sum(arr):8.4f}  "
              f"{np.mean(arr):8.6f}  {np.max(arr):8.6f}")

    zero_count = int(np.sum(importances == 0))
    near_zero = int(np.sum(importances < 1e-4))
    print(f"\n  Features with importance == 0: {zero_count}/{n_feats}")
    print(f"  Features with importance < 1e-4: {near_zero}/{n_feats}")


def main():
    parser = argparse.ArgumentParser(description='XGBoost Steering Predictor with ID/OOD Split')
    parser.add_argument('--llm', type=str, default='gemma-2-2b-it')
    parser.add_argument('--steering_method', type=str, default='diffmean')
    parser.add_argument('--curr_layer', type=int, default=10)
    parser.add_argument('--k_list', type=str, default="1,2,3,5,10,15,20,25")
    parser.add_argument('--target_token_n_list', type=str, default="1,5")
    parser.add_argument('--alpha_min', type=float, default=None)
    parser.add_argument('--alpha_max', type=float, default=None)
    parser.add_argument('--test_size', type=float, default=0.3, help='ID test set ratio (default 0.3 for 70/30 split)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='saved_predictors_xgb')
    parser.add_argument('--id_concept_max', type=int, default=119,
                        help='Max concept ID for in-distribution (inclusive). Concepts 0..id_concept_max are ID.')
    parser.add_argument('--ood_concept_min', type=int, default=120,
                        help='Min concept ID for OOD (inclusive). Concepts ood_concept_min..149 are OOD.')
    parser.add_argument('--ood_concept_max', type=int, default=149,
                        help='Max concept ID for OOD (inclusive).')
    parser.add_argument(
        '--force_regenerate_raw',
        action='store_true',
        help='Regenerate cached raw hidden states even if files already exist.',
    )
    parser.add_argument(
        '--classifier',
        type=str,
        default='xgb',
        choices=['xgb', 'mlp_topk'],
        help='Classifier to train. mlp_topk uses XGBoost feature ranking then trains MLP on top-k features.',
    )
    parser.add_argument(
        '--feature_groups', type=str, default=None,
        help='Comma-separated feature groups to use. '
             'Default: all groups (' + ','.join(ALL_FEATURE_GROUPS) + ').',
    )
    parser.add_argument('--mlp_topk', type=int, default=10, help='Top-k features selected by XGBoost for mlp_topk mode.')
    parser.add_argument(
        '--mlp_hidden_layers',
        type=str,
        default='64,32',
        help='Comma-separated hidden layer sizes for MLP, e.g. "64,32".',
    )
    parser.add_argument('--mlp_max_iter', type=int, default=500, help='Max iterations for MLP training.')
    parser.add_argument('--mlp_alpha', type=float, default=1e-4, help='L2 regularization term for MLP.')
    parser.add_argument('--mlp_learning_rate_init', type=float, default=1e-3, help='Initial learning rate for MLP.')
    args = parser.parse_args()

    k_list = [int(x.strip()) for x in args.k_list.split(',')]
    target_token_n_list = [int(x.strip()) for x in args.target_token_n_list.split(',')]
    feature_groups = (
        [g.strip() for g in args.feature_groups.split(',')]
        if args.feature_groups else None
    )

    id_concept_ids = list(range(0, args.id_concept_max + 1))
    ood_concept_ids = list(range(args.ood_concept_min, args.ood_concept_max + 1))

    print(f"Model: {args.llm}, Method: {args.steering_method}, Layer: {args.curr_layer}")
    print(f"Features: k_list={k_list}, n_list={target_token_n_list}")
    print(f"Classifier: {args.classifier}")
    print(f"Feature groups: {feature_groups or 'all (default)'}")
    print(f"ID concepts: 0-{args.id_concept_max} ({len(id_concept_ids)} concepts)")
    print(f"OOD concepts: {args.ood_concept_min}-{args.ood_concept_max} ({len(ood_concept_ids)} concepts)")
    if args.alpha_min is not None or args.alpha_max is not None:
        print(f"Alpha range filter: [{args.alpha_min}, {args.alpha_max}]")

    # ========== Load ID data ==========
    print(f"\n{'='*60}")
    print("  Loading In-Distribution Data (concepts 0-{})".format(args.id_concept_max))
    print(f"{'='*60}")

    id_pairs, id_labels = load_paired_data(
        method=args.steering_method,
        model=args.llm,
        layer=args.curr_layer,
        n_list=target_token_n_list,
        k_list=k_list,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        concept_ids=id_concept_ids,
        force_regenerate_raw=args.force_regenerate_raw,
    )

    if not id_pairs:
        print("No ID data found!")
        return

    print(f"Loaded {len(id_pairs)} ID samples.")

    # Build label map from all ID labels
    label_map = build_label_map(id_labels)
    if label_map:
        print(f"Label map: {label_map}")

    # Extract features for ID data
    print("Extracting ID features...")
    X_id, y_id = extract_features(id_pairs, id_labels, label_map, args.curr_layer, k_list, target_token_n_list,
                                  feature_groups=feature_groups)
    print(f"ID feature matrix: {X_id.shape}, labels: {y_id.shape}")
    print(f"ID label distribution: {dict(zip(*np.unique(y_id, return_counts=True)))}")

    # 70-30 split on ID data
    unique_labels_id = np.unique(y_id)
    stratify = y_id if len(unique_labels_id) > 1 else None

    X_train, X_test_id, y_train, y_test_id = train_test_split(
        X_id, y_id, test_size=args.test_size, random_state=args.seed, stratify=stratify
    )
    print(f"\nID split: Train={len(X_train)}, Test={len(X_test_id)} (test_size={args.test_size})")

    # Oversample training data
    print("Oversampling training data...")
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    print(f"  Before: {dict(zip(unique_train, counts_train))}")
    X_train, y_train = oversample(X_train, y_train, args.seed)
    unique_train2, counts_train2 = np.unique(y_train, return_counts=True)
    print(f"  After:  {dict(zip(unique_train2, counts_train2))}")

    # ========== Train classifier ==========
    if label_map:
        num_class = len(set(label_map.values()))
    else:
        num_class = len(set(y_id))

    xgb_params = {
        'n_estimators': 150,
        'learning_rate': 0.1,
        'max_depth': 5,
        'random_state': args.seed,
        'use_label_encoder': False,
        'eval_metric': 'mlogloss'
    }
    if num_class > 2:
        xgb_params['objective'] = 'multi:softmax'
        xgb_params['num_class'] = num_class
    else:
        xgb_params['objective'] = 'binary:logistic'

    xgb_selector_model = None
    top_indices = None
    scaler = None
    model = None

    if args.classifier == 'xgb':
        print("\nTraining XGBoost...")
        model = xgb.XGBClassifier(**xgb_params)
        model.fit(X_train, y_train)
    else:
        print("\nTraining XGBoost selector for top-k features...")
        xgb_selector_model = xgb.XGBClassifier(**xgb_params)
        xgb_selector_model.fit(X_train, y_train)

        importances = xgb_selector_model.feature_importances_
        top_k = min(args.mlp_topk, X_train.shape[1])
        top_indices = np.argsort(importances)[::-1][:top_k]
        top_indices = np.array(top_indices, dtype=int)

        feature_names = build_feature_names(args.curr_layer, k_list, target_token_n_list,
                                             feature_groups=feature_groups)
        print(f"Selected top-{len(top_indices)} features for MLP:")
        for rank, idx in enumerate(top_indices, start=1):
            feat_name = feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}"
            print(f"  {rank}. {feat_name} (importance={importances[idx]:.4f})")

        hidden_layers = tuple(
            int(v.strip()) for v in args.mlp_hidden_layers.split(',') if v.strip()
        )
        if not hidden_layers:
            hidden_layers = (64, 32)

        X_train_topk = X_train[:, top_indices]
        scaler = StandardScaler()
        X_train_topk_scaled = scaler.fit_transform(X_train_topk)

        print(f"\nTraining MLP on top-{len(top_indices)} XGB-selected features...")
        model = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation='relu',
            solver='adam',
            alpha=args.mlp_alpha,
            learning_rate_init=args.mlp_learning_rate_init,
            max_iter=args.mlp_max_iter,
            random_state=args.seed,
        )
        model.fit(X_train_topk_scaled, y_train)

    # ========== Evaluate on ID test set ==========
    if args.classifier == 'xgb':
        id_acc = evaluate_and_print(model, X_test_id, y_test_id, "In-Distribution Test Set (concepts 0-{})".format(args.id_concept_max))
    else:
        X_test_id_topk = X_test_id[:, top_indices]
        X_test_id_topk_scaled = scaler.transform(X_test_id_topk)
        id_acc = evaluate_and_print(model, X_test_id_topk_scaled, y_test_id, "In-Distribution Test Set (concepts 0-{})".format(args.id_concept_max))

    # ========== Load OOD data ==========
    print(f"\n{'='*60}")
    print("  Loading OOD Data (concepts {}-{})".format(args.ood_concept_min, args.ood_concept_max))
    print(f"{'='*60}")

    ood_pairs, ood_labels = load_paired_data(
        method=args.steering_method,
        model=args.llm,
        layer=args.curr_layer,
        n_list=target_token_n_list,
        k_list=k_list,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        concept_ids=ood_concept_ids,
        force_regenerate_raw=args.force_regenerate_raw,
    )

    if not ood_pairs:
        print("No OOD data found!")
    else:
        print(f"Loaded {len(ood_pairs)} OOD samples.")

        # Extend label_map for any unseen labels in OOD
        ood_unique = set(ood_labels)
        if label_map:
            new_labels = ood_unique - set(label_map.keys())
            if new_labels:
                next_id = max(label_map.values()) + 1
                for lbl in sorted(new_labels):
                    label_map[lbl] = next_id
                    next_id += 1
                print(f"Extended label map for OOD: {label_map}")

        print("Extracting OOD features...")
        X_ood, y_ood = extract_features(ood_pairs, ood_labels, label_map, args.curr_layer, k_list, target_token_n_list,
                                        feature_groups=feature_groups)
        print(f"OOD feature matrix: {X_ood.shape}, labels: {y_ood.shape}")
        print(f"OOD label distribution: {dict(zip(*np.unique(y_ood, return_counts=True)))}")

        if args.classifier == 'xgb':
            ood_acc = evaluate_and_print(model, X_ood, y_ood, "OOD Test Set (concepts {}-{})".format(args.ood_concept_min, args.ood_concept_max))
        else:
            X_ood_topk = X_ood[:, top_indices]
            X_ood_topk_scaled = scaler.transform(X_ood_topk)
            ood_acc = evaluate_and_print(model, X_ood_topk_scaled, y_ood, "OOD Test Set (concepts {}-{})".format(args.ood_concept_min, args.ood_concept_max))

    # ========== Feature Importances ==========
    if args.classifier == 'xgb':
        print_feature_importances(model, args.curr_layer, k_list, target_token_n_list,
                                  feature_groups=feature_groups)
    else:
        print_feature_importances(xgb_selector_model, args.curr_layer, k_list, target_token_n_list,
                                  feature_groups=feature_groups)

    # ========== Summary ==========
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"Classifier: {args.classifier}")
    print(f"ID Test Accuracy  (concepts 0-{args.id_concept_max}):   {id_acc:.4f}")
    if ood_pairs:
        print(f"OOD Test Accuracy (concepts {args.ood_concept_min}-{args.ood_concept_max}): {ood_acc:.4f}")

    # ========== Save model ==========
    os.makedirs(args.save_dir, exist_ok=True)
    if args.classifier == 'xgb':
        save_path = os.path.join(args.save_dir, f'xgb_{args.llm}_{args.steering_method}_l{args.curr_layer}.json')
        model.save_model(save_path)
        print(f"\nModel saved to {save_path}")
    else:
        save_path = os.path.join(
            args.save_dir,
            f'mlp_topk_{args.llm}_{args.steering_method}_l{args.curr_layer}.pkl'
        )
        payload = {
            'classifier': 'mlp_topk',
            'mlp_model': model,
            'xgb_selector_model': xgb_selector_model,
            'top_indices': top_indices,
            'scaler': scaler,
            'k_list': k_list,
            'target_token_n_list': target_token_n_list,
            'curr_layer': args.curr_layer,
            'mlp_hidden_layers': args.mlp_hidden_layers,
            'mlp_max_iter': args.mlp_max_iter,
            'mlp_alpha': args.mlp_alpha,
            'mlp_learning_rate_init': args.mlp_learning_rate_init,
        }
        with open(save_path, 'wb') as f:
            pickle.dump(payload, f)
        print(f"\nMLP-topk model bundle saved to {save_path}")

    if label_map:
        map_prefix = 'xgb' if args.classifier == 'xgb' else 'mlp_topk'
        map_path = os.path.join(args.save_dir, f'{map_prefix}_{args.llm}_{args.steering_method}_l{args.curr_layer}_labelmap.pkl')
        with open(map_path, 'wb') as f:
            pickle.dump(label_map, f)
        print(f"Label map saved to {map_path}")


if __name__ == '__main__':
    main()
