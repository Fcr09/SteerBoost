import torch
import numpy as np

ALL_FEATURE_GROUPS = [
    "MagDiff", "CosSim", "SteerAlign",
    "MagDiffRatio", "CosSimDelta", "SteerAlignDelta",
    "AfterSteering", "SteerProj",
]

DEFAULT_FEATURE_GROUPS = list(ALL_FEATURE_GROUPS)


def get_handcrafted_features(steered_sample, raw_sample, curr_layer, k_list, n_list,
                             alpha=None, feature_groups=None):
    """
    Extract handcrafted features from paired hidden states.

    Args:
        steered_sample: dict of hidden states from steering
        raw_sample: dict of hidden states from raw model
        curr_layer: int, current layer index (L)
        k_list: list of ints, offsets for L+k layers
        n_list: list of ints, token indices
        alpha: float or None, steering strength (appended as a feature if provided)
        feature_groups: list of str or None.  If None, DEFAULT_FEATURE_GROUPS.

    Feature groups (canonical order):
      MagDiff:         ||s_before - r_before|| per (token, layer) + stats
      CosSim:          cos(s_before, r_before) per (token, layer) + stats
      SteerAlign:      cos(s_before - r_before, steer_vec) per (token, layer)
                       *excludes* the intervention layer L (always zero) + stats
      SteerProj:       cos(h, steer_vec) where h = after_steering at L,
                       before_steering elsewhere, per (token, layer) + stats
      MagDiffRatio:    MagDiff(token_N) / MagDiff(token_0) per layer + stats
      CosSimDelta:     CosSim(token_N) - CosSim(token_0) per layer + stats
      SteerAlignDelta: SteerAlign(token_N) - SteerAlign(token_0), excludes L + stats
      AfterSteering:   InterventionMag/alpha (single scalar)

    Alpha is always appended when provided (not controlled by feature_groups).
    """
    if feature_groups is None:
        feature_groups = DEFAULT_FEATURE_GROUPS
    active = set(feature_groups)

    features = []

    token_keys = ['token_0'] + [f'token_{n}' for n in n_list]
    token_keys = sorted(list(set(token_keys)), key=lambda x: int(x.split('_')[1]))

    layer_indices = [curr_layer] + [curr_layer + k for k in k_list]
    layer_indices = sorted(list(set(layer_indices)))
    layer_keys = [f'layer_{l}' for l in layer_indices]

    nT = len(token_keys)
    nL = len(layer_keys)

    l0_col = layer_indices.index(curr_layer)
    non_l0_cols = [i for i in range(nL) if i != l0_col]

    EPS = 1e-8

    diff_matrix = np.zeros((nT, nL))
    cossim_matrix = np.zeros((nT, nL))
    steer_align_matrix = np.zeros((nT, nL))
    steer_proj_matrix = np.zeros((nT, nL))

    after_layer_key = f'layer_{curr_layer}'
    steer_vec = None
    steer_vec_norm = 0.0
    for token_key in token_keys:
        if token_key in steered_sample:
            s_after = steered_sample[token_key].get('after_steering', {})
            s_before_tmp = steered_sample[token_key].get('before_steering', {})
            if after_layer_key in s_after and after_layer_key in s_before_tmp:
                sa = s_after[after_layer_key].float()
                sb = s_before_tmp[after_layer_key].float()
                if sa.device != sb.device:
                    sb = sb.to(sa.device)
                if sa.shape == sb.shape:
                    steer_vec = (sa - sb).view(-1)
                    steer_vec_norm = torch.norm(steer_vec, p=2).item()
                    break

    need_diff = any(g in active for g in ("MagDiff", "MagDiffRatio"))
    need_cos = any(g in active for g in ("CosSim", "CosSimDelta"))
    need_align = any(g in active for g in ("SteerAlign", "SteerAlignDelta"))
    need_proj = "SteerProj" in active

    for t_idx, token_key in enumerate(token_keys):
        for l_idx, layer_key in enumerate(layer_keys):
            if token_key not in steered_sample or token_key not in raw_sample:
                continue

            s_before = steered_sample[token_key].get('before_steering', {})
            r_before = raw_sample[token_key].get('before_steering', {})

            if layer_key not in s_before or layer_key not in r_before:
                continue

            s_tensor = s_before[layer_key].float()
            r_tensor = r_before[layer_key].float()

            if s_tensor.device != r_tensor.device:
                r_tensor = r_tensor.to(s_tensor.device)

            if s_tensor.shape != r_tensor.shape:
                continue

            diff = s_tensor - r_tensor

            if need_diff:
                diff_matrix[t_idx, l_idx] = torch.norm(diff, p=2).item()

            if need_cos:
                s_flat = s_tensor.view(-1)
                r_flat = r_tensor.view(-1)
                dot_product = torch.dot(s_flat, r_flat).item()
                norm_s = torch.norm(s_flat, p=2).item()
                norm_r = torch.norm(r_flat, p=2).item()
                if norm_s > EPS and norm_r > EPS:
                    cossim_matrix[t_idx, l_idx] = dot_product / (norm_s * norm_r)

            if need_align and steer_vec is not None and steer_vec_norm > EPS:
                diff_flat = diff.view(-1)
                if diff_flat.shape == steer_vec.shape:
                    diff_norm = torch.norm(diff_flat, p=2).item()
                    if diff_norm > EPS:
                        steer_align_matrix[t_idx, l_idx] = (
                            torch.dot(diff_flat, steer_vec).item()
                            / (diff_norm * steer_vec_norm)
                        )

            if need_proj and steer_vec is not None and steer_vec_norm > EPS:
                if layer_indices[l_idx] == curr_layer:
                    s_after_dict = steered_sample[token_key].get('after_steering', {})
                    if layer_key in s_after_dict:
                        h = s_after_dict[layer_key].float().view(-1)
                        if h.shape[0] == steer_vec.shape[0]:
                            h_norm = torch.norm(h, p=2).item()
                            if h_norm > EPS:
                                steer_proj_matrix[t_idx, l_idx] = (
                                    torch.dot(h, steer_vec).item()
                                    / (h_norm * steer_vec_norm)
                                )
                else:
                    h = s_tensor.view(-1)
                    if h.shape[0] == steer_vec.shape[0]:
                        h_norm = torch.norm(h, p=2).item()
                        if h_norm > EPS:
                            steer_proj_matrix[t_idx, l_idx] = (
                                torch.dot(h, steer_vec).item()
                                / (h_norm * steer_vec_norm)
                            )

    def _emit_matrix(matrix):
        features.extend(matrix.flatten())
        flat = matrix.flatten()
        if len(flat) > 0:
            features.append(np.mean(flat))
            features.append(np.std(flat))
            features.append(np.max(flat))
            features.append(np.min(flat))
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        for ti in range(matrix.shape[0]):
            row = matrix[ti, :]
            features.append(np.mean(row))
            features.append(np.std(row))
            features.append(np.max(row))
        for li in range(matrix.shape[1]):
            col = matrix[:, li]
            features.append(np.mean(col))
            features.append(np.std(col))
            features.append(np.max(col))

    # --- base matrices (canonical order) ---
    if "MagDiff" in active:
        _emit_matrix(diff_matrix)
    if "CosSim" in active:
        _emit_matrix(cossim_matrix)
    if "SteerAlign" in active:
        _emit_matrix(steer_align_matrix[:, non_l0_cols])
    if "SteerProj" in active:
        _emit_matrix(steer_proj_matrix)

    # --- cross-token propagation ---
    if nT > 1:
        non_zero_count = nT - 1
        magdiff_ratio = np.zeros((non_zero_count, nL))
        cossim_delta = np.zeros((non_zero_count, nL))
        steer_align_delta = np.zeros((non_zero_count, nL))

        for i, t_idx in enumerate(range(1, nT)):
            for l_idx in range(nL):
                base_diff = diff_matrix[0, l_idx]
                if abs(base_diff) > EPS:
                    magdiff_ratio[i, l_idx] = diff_matrix[t_idx, l_idx] / base_diff
                cossim_delta[i, l_idx] = (
                    cossim_matrix[t_idx, l_idx] - cossim_matrix[0, l_idx]
                )
                steer_align_delta[i, l_idx] = (
                    steer_align_matrix[t_idx, l_idx] - steer_align_matrix[0, l_idx]
                )

        if "MagDiffRatio" in active:
            _emit_matrix(magdiff_ratio)
        if "CosSimDelta" in active:
            _emit_matrix(cossim_delta)
        if "SteerAlignDelta" in active:
            _emit_matrix(steer_align_delta[:, non_l0_cols])

    # --- after-steering features ---
    if "AfterSteering" in active:
        intervention_mag = steer_vec_norm if steer_vec is not None else 0.0
        if alpha is not None and alpha != 0:
            features.append(intervention_mag / alpha)
        else:
            features.append(intervention_mag)

    # --- alpha (always appended when provided) ---
    if alpha is not None:
        features.append(float(alpha))

    return np.array(features)
