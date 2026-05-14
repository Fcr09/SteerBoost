"""
Stratified ID vs OOD concept splits for 150 concepts (indices 0–149).

Levels (abstraction):
  low:  0–39   and 120–129  (50 concepts)
  mid:  40–79  and 130–139  (50 concepts)
  high: 80–119 and 140–149  (50 concepts)

Default stratified shuffle: per level, randomly assign 40 concepts to ID (train)
and 10 to OOD (test), preserving 40/10/level and 120 ID / 30 OOD overall.

Metadata is stored under splits.json["concept_partition"] for downstream tools
(alphasearch.py, etc.).
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Pools cover all 150 concepts exactly once across three levels.
LEVEL_POOLS: Dict[str, List[int]] = {
    "low": list(range(0, 40)) + list(range(120, 130)),
    "mid": list(range(40, 80)) + list(range(130, 140)),
    "high": list(range(80, 120)) + list(range(140, 150)),
}

N_ID_PER_LEVEL = 40
N_OOD_PER_LEVEL = 10


def build_static_partition(
    id_concept_max: int,
    ood_concept_min: int,
    ood_concept_max: int,
) -> Tuple[List[int], List[int], Dict[str, Any]]:
    """Fixed contiguous ranges (legacy behavior): ID 0..id_concept_max, OOD ood_min..ood_max."""
    id_concepts = list(range(0, id_concept_max + 1))
    ood_concepts = list(range(ood_concept_min, ood_concept_max + 1))
    meta: Dict[str, Any] = {
        "scheme": "static_ranges",
        "id_concept_max": id_concept_max,
        "ood_concept_min": ood_concept_min,
        "ood_concept_max": ood_concept_max,
        "id_concepts": id_concepts,
        "ood_concepts": ood_concepts,
    }
    return id_concepts, ood_concepts, meta


def build_stratified_shuffle_partition(
    seed: int,
) -> Tuple[List[int], List[int], Dict[str, Any]]:
    """
    Shuffle within each level; assign N_ID_PER_LEVEL to ID and N_OOD_PER_LEVEL to OOD.
    """
    rng = np.random.default_rng(seed)
    levels_out: Dict[str, Any] = {}
    id_all: List[int] = []
    ood_all: List[int] = []

    for level_name, pool in LEVEL_POOLS.items():
        if len(pool) != N_ID_PER_LEVEL + N_OOD_PER_LEVEL:
            raise ValueError(
                f"Level {level_name}: expected {N_ID_PER_LEVEL + N_OOD_PER_LEVEL} "
                f"concepts, got {len(pool)}"
            )
        order = rng.permutation(len(pool)).tolist()
        shuffled = [pool[i] for i in order]
        id_part = sorted(shuffled[:N_ID_PER_LEVEL])
        ood_part = sorted(shuffled[N_ID_PER_LEVEL:])
        id_all.extend(id_part)
        ood_all.extend(ood_part)
        levels_out[level_name] = {
            "pool": list(pool),
            "id": id_part,
            "ood": ood_part,
        }

    id_concepts = sorted(id_all)
    ood_concepts = sorted(ood_all)
    meta: Dict[str, Any] = {
        "scheme": "stratified_shuffle",
        "seed": seed,
        "n_id_per_level": N_ID_PER_LEVEL,
        "n_ood_per_level": N_OOD_PER_LEVEL,
        "levels": levels_out,
        "id_concepts": id_concepts,
        "ood_concepts": ood_concepts,
    }
    return id_concepts, ood_concepts, meta


def concept_to_level_map() -> Dict[int, str]:
    """Canonical mapping from concept id to abstraction level name."""
    m: Dict[int, str] = {}
    for level_name, pool in LEVEL_POOLS.items():
        for c in pool:
            m[c] = level_name
    return m


def load_concept_partition_from_splits_json(
    path: str,
) -> Optional[Tuple[List[int], List[int], Dict[str, Any]]]:
    """
    If splits.json contains concept_partition with id_concepts / ood_concepts,
    return (id_concepts, ood_concepts, partition_meta). Otherwise None.
    """
    if not path or not os.path.isfile(path):
        return None
    with open(path) as f:
        data = json.load(f)
    cp = data.get("concept_partition")
    if not isinstance(cp, dict):
        return None
    raw_id = cp.get("id_concepts")
    raw_ood = cp.get("ood_concepts")
    if raw_id is None or raw_ood is None:
        return None
    id_concepts = sorted(int(x) for x in raw_id)
    ood_concepts = sorted(int(x) for x in raw_ood)
    return id_concepts, ood_concepts, cp
