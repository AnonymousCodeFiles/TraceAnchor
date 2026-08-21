from __future__ import annotations

from collections import defaultdict

import numpy as np


def average_precision(targets: np.ndarray, scores: np.ndarray) -> float:
    targets = np.asarray(targets, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    positive_count = int(targets.sum())
    if positive_count == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    ordered = targets[order]
    cumulative_positive = np.cumsum(ordered)
    precision = cumulative_positive / np.arange(1, len(ordered) + 1)
    return float(precision[ordered == 1].sum() / positive_count)


def scene_macro_metrics(
    targets: np.ndarray,
    scores: np.ndarray,
    scene_indices: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for index, scene_index in enumerate(scene_indices):
        grouped[int(scene_index)].append(index)
    aps = []
    f1s = []
    for indices in grouped.values():
        scene_targets = targets[indices]
        scene_scores = scores[indices]
        ap = average_precision(scene_targets, scene_scores)
        if not np.isnan(ap):
            aps.append(ap)
        predicted = scene_scores >= threshold
        tp = int(np.logical_and(predicted, scene_targets == 1).sum())
        fp = int(np.logical_and(predicted, scene_targets == 0).sum())
        fn = int(np.logical_and(~predicted, scene_targets == 1).sum())
        denominator = 2 * tp + fp + fn
        f1s.append(0.0 if denominator == 0 else (2.0 * tp) / denominator)
    positive_mask = targets == 1
    event_recall = (
        0.0
        if not positive_mask.any()
        else float((scores[positive_mask] >= threshold).mean())
    )
    return {
        "scene_macro_pr_auc": float(np.mean(aps)) if aps else float("nan"),
        "scene_macro_f1": float(np.mean(f1s)) if f1s else float("nan"),
        "event_recall": event_recall,
        "effective_scenes": len(aps),
        "labeled_windows": int(len(targets)),
    }


__all__ = ["average_precision", "scene_macro_metrics"]
