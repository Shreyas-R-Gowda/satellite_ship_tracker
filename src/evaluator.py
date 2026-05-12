"""Evaluation metrics for synthetic ship tracking runs."""

import numpy as np


def compute_metrics(
    gt: dict,
    pred: dict,
    distance_threshold: float = 25.0,
) -> dict:
    """Compare predicted ship centers against ground-truth centers per frame."""
    tp = 0
    fp = 0
    fn = 0
    matched_distances = []

    for frame_id in sorted(set(gt) | set(pred)):
        gt_centers = gt.get(frame_id, [])
        pred_centers = pred.get(frame_id, [])
        matched_gt = set()
        matched_pred = set()

        pairs = []
        for gi, gt_center in enumerate(gt_centers):
            for pi, pred_center in enumerate(pred_centers):
                dist = float(np.linalg.norm(np.array(gt_center) - np.array(pred_center)))
                pairs.append((dist, gi, pi))

        for dist, gi, pi in sorted(pairs, key=lambda item: item[0]):
            if dist >= distance_threshold or gi in matched_gt or pi in matched_pred:
                continue
            matched_gt.add(gi)
            matched_pred.add(pi)
            matched_distances.append(dist)

        tp += len(matched_gt)
        fp += len(pred_centers) - len(matched_pred)
        fn += len(gt_centers) - len(matched_gt)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    avg_error = float(np.mean(matched_distances)) if matched_distances else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "avg_center_error_px": avg_error,
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
    }
