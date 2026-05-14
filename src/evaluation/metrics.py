# metrics.py
"""
Evaluation Metrics for PatchCore
==================================
This file answers the two key questions about our defect detector:
  1. Image AUROC  → "Did we correctly flag defective images?"
  2. Pixel AUROC  → "Did we correctly locate WHERE the defect is?"
  3. PRO          → "How well do we overlap with the actual defect region?"

AUROC (Area Under the ROC Curve) ranges from 0 to 1:
  - 1.0 = perfect detection
  - 0.5 = random guessing

PRO (Per-Region Overlap) is stricter — it measures spatial overlap
per connected defect region, integrated only up to 30% FPR.

How it fits: scoring.py produces scores + heatmaps → THIS FILE evaluates them.
"""

import logging
from typing import Dict, List

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. image_auroc
# ---------------------------------------------------------------------------
def image_auroc(
    image_scores: List[float],
    image_labels: List[int],
) -> float:
    """Compute image-level AUROC.

    Each image has one anomaly score (from the max of its heatmap) and
    one binary label (0=good, 1=defective). AUROC measures how well
    the scores separate good from defective images.

    Args:
        image_scores: Per-image anomaly scores.
        image_labels: Ground-truth labels (0 or 1).

    Returns:
        AUROC as a float in [0, 1].
    """
    if len(set(image_labels)) < 2:
        logger.warning("Only one class present — AUROC is undefined. Returning 0.0.")
        return 0.0

    score = roc_auc_score(image_labels, image_scores)
    logger.info("Image AUROC: %.4f", score)
    return float(score)


# ---------------------------------------------------------------------------
# 2. pixel_auroc
# ---------------------------------------------------------------------------
def pixel_auroc(
    heatmaps: List[np.ndarray],
    gt_masks: List[np.ndarray],
) -> float:
    """Compute pixel-level AUROC.

    Every pixel in the heatmap has an anomaly score and every pixel in
    the ground-truth mask has a binary label. We flatten ALL pixels
    across ALL test images and compute one global AUROC.

    Args:
        heatmaps: List of (H, W) anomaly heatmaps.
        gt_masks: List of (H, W) binary ground-truth masks.

    Returns:
        Pixel AUROC as a float in [0, 1].
    """
    all_scores: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    for heatmap, mask in zip(heatmaps, gt_masks):
        all_scores.append(heatmap.flatten())
        all_labels.append(mask.flatten())

    all_scores_flat = np.concatenate(all_scores)
    all_labels_flat = np.concatenate(all_labels)

    # Binarise labels (in case they're not already exactly 0/1)
    all_labels_flat = (all_labels_flat > 0.5).astype(np.int32)

    if len(np.unique(all_labels_flat)) < 2:
        logger.warning("Only one pixel class present — pixel AUROC undefined. Returning 0.0.")
        return 0.0

    score = roc_auc_score(all_labels_flat, all_scores_flat)
    logger.info("Pixel AUROC: %.4f", score)
    return float(score)


# ---------------------------------------------------------------------------
# 3. compute_pro
# ---------------------------------------------------------------------------
def compute_pro(
    heatmaps: List[np.ndarray],
    gt_masks: List[np.ndarray],
    num_thresholds: int = 200,
    integration_limit: float = 0.3,
) -> float:
    """Compute Per-Region Overlap (PRO) score.

    PRO is a stricter localisation metric than pixel AUROC. It:
      1. Identifies each connected defect region in the ground truth.
      2. At each threshold, computes the overlap (TPR) per region.
      3. Averages the per-region overlaps.
      4. Plots average-overlap vs FPR and integrates the curve
         only up to FPR = integration_limit (default 30%).
      5. Normalises by the integration range.

    This penalises detectors that miss small defects even if they
    catch large ones, because every region counts equally.

    Args:
        heatmaps: List of (H, W) anomaly heatmaps.
        gt_masks: List of (H, W) binary ground-truth masks.
        num_thresholds: Number of thresholds to sweep.
        integration_limit: Max FPR for integration (default 0.3).

    Returns:
        Normalised PRO score in [0, 1].
    """
    from scipy.ndimage import label as connected_components

    # Collect all heatmap values to determine threshold range
    all_scores = np.concatenate([h.flatten() for h in heatmaps])
    thresholds = np.linspace(all_scores.min(), all_scores.max(), num_thresholds)

    # For each image, find connected components in the ground truth
    pro_fprs: List[float] = []
    pro_pros: List[float] = []

    for thresh in thresholds:
        per_region_overlaps: List[float] = []
        total_fp = 0
        total_tn = 0

        for heatmap, mask in zip(heatmaps, gt_masks):
            binary_mask = (mask > 0.5).astype(np.int32)
            prediction = (heatmap >= thresh).astype(np.int32)

            # FPR components
            neg_mask = 1 - binary_mask
            total_fp += np.sum(prediction * neg_mask)
            total_tn += np.sum(neg_mask)

            # Per-region overlap
            labeled_mask, num_regions = connected_components(binary_mask)
            for region_id in range(1, num_regions + 1):
                region = (labeled_mask == region_id)
                region_size = region.sum()
                if region_size == 0:
                    continue
                overlap = np.sum(prediction[region]) / region_size
                per_region_overlaps.append(overlap)

        # Average overlap across all regions at this threshold
        if len(per_region_overlaps) > 0:
            avg_overlap = float(np.mean(per_region_overlaps))
        else:
            avg_overlap = 0.0

        # FPR at this threshold
        fpr = total_fp / max(total_tn, 1)

        pro_fprs.append(fpr)
        pro_pros.append(avg_overlap)

    # Sort by FPR
    sorted_pairs = sorted(zip(pro_fprs, pro_pros))
    pro_fprs = [p[0] for p in sorted_pairs]
    pro_pros = [p[1] for p in sorted_pairs]

    # Integrate up to integration_limit
    pro_fprs = np.array(pro_fprs)
    pro_pros = np.array(pro_pros)

    # Clip to integration range
    valid = pro_fprs <= integration_limit
    if valid.sum() < 2:
        logger.warning("Not enough points under FPR=%.2f — PRO returning 0.0.", integration_limit)
        return 0.0

    clipped_fprs = pro_fprs[valid]
    clipped_pros = pro_pros[valid]

    # Trapezoidal integration, normalised by the range
    pro_score = float(np.trapz(clipped_pros, clipped_fprs) / integration_limit)
    logger.info("PRO score (limit=%.2f): %.4f", integration_limit, pro_score)
    return pro_score


# ---------------------------------------------------------------------------
# 4. evaluate_all
# ---------------------------------------------------------------------------
def evaluate_all(
    image_scores: List[float],
    image_labels: List[int],
    heatmaps: List[np.ndarray],
    gt_masks: List[np.ndarray],
) -> Dict[str, float]:
    """Run all evaluation metrics and return them in a dictionary.

    Args:
        image_scores: Per-image anomaly scores.
        image_labels: Ground-truth labels (0 or 1).
        heatmaps: List of (H, W) anomaly heatmaps.
        gt_masks: List of (H, W) binary ground-truth masks.

    Returns:
        Dict with keys 'image_auroc', 'pixel_auroc', 'pro'.
    """
    results: Dict[str, float] = {
        "image_auroc": image_auroc(image_scores, image_labels),
        "pixel_auroc": pixel_auroc(heatmaps, gt_masks),
        "pro": compute_pro(heatmaps, gt_masks),
    }

    logger.info(
        "Evaluation complete — Image AUROC: %.4f | Pixel AUROC: %.4f | PRO: %.4f",
        results["image_auroc"], results["pixel_auroc"], results["pro"],
    )
    return results
