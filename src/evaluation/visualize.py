# visualize.py
"""
Visualisation Utilities for PatchCore
=======================================
This file creates the visual outputs that make the results human-readable:
  - Heatmap overlays on original images
  - Contact-sheet grids comparing input / ground truth / prediction
  - ROC curves
  - Score histograms

Why it exists: Numbers alone (AUROC = 0.98) don't tell the full story.
Seeing WHERE the model thinks the defect is builds trust and helps
debug failure cases.

How it fits: scoring.py + metrics.py → THIS FILE → saved figures on Drive.
"""

import logging
import os
from typing import List

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for Colab / headless
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. overlay_heatmap
# ---------------------------------------------------------------------------
def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """Blend a colormapped heatmap onto the original image.

    Steps:
      1. Normalise the heatmap to [0, 1].
      2. Apply a 'jet' colormap (blue=normal → red=anomalous).
      3. Alpha-blend with the original image.

    Args:
        image: Original image as (H, W, 3) float array in [0, 1],
               or uint8 in [0, 255].
        heatmap: (H, W) anomaly heatmap (any range).
        alpha: Heatmap opacity (0 = invisible, 1 = fully opaque).

    Returns:
        Blended image as (H, W, 3) uint8 array.
    """
    # Ensure image is float [0, 1]
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0

    # Normalise heatmap to [0, 1]
    h_min, h_max = heatmap.min(), heatmap.max()
    if h_max - h_min > 1e-8:
        heatmap_norm = (heatmap - h_min) / (h_max - h_min)
    else:
        heatmap_norm = np.zeros_like(heatmap)

    # Apply colormap (returns RGBA)
    colormap = plt.cm.jet(heatmap_norm)[:, :, :3]  # drop alpha → (H, W, 3)

    # Blend
    blended = (1 - alpha) * image + alpha * colormap
    blended = np.clip(blended * 255, 0, 255).astype(np.uint8)

    return blended


# ---------------------------------------------------------------------------
# 2. save_qualitative_grid
# ---------------------------------------------------------------------------
def save_qualitative_grid(
    images: List[np.ndarray],
    gt_masks: List[np.ndarray],
    heatmaps: List[np.ndarray],
    save_path: str,
    max_rows: int = 8,
) -> None:
    """Save a contact-sheet comparing original, ground truth, and prediction.

    Each row shows: Original | GT Mask | Heatmap | Overlay
    Useful for quickly eyeballing detection quality.

    Args:
        images: List of (H, W, 3) original images (uint8 or float).
        gt_masks: List of (H, W) binary ground-truth masks.
        heatmaps: List of (H, W) predicted anomaly heatmaps.
        save_path: Where to save the figure (e.g. .png).
        max_rows: Maximum number of rows to display.
    """
    n_rows = min(len(images), max_rows)
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, 4 * n_rows))

    if n_rows == 1:
        axes = axes[np.newaxis, :]  # ensure 2D indexing

    col_titles = ["Original", "Ground Truth", "Heatmap", "Overlay"]
    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontsize=14, fontweight="bold")

    for row in range(n_rows):
        img = images[row]
        mask = gt_masks[row]
        hmap = heatmaps[row]

        # Ensure image is displayable
        if img.dtype != np.uint8:
            img_display = np.clip(img * 255, 0, 255).astype(np.uint8)
        else:
            img_display = img

        overlay = overlay_heatmap(img, hmap)

        axes[row, 0].imshow(img_display)
        axes[row, 1].imshow(mask, cmap="gray", vmin=0, vmax=1)
        axes[row, 2].imshow(hmap, cmap="jet")
        axes[row, 3].imshow(overlay)

        for col in range(4):
            axes[row, col].axis("off")

    plt.tight_layout()

    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Qualitative grid saved to %s", save_path)
    except Exception as exc:
        logger.error("Failed to save grid to %s: %s", save_path, exc)
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# 3. plot_roc_curve
# ---------------------------------------------------------------------------
def plot_roc_curve(
    scores: List[float],
    labels: List[int],
    save_path: str,
    title: str = "ROC Curve",
) -> None:
    """Plot and save the ROC curve.

    Args:
        scores: Per-image anomaly scores.
        labels: Ground-truth binary labels.
        save_path: Where to save the figure.
        title: Plot title.
    """
    if len(set(labels)) < 2:
        logger.warning("Only one class present — cannot plot ROC curve.")
        return

    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(fpr, tpr, color="#e74c3c", lw=2,
            label=f"AUROC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--",
            label="Random (0.5)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(alpha=0.3)

    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("ROC curve saved to %s (AUROC=%.4f)", save_path, roc_auc)
    except Exception as exc:
        logger.error("Failed to save ROC curve: %s", exc)
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# 4. save_score_histogram
# ---------------------------------------------------------------------------
def save_score_histogram(
    scores: List[float],
    labels: List[int],
    save_path: str,
    title: str = "Anomaly Score Distribution",
) -> None:
    """Plot overlapping histograms of anomaly scores for good vs defective.

    A well-trained model will show two clearly separated distributions.
    If they overlap heavily, the model is struggling to distinguish.

    Args:
        scores: Per-image anomaly scores.
        labels: Ground-truth binary labels (0=good, 1=defective).
        save_path: Where to save the figure.
        title: Plot title.
    """
    scores_arr = np.array(scores)
    labels_arr = np.array(labels)

    good_scores = scores_arr[labels_arr == 0]
    bad_scores = scores_arr[labels_arr == 1]

    fig, ax = plt.subplots(figsize=(8, 5))

    if len(good_scores) > 0:
        ax.hist(good_scores, bins=30, alpha=0.6, color="#2ecc71",
                label=f"Good (n={len(good_scores)})", edgecolor="white")
    if len(bad_scores) > 0:
        ax.hist(bad_scores, bins=30, alpha=0.6, color="#e74c3c",
                label=f"Defective (n={len(bad_scores)})", edgecolor="white")

    ax.set_xlabel("Anomaly Score", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, axis="y")

    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Score histogram saved to %s", save_path)
    except Exception as exc:
        logger.error("Failed to save histogram: %s", exc)
    finally:
        plt.close(fig)
