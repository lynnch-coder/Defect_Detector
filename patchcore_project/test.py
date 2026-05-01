# test.py
"""
Testing / Inference Pipeline for PatchCore
============================================
This is the entry point for evaluating the trained defect detector.
It loads the saved memory banks, runs every test image through the
backbone, scores each patch against the memory bank, computes metrics
(Image AUROC, Pixel AUROC, PRO), and saves visualisations.

How to run: just call main() — it processes every category and writes
a final comparison report.

Pipeline per category:
  memory_bank (load) → backbone + hooks → feature_extractor →
  scoring → metrics + visualize → saved results on Drive
"""

import json
import logging
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image as PILImage
import torchvision.transforms as T

import config as cfg
from data.dataset import MVTecDataset, get_dataloader, get_transform
from models.backbone import load_backbone, register_hooks, remove_hooks
from core.feature_extractor import (
    extract_layer_features,
    locally_aware_patches,
    align_and_concat,
    flatten_patches,
)
from core.memory_bank import load_memory_bank, build_faiss_index
from core.scoring import score_dataset
from evaluation.metrics import evaluate_all
from evaluation.visualize import (
    overlay_heatmap,
    save_qualitative_grid,
    plot_roc_curve,
    save_score_histogram,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: extract test features one image at a time
# ---------------------------------------------------------------------------
def _extract_test_features(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    hook_dict: Dict[str, torch.Tensor],
    device: str,
) -> Tuple[List[torch.Tensor], List[Tuple[int, int]], List[int], List[torch.Tensor], List[np.ndarray]]:
    """Extract patch features for every test image.

    Unlike training, we keep features separated per image (not concatenated)
    because scoring needs to know which patches belong to which image.

    Args:
        model: Frozen backbone on device.
        dataloader: Test dataloader.
        hook_dict: Hook dictionary attached to the model.
        device: 'cuda' or 'cpu'.

    Returns:
        all_features: List of (H*W, D) tensors, one per image.
        all_shapes: List of (H, W) spatial shapes.
        all_labels: List of ground-truth labels (0 or 1).
        all_masks: List of ground-truth mask tensors.
        all_images: List of original images as numpy arrays (for visualisation).
    """
    all_features: List[torch.Tensor] = []
    all_shapes: List[Tuple[int, int]] = []
    all_labels: List[int] = []
    all_masks: List[torch.Tensor] = []
    all_images: List[np.ndarray] = []

    # Inverse normalisation for visualisation
    inv_normalize = T.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
    )

    for images, masks, labels, paths in tqdm(dataloader, desc="Extracting test features"):
        images = images.to(device)

        with torch.no_grad():
            feat_l2, feat_l3 = extract_layer_features(model, images, hook_dict)
            feat_l2 = locally_aware_patches(feat_l2)
            feat_l3 = locally_aware_patches(feat_l3)
            combined = align_and_concat(feat_l2, feat_l3)

        # Process each image in the batch individually
        for i in range(combined.shape[0]):
            single = combined[i].unsqueeze(0)  # (1, C, H, W)
            flat, spatial_shape = flatten_patches(single)
            all_features.append(flat.cpu())
            all_shapes.append(spatial_shape)
            all_labels.append(int(labels[i]))
            all_masks.append(masks[i].cpu())

            # De-normalise image for visualisation
            img_vis = inv_normalize(images[i].cpu())
            img_vis = img_vis.permute(1, 2, 0).numpy()
            img_vis = np.clip(img_vis, 0, 1)
            all_images.append(img_vis)

    logger.info("Extracted features for %d test images.", len(all_features))
    return all_features, all_shapes, all_labels, all_masks, all_images


# ---------------------------------------------------------------------------
# 1. run_patchcore_inference
# ---------------------------------------------------------------------------
def run_patchcore_inference(
    category: str,
) -> Dict[str, float]:
    """Full PatchCore inference + evaluation for one category.

    Steps:
      1. Load the saved memory bank from Drive.
      2. Build a FAISS index on it.
      3. Load the backbone and register hooks.
      4. Extract features for every test image.
      5. Score all images (k-NN + re-weighting + heatmaps).
      6. Compute metrics (Image AUROC, Pixel AUROC, PRO).
      7. Save visualisations (grid, ROC, histogram).
      8. Clean up hooks.

    Args:
        category: MVTec category name (e.g. 'bottle').

    Returns:
        Dict with keys 'image_auroc', 'pixel_auroc', 'pro'.
    """
    logger.info("========== PatchCore Inference: %s ==========", category)

    # ---- 1. Load memory bank ----
    bank_path = os.path.join(cfg.MEMORY_BANK_DIR, f"{category}_memory_bank.pt")
    try:
        memory_bank = load_memory_bank(bank_path, cfg.DEVICE)
    except FileNotFoundError:
        logger.error("[%s] Memory bank not found at %s — run train.py first!", category, bank_path)
        return {"image_auroc": 0.0, "pixel_auroc": 0.0, "pro": 0.0}

    # ---- 2. Build FAISS index ----
    faiss_index = build_faiss_index(memory_bank)

    # ---- 3. Load backbone + hooks ----
    model = load_backbone(cfg.DEVICE)
    hook_dict, handles = register_hooks(model, cfg.BACKBONE_LAYERS)

    try:
        # ---- 4. Extract test features ----
        dataloader = get_dataloader(
            category=category,
            split="test",
            batch_size=cfg.BATCH_SIZE,
            num_workers=cfg.NUM_WORKERS,
            shuffle=False,
            root_dir=cfg.DATA_ROOT,
            img_size=cfg.IMG_SIZE,
        )

        all_features, all_shapes, all_labels, all_masks, all_images = (
            _extract_test_features(model, dataloader, hook_dict, cfg.DEVICE)
        )

        # ---- 5. Score all images ----
        image_scores, heatmaps, labels, gt_masks = score_dataset(
            all_patch_features=all_features,
            all_spatial_shapes=all_shapes,
            all_labels=all_labels,
            all_masks=all_masks,
            faiss_index=faiss_index,
            memory_bank=memory_bank,
            output_size=cfg.IMG_SIZE,
            nn_k=cfg.NN_K,
            reweight_k=cfg.REWEIGHT_K,
            gaussian_sigma=cfg.GAUSSIAN_SIGMA,
        )

        # ---- 6. Compute metrics ----
        metrics = evaluate_all(image_scores, labels, heatmaps, gt_masks)
        logger.info("[%s] Metrics: %s", category, metrics)

        # ---- 7. Save visualisations ----
        vis_dir = os.path.join(cfg.HEATMAP_DIR, category)
        os.makedirs(vis_dir, exist_ok=True)

        # Qualitative grid (pick defective images for most interesting view)
        defective_idx = [i for i, l in enumerate(labels) if l == 1]
        if len(defective_idx) > 0:
            grid_idx = defective_idx[:8]
        else:
            grid_idx = list(range(min(8, len(all_images))))

        grid_images = [all_images[i] for i in grid_idx]
        grid_masks = [gt_masks[i] for i in grid_idx]
        grid_heatmaps = [heatmaps[i] for i in grid_idx]

        save_qualitative_grid(
            grid_images, grid_masks, grid_heatmaps,
            os.path.join(vis_dir, f"{category}_qualitative_grid.png"),
        )

        # ROC curve
        plot_roc_curve(
            image_scores, labels,
            os.path.join(vis_dir, f"{category}_roc_curve.png"),
            title=f"ROC Curve — {category}",
        )

        # Score histogram
        save_score_histogram(
            image_scores, labels,
            os.path.join(vis_dir, f"{category}_score_histogram.png"),
            title=f"Score Distribution — {category}",
        )

        return metrics

    finally:
        # ---- 8. Clean up hooks ----
        remove_hooks(handles)


# ---------------------------------------------------------------------------
# 2. main
# ---------------------------------------------------------------------------
def main() -> None:
    """Run PatchCore inference for every category and write a final report.

    For each category:
      - Runs run_patchcore_inference
      - Collects metrics

    At the end:
      - Writes a JSON report with all results
      - Logs a summary table
    """
    logger.info("=" * 60)
    logger.info("PatchCore Test Pipeline")
    logger.info("Categories: %s", cfg.CATEGORIES)
    logger.info("Device: %s", cfg.DEVICE)
    logger.info("=" * 60)

    all_results: Dict[str, Dict[str, float]] = {}

    for category in cfg.CATEGORIES:
        metrics = run_patchcore_inference(category)
        all_results[category] = metrics

    # ---- Summary table ----
    logger.info("")
    logger.info("=" * 60)
    logger.info("FINAL RESULTS")
    logger.info("%-12s | %-12s | %-12s | %-8s", "Category", "Image AUROC", "Pixel AUROC", "PRO")
    logger.info("-" * 60)
    for cat, m in all_results.items():
        logger.info(
            "%-12s | %-12.4f | %-12.4f | %-8.4f",
            cat, m["image_auroc"], m["pixel_auroc"], m["pro"],
        )
    logger.info("=" * 60)

    # ---- Averages ----
    avg_img = np.mean([m["image_auroc"] for m in all_results.values()])
    avg_pix = np.mean([m["pixel_auroc"] for m in all_results.values()])
    avg_pro = np.mean([m["pro"] for m in all_results.values()])
    logger.info(
        "%-12s | %-12.4f | %-12.4f | %-8.4f",
        "AVERAGE", avg_img, avg_pix, avg_pro,
    )

    all_results["average"] = {
        "image_auroc": float(avg_img),
        "pixel_auroc": float(avg_pix),
        "pro": float(avg_pro),
    }

    # ---- Save JSON report ----
    report_path = os.path.join(cfg.OUTPUT_DIR, "patchcore_results.json")
    try:
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info("Results saved to %s", report_path)
    except Exception as exc:
        logger.error("Failed to save results JSON: %s", exc)

    logger.info("🎉 All done!")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
