"""
Evaluate the improved CLIP anomaly baseline on MVTec-style folders.

Run from the project root:
    python scripts/evaluate_clip.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg
from src.baseline.clip_anomaly import (
    CATEGORIES,
    CLIPAnomalyDetector,
    ClipCalibration,
    resize_for_model,
)
from src.evaluation.metrics import evaluate_all
from src.evaluation.visualize import save_qualitative_grid, save_score_histogram

logger = logging.getLogger("evaluate_clip")


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


def image_files(folder: Path) -> List[Path]:
    if not folder.is_dir():
        return []
    return sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def mask_path_for(
    data_root: Path,
    category: str,
    defect_type: str,
    image_path: Path,
) -> Optional[Path]:
    if defect_type == "good":
        return None
    candidate = (
        data_root
        / category
        / "ground_truth"
        / defect_type
        / f"{image_path.stem}_mask.png"
    )
    return candidate if candidate.exists() else None


def collect_test_items(data_root: Path, category: str) -> List[dict]:
    test_dir = data_root / category / "test"
    gt_root = data_root / category / "ground_truth"
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Missing test folder: {test_dir}")

    items: List[dict] = []
    for defect_dir in sorted(path for path in test_dir.iterdir() if path.is_dir()):
        defect_type = defect_dir.name
        label = 0 if defect_type == "good" else 1
        for image_path in image_files(defect_dir):
            items.append(
                {
                    "image_path": image_path,
                    "label": label,
                    "defect_type": defect_type,
                    "mask_path": mask_path_for(data_root, category, defect_type, image_path),
                }
            )

    if not gt_root.exists():
        logger.warning("[%s] No ground_truth folder found. Pixel metrics may be 0.", category)
    return items


def load_mask(mask_path: Optional[Path], image_size: int) -> np.ndarray:
    if mask_path and mask_path.exists():
        mask = Image.open(mask_path).convert("L").resize((image_size, image_size), Image.NEAREST)
        return (np.array(mask) > 127).astype(np.float32)
    return np.zeros((image_size, image_size), dtype=np.float32)


def good_training_paths(data_root: Path, category: str) -> List[str]:
    good_dir = data_root / category / "train" / "good"
    return [str(path) for path in image_files(good_dir)]


def evaluate_category(
    detector: CLIPAnomalyDetector,
    data_root: Path,
    category: str,
    image_size: int,
    heatmap_dir: Path,
    calibration_images: int,
) -> Dict[str, object]:
    logger.info("========== Improved CLIP: %s ==========", category)

    calibration = detector.fit_calibration(
        category=category,
        good_image_paths=good_training_paths(data_root, category),
        image_size=image_size,
        max_images=calibration_images,
    )

    items = collect_test_items(data_root, category)
    image_scores: List[float] = []
    image_labels: List[int] = []
    heatmaps: List[np.ndarray] = []
    gt_masks: List[np.ndarray] = []
    images_for_grid: List[np.ndarray] = []
    masks_for_grid: List[np.ndarray] = []
    heatmaps_for_grid: List[np.ndarray] = []

    for item in tqdm(items, desc=f"CLIP [{category}]"):
        image = resize_for_model(Image.open(item["image_path"]), image_size)
        image_arr = np.asarray(image).astype(np.float32) / 255.0
        mask = load_mask(item["mask_path"], image_size)

        score = detector.image_score(image, category, calibration=calibration)
        heatmap = detector.heatmap(image, category, calibration=calibration)

        image_scores.append(float(score))
        image_labels.append(int(item["label"]))
        heatmaps.append(heatmap)
        gt_masks.append(mask)

        if item["label"] == 1 and len(images_for_grid) < 8:
            images_for_grid.append(image_arr)
            masks_for_grid.append(mask)
            heatmaps_for_grid.append(heatmap)

    if not images_for_grid:
        for item, heatmap, mask in zip(items[:8], heatmaps[:8], gt_masks[:8]):
            image = resize_for_model(Image.open(item["image_path"]), image_size)
            images_for_grid.append(np.asarray(image).astype(np.float32) / 255.0)
            masks_for_grid.append(mask)
            heatmaps_for_grid.append(heatmap)

    metrics = evaluate_all(image_scores, image_labels, heatmaps, gt_masks)

    category_dir = heatmap_dir / category
    category_dir.mkdir(parents=True, exist_ok=True)
    grid_path = category_dir / f"{category}_clip_qualitative_grid.png"
    histogram_path = category_dir / f"{category}_clip_score_histogram.png"

    save_qualitative_grid(
        images_for_grid,
        masks_for_grid,
        heatmaps_for_grid,
        str(grid_path),
        max_rows=8,
    )
    save_score_histogram(
        image_scores,
        image_labels,
        str(histogram_path),
        title=f"Improved CLIP Scores - {category}",
    )

    result: Dict[str, object] = {
        **metrics,
        "num_good": int(sum(1 for label in image_labels if label == 0)),
        "num_defective": int(sum(1 for label in image_labels if label == 1)),
        "score_mean": float(np.mean(image_scores)) if image_scores else 0.0,
        "score_std": float(np.std(image_scores)) if image_scores else 0.0,
        "calibration": calibration.to_dict(),
        "heatmap_examples": [str(grid_path), str(histogram_path)],
    }
    logger.info("[%s] Metrics: %s", category, metrics)
    return result


def add_average(results: Dict[str, dict]) -> None:
    category_results = [results[cat] for cat in CATEGORIES if cat in results]
    if not category_results:
        return
    results["average"] = {
        "image_auroc": float(np.mean([m["image_auroc"] for m in category_results])),
        "pixel_auroc": float(np.mean([m["pixel_auroc"] for m in category_results])),
        "pro": float(np.mean([m["pro"] for m in category_results])),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate improved CLIP anomaly baseline.")
    parser.add_argument("--data-root", default=cfg.DATA_ROOT, help="MVTec dataset root.")
    parser.add_argument("--results-dir", default=cfg.RESULTS_DIR, help="Where JSON results are saved.")
    parser.add_argument("--heatmap-dir", default=cfg.CLIP_HEATMAP_DIR, help="Where CLIP heatmaps are saved.")
    parser.add_argument("--image-size", type=int, default=cfg.IMG_SIZE)
    parser.add_argument("--calibration-images", type=int, default=30)
    parser.add_argument("--model-name", default="ViT-B-32")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--categories", nargs="+", default=cfg.CATEGORIES)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()

    data_root = Path(args.data_root)
    results_dir = Path(args.results_dir)
    heatmap_dir = Path(args.heatmap_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    detector = CLIPAnomalyDetector(
        model_name=args.model_name,
        pretrained=args.pretrained,
    )

    results: Dict[str, dict] = {}
    for category in args.categories:
        results[category] = evaluate_category(
            detector=detector,
            data_root=data_root,
            category=category,
            image_size=args.image_size,
            heatmap_dir=heatmap_dir,
            calibration_images=args.calibration_images,
        )

    add_average(results)

    report_path = results_dir / "clip_results.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print("\nImproved CLIP results")
    print("=" * 72)
    print(f"{'Category':<12} | {'Image AUROC':<12} | {'Pixel AUROC':<12} | {'PRO':<8}")
    print("-" * 72)
    for category, metrics in results.items():
        if category == "average":
            continue
        print(
            f"{category:<12} | "
            f"{metrics['image_auroc']:<12.4f} | "
            f"{metrics['pixel_auroc']:<12.4f} | "
            f"{metrics['pro']:<8.4f}"
        )
    if "average" in results:
        avg = results["average"]
        print("-" * 72)
        print(
            f"{'AVERAGE':<12} | "
            f"{avg['image_auroc']:<12.4f} | "
            f"{avg['pixel_auroc']:<12.4f} | "
            f"{avg['pro']:<8.4f}"
        )
    print("=" * 72)
    print(f"Saved JSON: {report_path}")
    print(f"Saved heatmaps: {heatmap_dir}")


if __name__ == "__main__":
    main()

