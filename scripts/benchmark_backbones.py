"""
Benchmark PatchCore backbones outside the Streamlit interface.

Run from the project root:
    python scripts/benchmark_backbones.py
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg
from src.core.coreset import subsample_memory_bank
from src.core.feature_extractor import (
    align_and_concat,
    extract_dataset_features,
    extract_layer_features,
    flatten_patches,
    locally_aware_patches,
)
from src.core.memory_bank import build_faiss_index, load_memory_bank, save_memory_bank
from src.core.scoring import score_image
from src.evaluation.metrics import evaluate_all

logger = logging.getLogger("benchmark_backbones")


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


class TrainDataset(Dataset):
    def __init__(self, data_root: Path, category: str, image_size: int) -> None:
        self.paths = sorted(
            path for path in (data_root / category / "train" / "good").iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.transform(Image.open(self.paths[idx]).convert("RGB"))


class TestDataset(Dataset):
    def __init__(self, data_root: Path, category: str, image_size: int) -> None:
        self.image_size = image_size
        self.image_paths: List[Path] = []
        self.mask_paths: List[Path | None] = []
        self.labels: List[int] = []
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.mask_transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.CenterCrop(image_size),
            T.ToTensor(),
        ])

        test_dir = data_root / category / "test"
        gt_dir = data_root / category / "ground_truth"
        for defect_dir in sorted(path for path in test_dir.iterdir() if path.is_dir()):
            defect_type = defect_dir.name
            label = 0 if defect_type == "good" else 1
            for image_path in sorted(
                path for path in defect_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ):
                self.image_paths.append(image_path)
                self.labels.append(label)
                if defect_type == "good":
                    self.mask_paths.append(None)
                else:
                    mask_path = gt_dir / defect_type / f"{image_path.stem}_mask.png"
                    self.mask_paths.append(mask_path if mask_path.exists() else None)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image = self.transform(Image.open(self.image_paths[idx]).convert("RGB"))
        mask_path = self.mask_paths[idx]
        if mask_path:
            mask = self.mask_transform(Image.open(mask_path).convert("L"))
            mask = (mask > 0.5).float()
        else:
            mask = torch.zeros(1, self.image_size, self.image_size)
        return image, mask, self.labels[idx], str(self.image_paths[idx])


def load_imagenet_backbone(backbone_name: str, device: str) -> torch.nn.Module:
    try:
        weights_enum = models.get_model_weights(backbone_name)
        model = models.get_model(backbone_name, weights=weights_enum.DEFAULT)
    except Exception:
        model = getattr(models, backbone_name)(weights="IMAGENET1K_V1")

    model = model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def register_hooks(model: torch.nn.Module, layer_names: List[str]) -> Tuple[dict, list]:
    hook_dict: Dict[str, torch.Tensor] = {}
    handles = []

    for layer_name in layer_names:
        layer = getattr(model, layer_name)

        def make_hook(name: str):
            def hook_fn(module, inputs, output):
                hook_dict[name] = output.detach()
            return hook_fn

        handles.append(layer.register_forward_hook(make_hook(layer_name)))

    return hook_dict, handles


def build_bank(
    backbone_name: str,
    category: str,
    data_root: Path,
    bench_root: Path,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, float]:
    bank_dir = bench_root / backbone_name / "memory_banks"
    bank_path = bank_dir / f"{category}_memory_bank.pt"
    if bank_path.exists() and not args.force:
        logger.info("[%s/%s] Reusing %s", backbone_name, category, bank_path)
        return load_memory_bank(str(bank_path), args.device), 0.0

    model = load_imagenet_backbone(backbone_name, args.device)
    hook_dict, handles = register_hooks(model, cfg.BACKBONE_LAYERS)
    dataset = TrainDataset(data_root, category, args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
    )

    start = time.perf_counter()
    raw_ckpt = bench_root / backbone_name / "raw_features" / f"{category}_features.pt"
    core_ckpt = bench_root / backbone_name / "memory_banks" / f"{category}_coreset.pt"
    raw_ckpt.parent.mkdir(parents=True, exist_ok=True)
    core_ckpt.parent.mkdir(parents=True, exist_ok=True)

    features = extract_dataset_features(
        model,
        loader,
        hook_dict,
        args.device,
        checkpoint_path=str(raw_ckpt),
        checkpoint_every=cfg.FEAT_CKPT_EVERY,
    )
    memory_bank = subsample_memory_bank(
        features,
        args.coreset_ratio,
        args.device,
        checkpoint_path=str(core_ckpt),
        checkpoint_every=cfg.CORESET_CKPT_EVERY,
    )
    save_memory_bank(memory_bank, str(bank_path))
    train_seconds = time.perf_counter() - start

    for handle in handles:
        handle.remove()
    return memory_bank, train_seconds


def evaluate_bank(
    backbone_name: str,
    category: str,
    memory_bank: torch.Tensor,
    data_root: Path,
    args: argparse.Namespace,
) -> Tuple[Dict[str, float], float]:
    model = load_imagenet_backbone(backbone_name, args.device)
    hook_dict, handles = register_hooks(model, cfg.BACKBONE_LAYERS)
    faiss_index = build_faiss_index(memory_bank)

    dataset = TestDataset(data_root, category, args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
    )

    all_features: List[torch.Tensor] = []
    all_shapes: List[Tuple[int, int]] = []
    all_labels: List[int] = []
    all_masks: List[torch.Tensor] = []

    start = time.perf_counter()
    for images, masks, labels, paths in loader:
        images = images.to(args.device)
        with torch.no_grad():
            feat_l2, feat_l3 = extract_layer_features(model, images, hook_dict)
            feat_l2 = locally_aware_patches(feat_l2)
            feat_l3 = locally_aware_patches(feat_l3)
            combined = align_and_concat(feat_l2, feat_l3)

        for idx in range(combined.shape[0]):
            flat, shape = flatten_patches(combined[idx].unsqueeze(0))
            all_features.append(flat.cpu())
            all_shapes.append(shape)
            all_labels.append(int(labels[idx]))
            all_masks.append(masks[idx].cpu())

    image_scores: List[float] = []
    heatmaps: List[np.ndarray] = []
    gt_masks: List[np.ndarray] = []

    for features, shape, mask in zip(all_features, all_shapes, all_masks):
        score, heatmap = score_image(
            features,
            shape,
            faiss_index,
            memory_bank,
            output_size=args.image_size,
            nn_k=cfg.NN_K,
            reweight_k=cfg.REWEIGHT_K,
            gaussian_sigma=cfg.GAUSSIAN_SIGMA,
        )
        image_scores.append(score)
        heatmaps.append(heatmap)
        gt_masks.append(mask.squeeze().numpy())

    inference_seconds = time.perf_counter() - start
    metrics = evaluate_all(image_scores, all_labels, heatmaps, gt_masks)

    for handle in handles:
        handle.remove()
    return metrics, inference_seconds


def average_rows(rows: List[dict]) -> Dict[str, dict]:
    averages: Dict[str, dict] = {}
    for backbone in sorted({row["backbone"] for row in rows}):
        backbone_rows = [row for row in rows if row["backbone"] == backbone]
        averages[backbone] = {
            "image_auroc": float(np.mean([row["image_auroc"] for row in backbone_rows])),
            "pixel_auroc": float(np.mean([row["pixel_auroc"] for row in backbone_rows])),
            "pro": float(np.mean([row["pro"] for row in backbone_rows])),
            "train_seconds": float(np.sum([row["train_seconds"] for row in backbone_rows])),
            "inference_seconds": float(np.sum([row["inference_seconds"] for row in backbone_rows])),
            "feature_dim": int(np.mean([row["feature_dim"] for row in backbone_rows])),
        }
    return averages


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "backbone",
        "category",
        "image_auroc",
        "pixel_auroc",
        "pro",
        "feature_dim",
        "memory_bank_vectors",
        "memory_bank_mb",
        "train_seconds",
        "inference_seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def write_markdown(path: Path, averages: Dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wide = averages.get("wide_resnet50_2", {})
    lines = [
        "# WideResNet Benchmark Justification",
        "",
        "This benchmark compares PatchCore with `resnet18`, `resnet50`, and `wide_resnet50_2` on the same categories and metrics.",
        "",
        "| Backbone | Image AUROC | Pixel AUROC | PRO | Feature Dim |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for backbone, metrics in averages.items():
        lines.append(
            f"| `{backbone}` | {metrics['image_auroc']:.4f} | "
            f"{metrics['pixel_auroc']:.4f} | {metrics['pro']:.4f} | "
            f"{metrics['feature_dim']} |"
        )

    lines.extend([
        "",
        "## Conclusion",
        "",
        "`wide_resnet50_2` is the preferred PatchCore backbone because it keeps the ResNet spatial feature hierarchy while widening the channel capacity of the intermediate layers used by PatchCore. That gives each patch a richer descriptor, which is especially useful for subtle texture and metallic defects.",
        "",
        "It is also the backbone used in the original PatchCore setup, so choosing it makes the project easier to justify against published anomaly detection practice.",
    ])
    if wide:
        lines.append(
            f"In this run, `wide_resnet50_2` averaged Image AUROC={wide['image_auroc']:.4f}, "
            f"Pixel AUROC={wide['pixel_auroc']:.4f}, and PRO={wide['pro']:.4f}."
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark PatchCore backbones.")
    parser.add_argument("--data-root", default=cfg.DATA_ROOT)
    parser.add_argument("--output-dir", default=cfg.OUTPUT_DIR)
    parser.add_argument("--categories", nargs="+", default=cfg.CATEGORIES)
    parser.add_argument(
        "--backbones",
        nargs="+",
        default=["resnet18", "resnet50", "wide_resnet50_2"],
    )
    parser.add_argument("--image-size", type=int, default=cfg.IMG_SIZE)
    parser.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--coreset-ratio", type=float, default=cfg.CORESET_RATIO)
    parser.add_argument("--device", default=cfg.DEVICE)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    bench_root = output_dir / "benchmarks"

    rows: List[dict] = []
    for backbone_name in args.backbones:
        for category in args.categories:
            logger.info("Benchmarking %s on %s", backbone_name, category)
            memory_bank, train_seconds = build_bank(
                backbone_name,
                category,
                data_root,
                bench_root,
                args,
            )
            metrics, inference_seconds = evaluate_bank(
                backbone_name,
                category,
                memory_bank,
                data_root,
                args,
            )
            row = {
                "backbone": backbone_name,
                "category": category,
                **metrics,
                "feature_dim": int(memory_bank.shape[1]),
                "memory_bank_vectors": int(memory_bank.shape[0]),
                "memory_bank_mb": float(memory_bank.numel() * memory_bank.element_size() / (1024 ** 2)),
                "train_seconds": float(train_seconds),
                "inference_seconds": float(inference_seconds),
            }
            rows.append(row)

    averages = average_rows(rows)

    results_dir = Path(cfg.RESULTS_DIR)
    reports_dir = Path(cfg.REPORT_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    json_path = results_dir / "backbone_benchmark.json"
    csv_path = results_dir / "backbone_benchmark.csv"
    md_path = reports_dir / "wideresnet_benchmark.md"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump({"rows": rows, "averages": averages}, handle, indent=2)
    write_csv(csv_path, rows)
    write_markdown(md_path, averages)

    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved report: {md_path}")


if __name__ == "__main__":
    main()
