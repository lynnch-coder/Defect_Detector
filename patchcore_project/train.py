# train.py
"""
Training Pipeline for PatchCore
=================================
This is the "conductor" that orchestrates the entire training process.
It doesn't do any heavy lifting itself — it calls the right functions
from feature_extractor, coreset, and memory_bank in the right order.

PatchCore has NO gradient-based training. "Training" means:
1. Extract patch features from all good images (feature_extractor)
2. Compress them with coreset sampling (coreset)
3. Save the result to disk (memory_bank)

How to run: just call main() and it processes every category automatically.
"""

import logging
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict

from core.feature_extractor import extract_dataset_features
from core.coreset import subsample_memory_bank
from core.memory_bank import save_memory_bank

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config (inline for notebook self-containment)
# ---------------------------------------------------------------------------
CATEGORIES    = ["bottle", "carpet", "screw"]
DATA_ROOT     = "/content/mvtec"
OUTPUT_DIR    = "/content/drive/MyDrive/defects-detection-CV/outputs"
IMAGE_SIZE    = 224
BATCH_SIZE    = 32
NUM_WORKERS   = 2
BACKBONE      = "wide_resnet50_2"
CORESET_RATIO = 0.01
CKPT_EVERY    = 10
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# 1. build_memory_bank
# ---------------------------------------------------------------------------
def build_memory_bank(
    model: nn.Module,
    dataloader: DataLoader,
    hook_dict: Dict[str, torch.Tensor],
    coreset_ratio: float,
    device: str,
    category: str = "",
) -> torch.Tensor:
    """Orchestrate feature extraction + coreset subsampling.

    This is the core training logic:
      Step 1 — Run every training image through the backbone and collect
               all patch features (via feature_extractor).
      Step 2 — Compress those features down to a small representative
               subset using greedy farthest-point sampling (via coreset).

    Args:
        model: Frozen backbone on device.
        dataloader: Training DataLoader (good images only).
        hook_dict: Hook dictionary attached to the backbone.
        coreset_ratio: Fraction of patches to keep (e.g. 0.01 = 1%).
        device: 'cuda' or 'cpu'.
        category: Category name (used for checkpoint file naming).

    Returns:
        Memory bank tensor of shape (n_select, D) on CPU.
    """
    logger.info("Step 1/2: Extracting features...")
    feat_ckpt = os.path.join(OUTPUT_DIR, "raw_features", f"{category}_feat_ckpt.pt")
    os.makedirs(os.path.dirname(feat_ckpt), exist_ok=True)

    all_features = extract_dataset_features(
        model, dataloader, hook_dict, device,
        checkpoint_path=feat_ckpt,
        checkpoint_every=CKPT_EVERY,
    )
    logger.info("Raw features: %s", all_features.shape)

    logger.info("Step 2/2: Coreset subsampling...")
    core_ckpt = os.path.join(OUTPUT_DIR, "memory_banks", f"{category}_coreset_ckpt.pt")
    os.makedirs(os.path.dirname(core_ckpt), exist_ok=True)

    memory_bank = subsample_memory_bank(
        all_features, coreset_ratio, device,
        checkpoint_path=core_ckpt,
        checkpoint_every=500,
    )
    logger.info("Memory bank: %s", memory_bank.shape)
    return memory_bank


# ---------------------------------------------------------------------------
# 2. build_and_save_bank
# ---------------------------------------------------------------------------
def build_and_save_bank(category: str) -> None:
    """Full training pipeline for a single category.

    1. Load the backbone and register hooks.
    2. Create a DataLoader for the category's training images.
    3. Call build_memory_bank to extract + subsample.
    4. Save the result to Google Drive.

    Args:
        category: MVTec category name (e.g. 'bottle').
    """
    import torchvision.models as models
    import torchvision.transforms as T
    from torch.utils.data import Dataset
    from PIL import Image

    logger.info("===== Category: %s =====", category)

    save_path = os.path.join(OUTPUT_DIR, "memory_banks", f"{category}_memory_bank.pt")
    if os.path.exists(save_path):
        logger.info("[%s] Memory bank already exists at %s — skipping.", category, save_path)
        return

    # ---- Backbone ----
    model = getattr(models, BACKBONE)(weights="IMAGENET1K_V1")
    model = model.to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad = False

    hook_dict: Dict[str, torch.Tensor] = {}

    def _hook(name: str):
        def fn(module, inp, out):
            hook_dict[name] = out.detach()
        return fn

    model.layer2.register_forward_hook(_hook("layer2"))
    model.layer3.register_forward_hook(_hook("layer3"))

    # ---- Dataset ----
    class _TrainDataset(Dataset):
        def __init__(self) -> None:
            img_dir = os.path.join(DATA_ROOT, category, "train", "good")
            self.paths = sorted([
                os.path.join(img_dir, f) for f in os.listdir(img_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ])
            self.transform = T.Compose([
                T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            logger.info("[%s] Found %d training images.", category, len(self.paths))

        def __len__(self) -> int:
            return len(self.paths)

        def __getitem__(self, idx: int) -> torch.Tensor:
            return self.transform(Image.open(self.paths[idx]).convert("RGB"))

    dataloader = DataLoader(
        _TrainDataset(), batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    # ---- Build & Save ----
    memory_bank = build_memory_bank(
        model, dataloader, hook_dict, CORESET_RATIO, DEVICE, category=category,
    )
    save_memory_bank(memory_bank, save_path)
    logger.info("[%s] Done!", category)


# ---------------------------------------------------------------------------
# 3. main
# ---------------------------------------------------------------------------
def main() -> None:
    """Train PatchCore for every configured category.

    Iterates over CATEGORIES and calls build_and_save_bank for each.
    Categories that already have a saved memory bank are skipped.
    """
    logger.info("Starting PatchCore training pipeline.")
    logger.info("Categories: %s", CATEGORIES)
    logger.info("Device: %s | Coreset ratio: %.2f%%", DEVICE, CORESET_RATIO * 100)

    for category in CATEGORIES:
        build_and_save_bank(category)

    logger.info("All categories complete!")


if __name__ == "__main__":
    main()
