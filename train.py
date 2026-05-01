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

import config as cfg
from src.core.feature_extractor import extract_dataset_features
from src.core.coreset import subsample_memory_bank
from src.core.memory_bank import save_memory_bank

logger = logging.getLogger(__name__)


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
    """Orchestrate feature extraction + coreset subsampling."""
    logger.info("Step 1/2: Extracting features...")
    feat_ckpt = os.path.join(cfg.OUTPUT_DIR, "raw_features", f"{category}_feat_ckpt.pt")
    os.makedirs(os.path.dirname(feat_ckpt), exist_ok=True)

    all_features = extract_dataset_features(
        model, dataloader, hook_dict, device,
        checkpoint_path=feat_ckpt,
        checkpoint_every=cfg.FEAT_CKPT_EVERY,
    )
    logger.info("Raw features: %s", all_features.shape)

    logger.info("Step 2/2: Coreset subsampling...")
    core_ckpt = os.path.join(cfg.OUTPUT_DIR, "memory_banks", f"{category}_coreset_ckpt.pt")
    os.makedirs(os.path.dirname(core_ckpt), exist_ok=True)

    memory_bank = subsample_memory_bank(
        all_features, coreset_ratio, device,
        checkpoint_path=core_ckpt,
        checkpoint_every=cfg.CORESET_CKPT_EVERY,
    )
    logger.info("Memory bank: %s", memory_bank.shape)
    return memory_bank


# ---------------------------------------------------------------------------
# 2. build_and_save_bank
# ---------------------------------------------------------------------------
def build_and_save_bank(category: str) -> None:
    """Full training pipeline for a single category."""
    import torchvision.models as models
    import torchvision.transforms as T
    from torch.utils.data import Dataset
    from PIL import Image

    logger.info("===== Category: %s =====", category)

    save_path = os.path.join(cfg.OUTPUT_DIR, "memory_banks", f"{category}_memory_bank.pt")
    if os.path.exists(save_path):
        logger.info("[%s] Memory bank already exists at %s — skipping.", category, save_path)
        return

    # ---- Backbone ----
    model = getattr(models, cfg.BACKBONE)(weights="IMAGENET1K_V1")
    model = model.to(cfg.DEVICE).eval()
    for p in model.parameters():
        p.requires_grad = False

    hook_dict: Dict[str, torch.Tensor] = {}

    def _hook(name: str):
        def fn(module, inp, out):
            hook_dict[name] = out.detach()
        return fn

    for layer_name in cfg.BACKBONE_LAYERS:
        getattr(model, layer_name).register_forward_hook(_hook(layer_name))

    # ---- Dataset ----
    class _TrainDataset(Dataset):
        def __init__(self) -> None:
            img_dir = os.path.join(cfg.DATA_ROOT, category, "train", "good")
            self.paths = sorted([
                os.path.join(img_dir, f) for f in os.listdir(img_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ])
            self.transform = T.Compose([
                T.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            logger.info("[%s] Found %d training images.", category, len(self.paths))

        def __len__(self) -> int:
            return len(self.paths)

        def __getitem__(self, idx: int) -> torch.Tensor:
            return self.transform(Image.open(self.paths[idx]).convert("RGB"))

    dataloader = DataLoader(
        _TrainDataset(), batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, pin_memory=True,
    )

    # ---- Build & Save ----
    memory_bank = build_memory_bank(
        model, dataloader, hook_dict, cfg.CORESET_RATIO, cfg.DEVICE, category=category,
    )
    save_memory_bank(memory_bank, save_path)
    logger.info("[%s] Done!", category)


# ---------------------------------------------------------------------------
# 3. main
# ---------------------------------------------------------------------------
def main() -> None:
    """Train PatchCore for every configured category."""
    logger.info("Starting PatchCore training pipeline.")
    logger.info("Categories: %s", cfg.CATEGORIES)
    logger.info("Device: %s | Coreset ratio: %.2f%%", cfg.DEVICE, cfg.CORESET_RATIO * 100)

    for category in cfg.CATEGORIES:
        build_and_save_bank(category)

    logger.info("All categories complete!")


if __name__ == "__main__":
    main()
