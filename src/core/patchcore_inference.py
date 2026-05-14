# patchcore_inference.py
"""
Single-image PatchCore inference used by the Streamlit dashboard.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

import config as cfg
from src.core.feature_extractor import (
    align_and_concat,
    extract_layer_features,
    flatten_patches,
    locally_aware_patches,
)
from src.core.memory_bank import build_faiss_index, load_memory_bank
from src.core.scoring import score_image


class PatchCorePredictor:
    """Load one category memory bank and score uploaded images."""

    def __init__(
        self,
        category: str,
        memory_bank_dir: str = cfg.MEMORY_BANK_DIR,
        device: str = cfg.DEVICE,
        backbone_name: str = cfg.BACKBONE,
    ) -> None:
        self.category = category
        self.device = device
        self.backbone_name = backbone_name

        bank_path = Path(memory_bank_dir) / f"{category}_memory_bank.pt"
        if not bank_path.exists():
            raise FileNotFoundError(
                f"PatchCore memory bank not found for '{category}': {bank_path}"
            )

        self.memory_bank = load_memory_bank(str(bank_path), device)
        self.faiss_index = build_faiss_index(self.memory_bank)
        self.model = getattr(models, backbone_name)(weights="IMAGENET1K_V1")
        self.model = self.model.to(device).eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.hook_dict: Dict[str, torch.Tensor] = {}
        self.handles = []
        for layer_name in cfg.BACKBONE_LAYERS:
            layer = getattr(self.model, layer_name)
            self.handles.append(layer.register_forward_hook(self._make_hook(layer_name)))

        self.transform = T.Compose([
            T.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
            T.CenterCrop(cfg.IMG_SIZE),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _make_hook(self, name: str):
        def hook_fn(module, inputs, output):
            self.hook_dict[name] = output.detach()
        return hook_fn

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def _extract_features(self, image: Image.Image) -> Tuple[torch.Tensor, Tuple[int, int]]:
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat_l2, feat_l3 = extract_layer_features(self.model, tensor, self.hook_dict)
            feat_l2 = locally_aware_patches(feat_l2)
            feat_l3 = locally_aware_patches(feat_l3)
            combined = align_and_concat(feat_l2, feat_l3)
            flat, shape = flatten_patches(combined)
        return flat.cpu(), shape

    def predict(self, image: Image.Image) -> Tuple[float, np.ndarray, np.ndarray]:
        """Return image score, heatmap, and resized RGB image array."""
        image_rgb = image.convert("RGB")
        display_image = image_rgb.resize((cfg.IMG_SIZE, cfg.IMG_SIZE), Image.BICUBIC)
        features, spatial_shape = self._extract_features(image_rgb)
        score, heatmap = score_image(
            features,
            spatial_shape,
            self.faiss_index,
            self.memory_bank,
            output_size=cfg.IMG_SIZE,
            nn_k=cfg.NN_K,
            reweight_k=cfg.REWEIGHT_K,
            gaussian_sigma=cfg.GAUSSIAN_SIGMA,
        )
        image_array = np.asarray(display_image).astype(np.float32) / 255.0
        return score, heatmap, image_array

