# feature_extractor.py
"""
Feature Extractor for PatchCore
================================
This file is the "eyes" of our defect detector. It takes product images,
passes them through a pretrained CNN (WideResNet50), and extracts the
internal feature maps that describe what the network "sees" at different
levels of detail.

Why it exists: PatchCore doesn't train a model — it just extracts features
from a frozen backbone. This file handles all of that extraction logic.

How it fits: dataset.py loads images → THIS FILE extracts features →
coreset.py selects the most important ones → memory_bank.py stores them.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Tuple
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. extract_layer_features
# ---------------------------------------------------------------------------
def extract_layer_features(
    model: nn.Module,
    image_batch: torch.Tensor,
    hook_dict: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run a forward pass and grab the layer2 + layer3 feature maps.

    The backbone has "hooks" registered on layer2 and layer3. When we
    pass an image through the network, these hooks automatically capture
    the intermediate outputs. We just need to trigger the forward pass
    and then read the captured tensors from hook_dict.

    Args:
        model: The frozen WideResNet50 backbone (already on device).
        image_batch: A batch of images, shape (B, 3, 224, 224).
        hook_dict: Dictionary that hooks populate with feature maps.

    Returns:
        Tuple of (layer2_features, layer3_features).
        layer2 shape: (B, 512, 28, 28)
        layer3 shape: (B, 1024, 14, 14)
    """
    with torch.no_grad():
        _ = model(image_batch)

    feat_layer2 = hook_dict["layer2"]
    feat_layer3 = hook_dict["layer3"]
    logger.debug(
        "Extracted features — layer2: %s, layer3: %s",
        feat_layer2.shape,
        feat_layer3.shape,
    )
    return feat_layer2, feat_layer3


# ---------------------------------------------------------------------------
# 2. locally_aware_patches
# ---------------------------------------------------------------------------
def locally_aware_patches(
    feature_map: torch.Tensor,
    patch_size: int = 3,
) -> torch.Tensor:
    """Smooth each spatial position by averaging its 3×3 neighbourhood.

    Why? A single pixel in the feature map only knows about a small part
    of the image. By averaging over its neighbours we give it "local
    context" — it now also knows what surrounds it. This makes defect
    detection more robust because defects rarely sit on a single pixel.

    Implementation: a simple average-pool with stride=1 and same-padding
    so the output shape is identical to the input.

    Args:
        feature_map: Tensor of shape (B, C, H, W).
        patch_size: Size of the neighbourhood window (default 3).

    Returns:
        Locally-aware feature map, same shape (B, C, H, W).
    """
    padding = patch_size // 2
    result = F.avg_pool2d(
        feature_map,
        kernel_size=patch_size,
        stride=1,
        padding=padding,
    )
    logger.debug("Locally-aware patches applied with kernel %d", patch_size)
    return result


# ---------------------------------------------------------------------------
# 3. align_and_concat
# ---------------------------------------------------------------------------
def align_and_concat(
    feat_layer2: torch.Tensor,
    feat_layer3: torch.Tensor,
) -> torch.Tensor:
    """Resize layer3 to match layer2 and stack them together.

    layer2 is (B, 512, 28, 28) — captures fine details like edges.
    layer3 is (B, 1024, 14, 14) — captures higher-level shapes.

    We upsample layer3 from 14×14 → 28×28 using bilinear interpolation,
    then concatenate along the channel axis so every spatial location
    gets BOTH fine and coarse information → (B, 1536, 28, 28).

    Args:
        feat_layer2: Layer-2 features, shape (B, 512, 28, 28).
        feat_layer3: Layer-3 features, shape (B, 1024, 14, 14).

    Returns:
        Combined feature map, shape (B, 1536, 28, 28).
    """
    target_h, target_w = feat_layer2.shape[2], feat_layer2.shape[3]
    feat_layer3_up = F.interpolate(
        feat_layer3,
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    )
    combined = torch.cat([feat_layer2, feat_layer3_up], dim=1)
    logger.debug("Aligned & concatenated → %s", combined.shape)
    return combined


# ---------------------------------------------------------------------------
# 4. flatten_patches
# ---------------------------------------------------------------------------
def flatten_patches(
    feature_map: torch.Tensor,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Reshape (B, C, H, W) into a 2-D table of patch vectors.

    Each spatial position (h, w) in the feature map is one "patch".
    We stack ALL patches from ALL images into a single big matrix so
    we can later compare them easily.

    Example: batch of 32 images with 28×28 spatial grid
    → 32 × 28 × 28 = 25 088 patch vectors, each of length C=1536.

    We also return (H, W) so we can later reshape back to a 2-D map
    for generating heatmaps.

    Args:
        feature_map: Tensor of shape (B, C, H, W).

    Returns:
        flat: Tensor of shape (B*H*W, C).
        spatial_shape: Tuple (H, W) for un-flattening later.
    """
    B, C, H, W = feature_map.shape
    # permute → (B, H, W, C) then flatten spatial dims
    flat = feature_map.permute(0, 2, 3, 1).reshape(-1, C)
    logger.debug("Flattened %s → %s, spatial=(%d, %d)", feature_map.shape, flat.shape, H, W)
    return flat, (H, W)


# ---------------------------------------------------------------------------
# 5. extract_dataset_features
# ---------------------------------------------------------------------------
def extract_dataset_features(
    model: nn.Module,
    dataloader: DataLoader,
    hook_dict: Dict[str, torch.Tensor],
    device: str,
    checkpoint_path: str = "feature_checkpoint.pt",
    checkpoint_every: int = 10,
) -> torch.Tensor:
    """Extract features for EVERY training image and stack them.

    This is the main workhorse. It loops through all batches, extracts
    layer2+layer3, makes them locally-aware, aligns, flattens, and
    collects everything into one giant (N_total_patches, 1536) tensor.

    Includes a checkpoint system: every `checkpoint_every` batches the
    progress is saved to disk so we can resume if Colab disconnects.

    Args:
        model: Frozen backbone on device.
        dataloader: Training dataloader (good images only).
        hook_dict: Hook dictionary attached to the model.
        device: 'cuda' or 'cpu'.
        checkpoint_path: Where to save intermediate progress.
        checkpoint_every: Save a checkpoint every N batches.

    Returns:
        all_features: Tensor of shape (N_total_patches, C) on CPU.
    """
    all_features: list[torch.Tensor] = []
    start_batch: int = 0

    # --- Resume from checkpoint if available ---
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        all_features = ckpt["features"]
        start_batch = ckpt["next_batch"]
        logger.info("Resumed from checkpoint at batch %d", start_batch)
    except FileNotFoundError:
        logger.info("No checkpoint found — starting fresh.")
    except Exception as exc:
        logger.warning("Could not load checkpoint (%s) — starting fresh.", exc)

    batches = list(dataloader)
    remaining = batches[start_batch:]

    for batch_idx, images in enumerate(
        tqdm(remaining, desc="Extracting features", initial=start_batch, total=len(batches)),
        start=start_batch,
    ):
        images = images.to(device)

        # Step 1 — forward pass to get raw layer features
        feat_l2, feat_l3 = extract_layer_features(model, images, hook_dict)

        # Step 2 — locally-aware smoothing
        feat_l2 = locally_aware_patches(feat_l2)
        feat_l3 = locally_aware_patches(feat_l3)

        # Step 3 — align spatial sizes and concatenate channels
        combined = align_and_concat(feat_l2, feat_l3)

        # Step 4 — flatten to (num_patches_in_batch, C) and keep on CPU
        flat, _ = flatten_patches(combined)
        all_features.append(flat.cpu())

        # Step 5 — periodic checkpoint
        if (batch_idx + 1) % checkpoint_every == 0:
            try:
                torch.save(
                    {"features": all_features, "next_batch": batch_idx + 1},
                    checkpoint_path,
                )
                logger.info("Checkpoint saved at batch %d", batch_idx + 1)
            except Exception as exc:
                logger.warning("Checkpoint save failed: %s", exc)

    result = torch.cat(all_features, dim=0)
    logger.info("Total extracted features: %s", result.shape)
    return result
