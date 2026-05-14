# scoring.py
"""
Anomaly Scoring for PatchCore
===============================
This file takes test image features and decides HOW anomalous each patch
(and the whole image) is by comparing them to the memory bank.

The pipeline for one image:
  1. k-NN lookup  → find the closest normal patch for each test patch
  2. Re-weighting → adjust scores using the local density around the
                     nearest memory bank vector (PatchCore Eq. 7)
  3. Heatmap      → reshape scores to 2D, upsample, Gaussian-smooth
  4. Image score  → take the max of the heatmap

Why re-weighting? If a memory bank vector sits in a dense cluster of
normal patches, a match to it is very trustworthy. If it sits alone,
the match is less certain. Re-weighting captures this intuition.

How it fits:
  test.py → feature_extractor → THIS FILE → metrics + visualize
"""

import logging
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

import faiss

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. compute_patch_distances
# ---------------------------------------------------------------------------
def compute_patch_distances(
    test_features: torch.Tensor,
    faiss_index: faiss.Index,
    k: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """k-NN lookup of every test patch against the memory bank.

    Args:
        test_features: (Q, D) tensor of test patch vectors.
        faiss_index: Populated FAISS index built from the memory bank.
        k: Number of nearest neighbours.

    Returns:
        distances: (Q, k) numpy array of L2 distances.
        indices:   (Q, k) numpy array of neighbour indices into the
                   memory bank.
    """
    queries = test_features.detach().cpu().numpy().astype(np.float32)
    distances, indices = faiss_index.search(queries, k)
    logger.debug(
        "Patch distances: %d queries, k=%d, max=%.4f, min=%.4f",
        queries.shape[0], k, distances.max(), distances.min(),
    )
    return distances, indices


# ---------------------------------------------------------------------------
# 2. reweight_score
# ---------------------------------------------------------------------------
def reweight_score(
    distances: np.ndarray,
    neighbor_indices: np.ndarray,
    memory_bank: torch.Tensor,
    reweight_k: int = 3,
) -> np.ndarray:
    """Apply PatchCore Equation 7 re-weighting.

    For each test patch, we found its nearest memory bank vector m*.
    Now we look at the reweight_k nearest neighbours OF m* inside the
    memory bank itself. If those neighbours are far apart (sparse area),
    the anomaly score is amplified. If they are close (dense area), it
    is dampened.

    Score = distance(test_patch, m*) * weight
    weight = max(distances_among_m*_neighbours)

    Args:
        distances: (Q, 1) distances from k-NN lookup (k=1).
        neighbor_indices: (Q, 1) indices of nearest memory bank vectors.
        memory_bank: (M, D) full memory bank tensor.
        reweight_k: Number of neighbours to examine around m*.

    Returns:
        Reweighted scores, shape (Q,).
    """
    mb_np = memory_bank.detach().cpu().numpy().astype(np.float32)
    dim = mb_np.shape[1]

    # Build a small FAISS index on the memory bank for self-lookup
    mb_index = faiss.IndexFlatL2(dim)
    mb_index.add(mb_np)

    # For each test patch's nearest memory bank vector, find ITS neighbours
    nn_indices = neighbor_indices[:, 0]  # (Q,)
    nn_vectors = mb_np[nn_indices]       # (Q, D)

    # Search the memory bank for reweight_k neighbours of each m*
    mb_dists, _ = mb_index.search(nn_vectors, reweight_k)  # (Q, reweight_k)

    # Weight = max distance among m*'s neighbourhood
    # Add small epsilon to avoid division by zero
    weights = mb_dists[:, -1]  # farthest neighbour distance
    weights = np.maximum(weights, 1e-8)

    # Reweighted score = raw distance / local density proxy
    raw_distances = distances[:, 0]  # (Q,)
    reweighted = raw_distances / weights

    logger.debug(
        "Reweighting: raw_max=%.4f, weight_max=%.4f, reweighted_max=%.4f",
        raw_distances.max(), weights.max(), reweighted.max(),
    )
    return reweighted


# ---------------------------------------------------------------------------
# 3. patch_scores_to_heatmap
# ---------------------------------------------------------------------------
def patch_scores_to_heatmap(
    patch_scores: np.ndarray,
    spatial_shape: Tuple[int, int],
    output_size: int = 224,
    gaussian_sigma: float = 4.0,
) -> np.ndarray:
    """Convert flat patch scores into a smooth 2D heatmap.

    Steps:
      1. Reshape (H*W,) → (H, W)
      2. Bilinear upsample to output_size × output_size
      3. Apply Gaussian smoothing to remove grid artefacts

    Args:
        patch_scores: (H*W,) array of per-patch anomaly scores.
        spatial_shape: (H, W) of the feature map grid.
        output_size: Target heatmap resolution (matches input image).
        gaussian_sigma: Sigma for Gaussian blur.

    Returns:
        Smoothed heatmap, shape (output_size, output_size).
    """
    H, W = spatial_shape

    # Reshape to 2D
    heatmap = patch_scores.reshape(H, W)

    # Upsample with bilinear interpolation
    heatmap_tensor = torch.tensor(heatmap, dtype=torch.float32)
    heatmap_tensor = heatmap_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    heatmap_tensor = F.interpolate(
        heatmap_tensor,
        size=(output_size, output_size),
        mode="bilinear",
        align_corners=False,
    )
    heatmap = heatmap_tensor.squeeze().numpy()

    # Gaussian smoothing
    heatmap = gaussian_filter(heatmap, sigma=gaussian_sigma)

    logger.debug(
        "Heatmap: (%d,%d) → (%d,%d), sigma=%.1f, max=%.4f",
        H, W, output_size, output_size, gaussian_sigma, heatmap.max(),
    )
    return heatmap


# ---------------------------------------------------------------------------
# 4. image_anomaly_score
# ---------------------------------------------------------------------------
def image_anomaly_score(
    heatmap: np.ndarray,
) -> float:
    """Compute a single image-level anomaly score from the heatmap.

    We simply take the maximum value — the most anomalous patch
    determines whether the whole image is flagged.

    Args:
        heatmap: (H, W) anomaly heatmap.

    Returns:
        Scalar anomaly score (higher = more anomalous).
    """
    score = float(heatmap.max())
    logger.debug("Image anomaly score: %.6f", score)
    return score


# ---------------------------------------------------------------------------
# 5. score_image
# ---------------------------------------------------------------------------
def score_image(
    patch_features: torch.Tensor,
    spatial_shape: Tuple[int, int],
    faiss_index: faiss.Index,
    memory_bank: torch.Tensor,
    output_size: int = 224,
    nn_k: int = 1,
    reweight_k: int = 3,
    gaussian_sigma: float = 4.0,
) -> Tuple[float, np.ndarray]:
    """Full scoring pipeline for a single image.

    Takes already-extracted patch features and produces an image-level
    anomaly score plus a pixel-level heatmap.

    Args:
        patch_features: (H*W, D) features for one image's patches.
        spatial_shape: (H, W) of the feature grid.
        faiss_index: Populated FAISS index.
        memory_bank: (M, D) memory bank tensor.
        output_size: Heatmap resolution.
        nn_k: Neighbours for distance lookup.
        reweight_k: Neighbours for re-weighting.
        gaussian_sigma: Gaussian blur sigma.

    Returns:
        score: Scalar image-level anomaly score.
        heatmap: (output_size, output_size) numpy array.
    """
    # Step 1 — k-NN distances
    distances, indices = compute_patch_distances(patch_features, faiss_index, k=nn_k)

    # Step 2 — re-weight
    patch_scores = reweight_score(distances, indices, memory_bank, reweight_k)

    # Step 3 — build heatmap
    heatmap = patch_scores_to_heatmap(
        patch_scores, spatial_shape, output_size, gaussian_sigma,
    )

    # Step 4 — image score
    score = image_anomaly_score(heatmap)

    return score, heatmap


# ---------------------------------------------------------------------------
# 6. score_dataset
# ---------------------------------------------------------------------------
def score_dataset(
    all_patch_features: List[torch.Tensor],
    all_spatial_shapes: List[Tuple[int, int]],
    all_labels: List[int],
    all_masks: List[torch.Tensor],
    faiss_index: faiss.Index,
    memory_bank: torch.Tensor,
    output_size: int = 224,
    nn_k: int = 1,
    reweight_k: int = 3,
    gaussian_sigma: float = 4.0,
) -> Tuple[List[float], List[np.ndarray], List[int], List[np.ndarray]]:
    """Score every image in the test set.

    Iterates over pre-extracted features and calls score_image for each.

    Args:
        all_patch_features: List of (H*W, D) tensors, one per image.
        all_spatial_shapes: List of (H, W) tuples.
        all_labels: Ground-truth labels (0=good, 1=defective).
        all_masks: Ground-truth binary masks as tensors.
        faiss_index: Populated FAISS index.
        memory_bank: (M, D) memory bank tensor.
        output_size: Heatmap resolution.
        nn_k: Neighbours for distance lookup.
        reweight_k: Neighbours for re-weighting.
        gaussian_sigma: Gaussian blur sigma.

    Returns:
        image_scores: List of scalar anomaly scores.
        heatmaps: List of (output_size, output_size) numpy arrays.
        labels: Pass-through of ground-truth labels.
        gt_masks: Ground-truth masks as numpy arrays.
    """
    image_scores: List[float] = []
    heatmaps: List[np.ndarray] = []
    gt_masks: List[np.ndarray] = []

    for i, (features, shape) in enumerate(
        tqdm(
            zip(all_patch_features, all_spatial_shapes),
            total=len(all_patch_features),
            desc="Scoring images",
        )
    ):
        score, heatmap = score_image(
            features, shape, faiss_index, memory_bank,
            output_size, nn_k, reweight_k, gaussian_sigma,
        )
        image_scores.append(score)
        heatmaps.append(heatmap)

        # Convert mask tensor → numpy
        mask = all_masks[i]
        if isinstance(mask, torch.Tensor):
            mask = mask.squeeze().cpu().numpy()
        gt_masks.append(mask)

    logger.info(
        "Scored %d images — max_score=%.4f, min_score=%.4f",
        len(image_scores), max(image_scores), min(image_scores),
    )
    return image_scores, heatmaps, all_labels, gt_masks
