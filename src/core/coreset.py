# coreset.py
"""
Coreset Sampling for PatchCore
================================
After the feature extractor produces hundreds of thousands of patch vectors,
storing ALL of them would be too slow and memory-heavy at test time.

Coreset sampling solves this: it picks a small, representative subset of
patches that still covers the full diversity of the training data.
The algorithm is "greedy farthest-point sampling" — it always picks the
next point that is FARTHEST from everything already selected, ensuring
maximum coverage.

How it fits: feature_extractor.py → THIS FILE → memory_bank.py
"""

import logging
import torch
from typing import Tuple
from tqdm import tqdm

logger = logging.getLogger(__name__)


def greedy_coreset_sampling(
    features: torch.Tensor,
    ratio: float,
    device: str,
    checkpoint_path: str = "",
    checkpoint_every: int = 500,
) -> torch.Tensor:
    """Select a diverse subset of feature vectors using farthest-point sampling.

    The algorithm works like placing cell-phone towers: you always put the
    next tower at the location that is farthest from any existing tower,
    guaranteeing the best possible coverage with a limited budget.

    1. Start with a random seed point.
    2. Compute distances from ALL points to the selected set.
    3. Pick the point with the largest minimum-distance.
    4. Repeat until we have enough points.

    Includes a checkpoint system so long runs can resume after a disconnect.

    Args:
        features: (N, D) tensor of patch feature vectors (on CPU).
        ratio: Fraction of points to keep (e.g. 0.01 = 1%).
        device: 'cuda' or 'cpu' — GPU is used for distance computation.
        checkpoint_path: Path to save/load checkpoint. Empty string = no checkpointing.
        checkpoint_every: Save a checkpoint every N iterations.

    Returns:
        Tensor of selected indices, shape (n_select,).
    """
    n: int = features.shape[0]
    n_select: int = max(1, int(n * ratio))
    logger.info(
        "Coreset sampling: keeping %d / %d patches (%.1f%%)",
        n_select, n, ratio * 100,
    )

    # Move features to GPU for fast distance computation
    features_gpu: torch.Tensor = features.to(device)

    # --- Try to resume from checkpoint ---
    selected_idx: list[int] = []
    min_distances: torch.Tensor = torch.full((n,), float("inf"), device=device)
    start_iter: int = 0

    if checkpoint_path:
        try:
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
            selected_idx = ckpt["selected_idx"]
            min_distances = ckpt["min_distances"].to(device)
            start_iter = len(selected_idx)
            logger.info("Resumed coreset checkpoint at iteration %d", start_iter)
        except FileNotFoundError:
            logger.info("No coreset checkpoint found — starting fresh.")
        except Exception as exc:
            logger.warning("Could not load coreset checkpoint (%s) — starting fresh.", exc)

    # Pick random seed if starting fresh
    if len(selected_idx) == 0:
        seed: int = torch.randint(0, n, (1,)).item()
        selected_idx.append(seed)
        start_iter = 1
        # Initialize distances from the seed point
        diff = features_gpu - features_gpu[seed].unsqueeze(0)
        min_distances = torch.sum(diff ** 2, dim=1)

    # --- Main loop: always pick the farthest point ---
    remaining = n_select - len(selected_idx)
    pbar = tqdm(
        range(remaining),
        desc="Coreset sampling",
        initial=len(selected_idx),
        total=n_select,
    )
    for i in pbar:
        # Pick point with largest minimum distance to selected set
        next_idx: int = torch.argmax(min_distances).item()
        selected_idx.append(next_idx)

        # Update minimum distances with the newly added point
        diff = features_gpu - features_gpu[next_idx].unsqueeze(0)
        distances = torch.sum(diff ** 2, dim=1)
        min_distances = torch.minimum(min_distances, distances)

        # Periodic checkpoint
        if checkpoint_path and (i + 1) % checkpoint_every == 0:
            try:
                torch.save(
                    {
                        "selected_idx": selected_idx,
                        "min_distances": min_distances.cpu(),
                    },
                    checkpoint_path,
                )
                logger.info("Coreset checkpoint saved (%d / %d)", len(selected_idx), n_select)
            except Exception as exc:
                logger.warning("Coreset checkpoint save failed: %s", exc)

    pbar.close()

    # Clean up checkpoint after successful completion
    if checkpoint_path:
        import os
        try:
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
                logger.info("Coreset checkpoint file cleaned up.")
        except Exception as exc:
            logger.warning("Could not remove checkpoint: %s", exc)

    result = torch.tensor(selected_idx, dtype=torch.long)
    logger.info("Coreset selection complete: %d indices", len(result))
    return result


def subsample_memory_bank(
    features: torch.Tensor,
    ratio: float,
    device: str,
    checkpoint_path: str = "",
    checkpoint_every: int = 500,
) -> torch.Tensor:
    """Apply coreset subsampling and return the reduced memory bank.

    This is a convenience wrapper: it calls greedy_coreset_sampling to get
    the indices, then uses them to slice the original feature tensor.

    Args:
        features: (N, D) tensor of all patch features.
        ratio: Fraction to keep (e.g. 0.01).
        device: 'cuda' or 'cpu'.
        checkpoint_path: Path for checkpoint file.
        checkpoint_every: Checkpoint interval.

    Returns:
        Reduced feature tensor of shape (n_select, D) on CPU.
    """
    indices = greedy_coreset_sampling(
        features, ratio, device,
        checkpoint_path=checkpoint_path,
        checkpoint_every=checkpoint_every,
    )
    reduced = features[indices].cpu()
    logger.info(
        "Memory bank subsampled: %s → %s",
        features.shape, reduced.shape,
    )
    return reduced
