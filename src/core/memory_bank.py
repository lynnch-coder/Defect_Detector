# memory_bank.py
"""
Memory Bank for PatchCore
==========================
This file is the "warehouse" of our defect detector. It handles saving,
loading, and indexing the compressed patch features (memory bank) that
were produced by coreset sampling.

At test time we need to quickly find the nearest normal patch for every
test patch. Doing this with raw PyTorch would be too slow, so we build
a FAISS index — a specialised data structure for ultra-fast nearest-
neighbour search on GPU.

How it fits:
  Train:  coreset.py → THIS FILE (save)
  Test:   THIS FILE (load + build index + search) → scoring.py
"""

import logging
import os
import numpy as np
import torch
import faiss

from typing import Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. save_memory_bank
# ---------------------------------------------------------------------------
def save_memory_bank(
    memory_bank: torch.Tensor,
    save_path: str,
) -> None:
    """Serialize the memory bank tensor to disk.

    Creates parent directories automatically if they don't exist.

    Args:
        memory_bank: (M, D) tensor of coreset patch features.
        save_path: File path ending in .pt where the tensor is saved.
    """
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(memory_bank, save_path)
        logger.info(
            "Memory bank saved: %s → %s",
            memory_bank.shape, save_path,
        )
    except Exception as exc:
        logger.error("Failed to save memory bank to %s: %s", save_path, exc)
        raise


# ---------------------------------------------------------------------------
# 2. load_memory_bank
# ---------------------------------------------------------------------------
def load_memory_bank(
    load_path: str,
    device: str,
) -> torch.Tensor:
    """Load a memory bank from disk and move it to the given device.

    Args:
        load_path: Path to the .pt file.
        device: 'cuda' or 'cpu'.

    Returns:
        Memory bank tensor of shape (M, D) on the requested device.
    """
    try:
        memory_bank = torch.load(load_path, map_location=device, weights_only=True)
        logger.info(
            "Memory bank loaded: %s from %s",
            memory_bank.shape, load_path,
        )
        return memory_bank
    except FileNotFoundError:
        logger.error("Memory bank file not found: %s", load_path)
        raise
    except Exception as exc:
        logger.error("Failed to load memory bank from %s: %s", load_path, exc)
        raise


# ---------------------------------------------------------------------------
# 3. build_faiss_index
# ---------------------------------------------------------------------------
def build_faiss_index(
    memory_bank: torch.Tensor,
) -> faiss.Index:
    """Build a FAISS GPU index for fast nearest-neighbour search.

    We use an L2 (Euclidean distance) flat index — exact search, no
    approximation. The index is placed on GPU for maximum speed.

    Args:
        memory_bank: (M, D) tensor of coreset features.

    Returns:
        A faiss GpuIndexFlatL2 populated with all memory bank vectors.
    """
    # Convert to float32 numpy (faiss requirement)
    vectors = memory_bank.detach().cpu().numpy().astype(np.float32)
    dim: int = vectors.shape[1]

    # Build CPU index first, then move to GPU
    cpu_index = faiss.IndexFlatL2(dim)

    try:
        # Try GPU index
        gpu_resource = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(gpu_resource, 0, cpu_index)
        gpu_index.add(vectors)
        logger.info(
            "FAISS GPU index built: %d vectors, dim=%d",
            gpu_index.ntotal, dim,
        )
        return gpu_index
    except Exception as exc:
        logger.warning("FAISS GPU failed (%s), falling back to CPU index.", exc)
        cpu_index.add(vectors)
        logger.info(
            "FAISS CPU index built: %d vectors, dim=%d",
            cpu_index.ntotal, dim,
        )
        return cpu_index


# ---------------------------------------------------------------------------
# 4. faiss_knn_search
# ---------------------------------------------------------------------------
def faiss_knn_search(
    index: faiss.Index,
    query_features: torch.Tensor,
    k: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Search the FAISS index for the k nearest neighbours.

    For each test patch, find the closest patch(es) in the memory bank.
    The distance tells us how "abnormal" that patch is — higher distance
    means it looks nothing like any normal training patch.

    Args:
        index: A populated FAISS index (GPU or CPU).
        query_features: (Q, D) tensor of test patch features.
        k: Number of nearest neighbours to return.

    Returns:
        distances: (Q, k) numpy array of L2 distances.
        indices: (Q, k) numpy array of neighbour indices.
    """
    queries = query_features.detach().cpu().numpy().astype(np.float32)
    distances, indices = index.search(queries, k)
    logger.debug(
        "KNN search: %d queries, k=%d, max_dist=%.4f",
        queries.shape[0], k, distances.max(),
    )
    return distances, indices
