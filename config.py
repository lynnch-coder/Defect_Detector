# config.py
"""
Central Configuration for PatchCore
=====================================
Every tuneable value lives here so nothing is hard-coded inside functions.
Change a value once and every file that imports config picks it up.

Usage in other files:
    import config as cfg
    model = load_backbone(cfg.BACKBONE, cfg.DEVICE)
"""

import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_ROOT       = "/content/mvtec"
DRIVE_DATA      = "/content/drive/MyDrive/defects-detection-CV"
OUTPUT_DIR      = "/content/drive/MyDrive/defects-detection-CV/outputs"
MEMORY_BANK_DIR = "/content/drive/MyDrive/defects-detection-CV/outputs/memory_banks"
FAISS_INDEX_DIR = "/content/drive/MyDrive/defects-detection-CV/outputs/faiss_indices"
HEATMAP_DIR     = "/content/drive/MyDrive/defects-detection-CV/outputs/heatmaps"

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
CATEGORIES = ["bottle", "carpet", "screw"]
IMG_SIZE   = 224
BATCH_SIZE = 32
NUM_WORKERS = 2

# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------
BACKBONE        = "wide_resnet50_2"
BACKBONE_LAYERS = ["layer2", "layer3"]

# ---------------------------------------------------------------------------
# Coreset
# ---------------------------------------------------------------------------
CORESET_RATIO = 0.01          # keep 1% of patches

# ---------------------------------------------------------------------------
# Scoring / Nearest-Neighbour
# ---------------------------------------------------------------------------
NN_K          = 1             # number of nearest neighbours for anomaly score
REWEIGHT_K    = 3             # neighbours used for score re-weighting
GAUSSIAN_SIGMA = 4.0         # sigma for Gaussian smoothing of heatmaps

# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
FEAT_CKPT_EVERY    = 10      # save feature extraction progress every N batches
CORESET_CKPT_EVERY = 500     # save coreset progress every N iterations

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
