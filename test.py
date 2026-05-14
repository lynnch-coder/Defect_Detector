# test.py
"""
Evaluation Pipeline for PatchCore
===================================
This script tests the model against unseen data.
1. Loads the pre-trained memory bank from disk.
2. Extracts features from test images.
3. Scores patches using nearest-neighbor distances (FAISS).
4. Computes AUROC and generates heatmaps.
"""

import logging
import os
import json
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
import numpy as np

import config as cfg
from src.core.feature_extractor import extract_dataset_features
from src.core.memory_bank import load_memory_bank, build_faiss_index
from src.core.scoring import score_dataset
from src.evaluation.metrics import evaluate_all
from src.evaluation.visualize import save_qualitative_grid, plot_roc_curve, save_score_histogram

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. run_patchcore_inference
# ---------------------------------------------------------------------------
def run_patchcore_inference(category: str) -> Dict[str, float]:
    """Test pipeline for a single category."""
    import torchvision.models as models
    import torchvision.transforms as T
    from torch.utils.data import Dataset
    from PIL import Image

    logger.info("========== PatchCore Inference: %s ==========", category)

    # 1. Load memory bank
    bank_path = os.path.join(cfg.MEMORY_BANK_DIR, f"{category}_memory_bank.pt")
    if not os.path.exists(bank_path):
        logger.error("[%s] Memory bank not found at %s", category, bank_path)
        return {"image_auroc": 0.0, "pixel_auroc": 0.0, "pro": 0.0}

    memory_bank = load_memory_bank(bank_path, cfg.DEVICE)
    faiss_index = build_faiss_index(memory_bank)

    # 2. Backbone
    model = getattr(models, cfg.BACKBONE)(weights="IMAGENET1K_V1")
    model = model.to(cfg.DEVICE).eval()
    for p in model.parameters():
        p.requires_grad = False

    hook_dict: Dict[str, torch.Tensor] = {}

    def _hook(name: str):
        def fn(module, inp, out):
            hook_dict[name] = out.detach()
        return fn

    h1 = model.layer2.register_forward_hook(_hook("layer2"))
    h2 = model.layer3.register_forward_hook(_hook("layer3"))

    # 3. Test Dataset
    class _TestDataset(Dataset):
        def __init__(self) -> None:
            split_dir = os.path.join(cfg.DATA_ROOT, category, "test")
            gt_dir = os.path.join(cfg.DATA_ROOT, category, "ground_truth")
            self.image_paths, self.mask_paths, self.labels = [], [], []
            
            self.transform = T.Compose([
                T.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
                T.CenterCrop(cfg.IMG_SIZE),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            self.mask_transform = T.Compose([
                T.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
                T.CenterCrop(cfg.IMG_SIZE),
                T.ToTensor(),
            ])

            if os.path.isdir(split_dir):
                for defect_type in sorted(os.listdir(split_dir)):
                    defect_dir = os.path.join(split_dir, defect_type)
                    if not os.path.isdir(defect_dir): continue
                    is_good = (defect_type == "good")
                    label = 0 if is_good else 1
                    
                    for fname in sorted(os.listdir(defect_dir)):
                        if not fname.lower().endswith((".png", ".jpg", ".jpeg")): continue
                        self.image_paths.append(os.path.join(defect_dir, fname))
                        self.labels.append(label)
                        
                        if is_good:
                            self.mask_paths.append(None)
                        else:
                            mask_name = fname.replace(".png", "_mask.png")
                            mp = os.path.join(gt_dir, defect_type, mask_name)
                            self.mask_paths.append(mp if os.path.exists(mp) else None)
                            
        def __len__(self): return len(self.image_paths)
        
        def __getitem__(self, idx):
            img = self.transform(Image.open(self.image_paths[idx]).convert("RGB"))
            if self.mask_paths[idx]:
                mask = self.mask_transform(Image.open(self.mask_paths[idx]).convert("L"))
                mask = (mask > 0.5).float()
            else:
                mask = torch.zeros(1, cfg.IMG_SIZE, cfg.IMG_SIZE)
            return img, mask, self.labels[idx], self.image_paths[idx]

    ds = _TestDataset()
    dl = DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=True)

    # 4. Extract features
    all_features, all_shapes, all_labels, all_masks, all_images = [], [], [], [], []
    inv_norm = T.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
    
    logger.info("Extracting test features...")
    for images, masks, labels, paths in dl:
        images = images.to(cfg.DEVICE)
        from src.core.feature_extractor import extract_layer_features, locally_aware_patches, align_and_concat, flatten_patches
        with torch.no_grad():
            fl2, fl3 = extract_layer_features(model, images, hook_dict)
            fl2 = locally_aware_patches(fl2)
            fl3 = locally_aware_patches(fl3)
            combined = align_and_concat(fl2, fl3)
            
        for i in range(combined.shape[0]):
            flat, shape = flatten_patches(combined[i].unsqueeze(0))
            all_features.append(flat.cpu())
            all_shapes.append(shape)
            all_labels.append(int(labels[i]))
            all_masks.append(masks[i].cpu())
            img_vis = inv_norm(images[i].cpu()).permute(1,2,0).numpy()
            all_images.append(np.clip(img_vis, 0, 1))

    # 5. Score images
    logger.info("Scoring patches via FAISS...")
    image_scores, heatmaps, gt_masks_np = [], [], []
    for feat, shape in zip(all_features, all_shapes):
        from src.core.scoring import score_image
        score, heatmap = score_image(feat, shape, faiss_index, memory_bank, output_size=cfg.IMG_SIZE)
        image_scores.append(score)
        heatmaps.append(heatmap)
    for m in all_masks:
        gt_masks_np.append(m.squeeze().numpy() if isinstance(m, torch.Tensor) else m)

    # 6. Evaluate
    metrics = evaluate_all(image_scores, all_labels, heatmaps, gt_masks_np)
    logger.info("[%s] Metrics: %s", category, metrics)

    # 7. Visualizations
    vis_dir = os.path.join(cfg.HEATMAP_DIR, category)
    os.makedirs(vis_dir, exist_ok=True)
    
    defective_idx = [i for i, l in enumerate(all_labels) if l == 1]
    grid_idx = defective_idx[:8] if defective_idx else list(range(min(8, len(all_images))))
    
    save_qualitative_grid([all_images[i] for i in grid_idx], [gt_masks_np[i] for i in grid_idx], [heatmaps[i] for i in grid_idx], os.path.join(vis_dir, f"{category}_qualitative_grid.png"))
    plot_roc_curve(image_scores, all_labels, os.path.join(vis_dir, f"{category}_roc_curve.png"), title=f"ROC — {category}")
    save_score_histogram(image_scores, all_labels, os.path.join(vis_dir, f"{category}_score_histogram.png"), title=f"Scores — {category}")

    h1.remove()
    h2.remove()
    return metrics

# ---------------------------------------------------------------------------
# 2. main
# ---------------------------------------------------------------------------
def main() -> None:
    all_results = {}
    for category in cfg.CATEGORIES:
        all_results[category] = run_patchcore_inference(category)

    # Summary table
    print("\n" + "="*60)
    print(f"{'Category':<12} | {'Image AUROC':<12} | {'Pixel AUROC':<12} | {'PRO':<8}")
    print("-" * 60)
    for cat, m in all_results.items():
        print(f"{cat:<12} | {m['image_auroc']:<12.4f} | {m['pixel_auroc']:<12.4f} | {m['pro']:<8.4f}")
    
    avg_img = np.mean([m['image_auroc'] for m in all_results.values()])
    avg_pix = np.mean([m['pixel_auroc'] for m in all_results.values()])
    avg_pro = np.mean([m['pro'] for m in all_results.values()])
    print("-" * 60)
    print(f"{'AVERAGE':<12} | {avg_img:<12.4f} | {avg_pix:<12.4f} | {avg_pro:<8.4f}")
    print("=" * 60)

    # Save JSON report
    all_results["average"] = {"image_auroc": float(avg_img), "pixel_auroc": float(avg_pix), "pro": float(avg_pro)}
    report_path = os.path.join(cfg.OUTPUT_DIR, "patchcore_results.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2)

if __name__ == "__main__":
    main()
