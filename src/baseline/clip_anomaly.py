# clip_anomaly.py
"""
Improved CLIP baseline for industrial anomaly detection.

This module keeps CLIP as a baseline, but makes it stronger than the first
notebook version:
  - prompt ensembling per category
  - zero-shot category suggestion
  - calibration on good training images
  - overlapping multi-scale crops for localization

CLIP is still not expected to beat PatchCore for pixel localization. It was
trained for image-text alignment, not for dense industrial defect maps.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)


CATEGORIES = ["bottle", "carpet", "screw"]


PROMPT_BANK: Dict[str, Dict[str, List[str]]] = {
    "bottle": {
        "normal": [
            "a close-up photo of a normal bottle",
            "a good industrial bottle with no defect",
            "an undamaged bottle surface",
            "a clean bottle opening with regular shape",
            "a defect-free bottle in an inspection image",
        ],
        "defect": [
            "a bottle with a crack",
            "a bottle with a broken edge",
            "a bottle with contamination",
            "a bottle with a chip or missing part",
            "a defective bottle in an industrial inspection photo",
        ],
        "category": [
            "a close-up inspection photo of a bottle",
            "a bottle product",
            "the circular mouth of a bottle",
        ],
    },
    "carpet": {
        "normal": [
            "a close-up photo of normal carpet texture",
            "a clean carpet with uniform woven texture",
            "a good carpet sample with no defect",
            "a regular fabric weave pattern",
            "a defect-free carpet inspection image",
        ],
        "defect": [
            "a carpet with a hole",
            "a carpet with a cut",
            "a carpet with a color stain",
            "a carpet with pulled thread",
            "a carpet with metal contamination",
            "a defective carpet texture in an industrial inspection photo",
        ],
        "category": [
            "a close-up inspection photo of carpet texture",
            "a woven carpet sample",
            "a fabric texture product",
        ],
    },
    "screw": {
        "normal": [
            "a close-up photo of a normal screw",
            "a good screw with clean even threads",
            "an undamaged metallic screw",
            "a screw with regular head and thread geometry",
            "a defect-free screw inspection image",
        ],
        "defect": [
            "a screw with damaged threads",
            "a screw with scratches on the head",
            "a screw with scratches on the neck",
            "a screw with a bent or manipulated front",
            "a defective metallic screw in an industrial inspection photo",
        ],
        "category": [
            "a close-up inspection photo of a screw",
            "a metallic screw product",
            "a screw with threads",
        ],
    },
}


@dataclass
class ClipCalibration:
    """Z-score calibration values estimated from good training images."""

    image_mean: float = 0.0
    image_std: float = 1.0
    patch_mean: float = 0.0
    patch_std: float = 1.0
    num_images: int = 0
    num_patches: int = 0

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "ClipCalibration":
        if not data:
            return cls()
        return cls(
            image_mean=float(data.get("image_mean", 0.0)),
            image_std=max(float(data.get("image_std", 1.0)), 1e-6),
            patch_mean=float(data.get("patch_mean", 0.0)),
            patch_std=max(float(data.get("patch_std", 1.0)), 1e-6),
            num_images=int(data.get("num_images", 0)),
            num_patches=int(data.get("num_patches", 0)),
        )

    def normalize_image_score(self, score: float) -> float:
        return float((score - self.image_mean) / self.image_std)

    def normalize_patch_scores(self, scores: np.ndarray) -> np.ndarray:
        return (scores - self.patch_mean) / self.patch_std

    def to_dict(self) -> dict:
        return asdict(self)


def ensure_rgb(image: Image.Image) -> Image.Image:
    """Convert PIL images to RGB while preserving content."""
    return image.convert("RGB") if image.mode != "RGB" else image


def resize_for_model(image: Image.Image, image_size: int = 224) -> Image.Image:
    """Use a fixed square size so masks, heatmaps, and PatchCore align."""
    return ensure_rgb(image).resize((image_size, image_size), Image.BICUBIC)


def build_windows(
    width: int,
    height: int,
    patch_ratios: Sequence[float] = (0.32, 0.48, 0.64),
    stride_ratio: float = 0.45,
) -> List[Tuple[int, int, int, int]]:
    """Create overlapping multi-scale crop windows over an image."""
    windows: List[Tuple[int, int, int, int]] = []
    base = min(width, height)

    for ratio in patch_ratios:
        size = max(16, min(width, height, int(round(base * ratio))))
        stride = max(4, int(round(size * stride_ratio)))

        xs = list(range(0, max(width - size + 1, 1), stride))
        ys = list(range(0, max(height - size + 1, 1), stride))

        if not xs or xs[-1] != width - size:
            xs.append(max(width - size, 0))
        if not ys or ys[-1] != height - size:
            ys.append(max(height - size, 0))

        for y in ys:
            for x in xs:
                windows.append((x, y, x + size, y + size))

    # Deduplicate edge windows produced by the final-position correction.
    return sorted(set(windows))


def normalize_for_display(
    heatmap: np.ndarray,
    vmin: float = -2.0,
    vmax: float = 4.0,
) -> np.ndarray:
    """Map calibrated heatmaps to [0, 1] using fixed display limits."""
    clipped = np.clip(heatmap, vmin, vmax)
    return (clipped - vmin) / max(vmax - vmin, 1e-8)


class CLIPAnomalyDetector:
    """CLIP-based anomaly scoring and localization helper."""

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: Optional[str] = None,
    ) -> None:
        try:
            import open_clip
        except ImportError as exc:
            raise ImportError(
                "open_clip_torch is required for the improved CLIP baseline. "
                "Install it with `pip install open_clip_torch`."
            ) from exc

        self.open_clip = open_clip
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=self.device,
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self._prompt_cache: Dict[Tuple[str, str], torch.Tensor] = {}
        logger.info("Loaded CLIP model %s/%s on %s", model_name, pretrained, self.device)

    def _encode_texts(self, texts: Sequence[str]) -> torch.Tensor:
        tokens = self.tokenizer(list(texts)).to(self.device)
        with torch.no_grad():
            features = self.model.encode_text(tokens)
            features = F.normalize(features.float(), dim=-1)
        return features

    def prompt_features(self, category: str, kind: str) -> torch.Tensor:
        if category not in PROMPT_BANK:
            raise ValueError(f"Unknown category: {category}")
        key = (category, kind)
        if key not in self._prompt_cache:
            self._prompt_cache[key] = self._encode_texts(PROMPT_BANK[category][kind])
        return self._prompt_cache[key]

    def _preprocess_images(self, images: Sequence[Image.Image]) -> torch.Tensor:
        tensors = [self.preprocess(ensure_rgb(img)) for img in images]
        return torch.stack(tensors, dim=0).to(self.device)

    def _score_tensors(self, tensors: torch.Tensor, category: str) -> np.ndarray:
        normal_features = self.prompt_features(category, "normal")
        defect_features = self.prompt_features(category, "defect")

        with torch.no_grad():
            image_features = self.model.encode_image(tensors)
            image_features = F.normalize(image_features.float(), dim=-1)

            normal_sim = image_features @ normal_features.T
            defect_sim = image_features @ defect_features.T

            normal_score = normal_sim.mean(dim=1)
            top_k = min(3, defect_sim.shape[1])
            defect_score = defect_sim.topk(top_k, dim=1).values.mean(dim=1)
            scores = defect_score - normal_score

        return scores.detach().cpu().numpy().astype(np.float32)

    def raw_image_score(self, image: Image.Image, category: str) -> float:
        tensor = self._preprocess_images([image])
        return float(self._score_tensors(tensor, category)[0])

    def image_score(
        self,
        image: Image.Image,
        category: str,
        calibration: Optional[ClipCalibration] = None,
    ) -> float:
        raw = self.raw_image_score(image, category)
        return calibration.normalize_image_score(raw) if calibration else raw

    def suggest_category(self, image: Image.Image) -> Tuple[str, Dict[str, float]]:
        """Suggest bottle/carpet/screw with category-only CLIP prompts."""
        category_features = {
            category: self.prompt_features(category, "category")
            for category in CATEGORIES
        }
        tensor = self._preprocess_images([image])
        with torch.no_grad():
            image_features = self.model.encode_image(tensor)
            image_features = F.normalize(image_features.float(), dim=-1)

        scores: Dict[str, float] = {}
        for category, text_features in category_features.items():
            sim = image_features @ text_features.T
            scores[category] = float(sim.mean().item())

        suggested = max(scores, key=scores.get)
        return suggested, scores

    def crop_scores(
        self,
        image: Image.Image,
        category: str,
        windows: Sequence[Tuple[int, int, int, int]],
        batch_size: int = 64,
    ) -> np.ndarray:
        """Score crop windows with CLIP in batches."""
        scores: List[np.ndarray] = []
        crops: List[Image.Image] = []

        for window in windows:
            crops.append(ensure_rgb(image).crop(window))
            if len(crops) == batch_size:
                scores.append(self._score_tensors(self._preprocess_images(crops), category))
                crops = []

        if crops:
            scores.append(self._score_tensors(self._preprocess_images(crops), category))

        if not scores:
            return np.empty((0,), dtype=np.float32)
        return np.concatenate(scores).astype(np.float32)

    def heatmap(
        self,
        image: Image.Image,
        category: str,
        calibration: Optional[ClipCalibration] = None,
        patch_ratios: Sequence[float] = (0.32, 0.48, 0.64),
        stride_ratio: float = 0.45,
        batch_size: int = 64,
        gaussian_sigma: Optional[float] = None,
    ) -> np.ndarray:
        """Generate a dense CLIP heatmap from overlapping crop scores."""
        image = ensure_rgb(image)
        width, height = image.size
        windows = build_windows(width, height, patch_ratios, stride_ratio)
        raw_scores = self.crop_scores(image, category, windows, batch_size=batch_size)
        if calibration:
            scores = calibration.normalize_patch_scores(raw_scores)
        else:
            scores = raw_scores

        score_sum = np.zeros((height, width), dtype=np.float32)
        weight_sum = np.zeros((height, width), dtype=np.float32)

        for (x1, y1, x2, y2), score in zip(windows, scores):
            h = max(y2 - y1, 1)
            w = max(x2 - x1, 1)
            # Hann weights reduce hard crop boundaries in overlapping maps.
            wy = np.hanning(h) if h > 2 else np.ones(h, dtype=np.float32)
            wx = np.hanning(w) if w > 2 else np.ones(w, dtype=np.float32)
            weights = np.outer(wy, wx).astype(np.float32)
            weights = np.maximum(weights, 0.05)

            score_sum[y1:y2, x1:x2] += float(score) * weights
            weight_sum[y1:y2, x1:x2] += weights

        heatmap = score_sum / np.maximum(weight_sum, 1e-6)

        if gaussian_sigma is None:
            gaussian_sigma = max(1.0, min(width, height) / 70.0)
        heatmap = gaussian_filter(heatmap, sigma=gaussian_sigma)
        return heatmap.astype(np.float32)

    def fit_calibration(
        self,
        category: str,
        good_image_paths: Sequence[str],
        image_size: int = 224,
        max_images: int = 30,
        max_patch_scores: int = 2500,
    ) -> ClipCalibration:
        """Estimate score normalization from good training images."""
        selected_paths = list(good_image_paths)[:max_images]
        image_scores: List[float] = []
        patch_scores: List[np.ndarray] = []

        for path in selected_paths:
            try:
                image = resize_for_model(Image.open(path), image_size)
            except Exception as exc:
                logger.warning("Skipping calibration image %s: %s", path, exc)
                continue

            image_scores.append(self.raw_image_score(image, category))
            windows = build_windows(image.width, image.height)
            raw = self.crop_scores(image, category, windows)
            if raw.size:
                patch_scores.append(raw)

        if image_scores:
            image_arr = np.array(image_scores, dtype=np.float32)
            image_mean = float(image_arr.mean())
            image_std = float(max(image_arr.std(), 1e-6))
        else:
            image_mean, image_std = 0.0, 1.0

        if patch_scores:
            patch_arr = np.concatenate(patch_scores).astype(np.float32)
            if patch_arr.size > max_patch_scores:
                idx = np.linspace(0, patch_arr.size - 1, max_patch_scores).astype(np.int64)
                patch_arr = patch_arr[idx]
            patch_mean = float(patch_arr.mean())
            patch_std = float(max(patch_arr.std(), 1e-6))
            num_patches = int(patch_arr.size)
        else:
            patch_mean, patch_std, num_patches = 0.0, 1.0, 0

        calibration = ClipCalibration(
            image_mean=image_mean,
            image_std=image_std,
            patch_mean=patch_mean,
            patch_std=patch_std,
            num_images=len(image_scores),
            num_patches=num_patches,
        )
        logger.info("CLIP calibration for %s: %s", category, calibration)
        return calibration


def category_from_path(path: str) -> Optional[str]:
    parts = path.replace("\\", "/").split("/")
    for category in CATEGORIES:
        if category in parts:
            return category
    return None

