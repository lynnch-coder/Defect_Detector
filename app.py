"""
Streamlit dashboard — PatchCore vs Basic CLIP vs Improved CLIP
Run: streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

import config as cfg


CATEGORIES = ["bottle", "carpet", "screw"]

METRICS = {
    "image_auroc": "Image AUROC",
    "pixel_auroc": "Pixel AUROC",
    "pro":         "PRO",
}

METHOD_COLORS = {
    "PatchCore":      "#1f77b4",
    "Basic CLIP":     "#2ca02c",
    "Improved CLIP":  "#ff7f0e",
}

# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────

def load_json(path: str) -> Dict[str, dict]:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_results() -> Tuple[Dict, Dict, Dict]:
    """Load all three result JSONs. Returns (patchcore, basic_clip, improved_clip)."""
    outputs = Path(cfg.OUTPUT_DIR)
    results_dir = Path(cfg.RESULTS_DIR)

    patchcore     = load_json(str(outputs / "patchcore_results.json"))
    basic_clip    = load_json(str(outputs / "clip_results" / "basic_clip_results.json"))
    improved_clip = load_json(str(outputs / "clip_results" / "clip_results.json"))

    return patchcore, basic_clip, improved_clip


# ─────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading CLIP model...")
def get_clip_detector():
    try:
        from src.baseline.clip_anomaly import CLIPAnomalyDetector
        return CLIPAnomalyDetector(), None
    except Exception as exc:
        return None, exc


@st.cache_resource(show_spinner=False)
def get_patchcore_predictor(category: str):
    try:
        from src.core.patchcore_inference import PatchCorePredictor
        return PatchCorePredictor(category), None
    except Exception as exc:
        return None, exc


def calibration_for(category: str, improved_clip_results: Dict) -> Optional[object]:
    from src.baseline.clip_anomaly import ClipCalibration
    data = improved_clip_results.get(category, {}).get("calibration")
    if not data:
        return None
    return ClipCalibration.from_dict(data)


# ─────────────────────────────────────────────
# Image helpers
# ─────────────────────────────────────────────

def resize_for_model(image: Image.Image, size: int = cfg.IMG_SIZE) -> Image.Image:
    return image.convert("RGB").resize((size, size), Image.BICUBIC)


def normalize_heatmap(heatmap: np.ndarray, vmin=-2.0, vmax=4.0) -> np.ndarray:
    clipped = np.clip(heatmap, vmin, vmax)
    return (clipped - vmin) / max(vmax - vmin, 1e-8)


def heatmap_to_rgb(heatmap: np.ndarray) -> np.ndarray:
    heatmap = np.clip(heatmap, 0.0, 1.0)
    red   = np.clip(2.0 * heatmap, 0.0, 1.0)
    blue  = np.clip(2.0 * (1.0 - heatmap), 0.0, 1.0)
    green = 1.0 - np.abs(2.0 * heatmap - 1.0)
    return np.stack([red, green, blue], axis=-1)


def make_overlay(image_arr: np.ndarray, heatmap: np.ndarray,
                 alpha=0.45, assume_normalized=False) -> np.ndarray:
    if assume_normalized:
        heatmap = np.clip(heatmap, 0.0, 1.0)
    elif heatmap.max() - heatmap.min() > 1e-8:
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    else:
        heatmap = np.zeros_like(heatmap)
    blended = (1.0 - alpha) * image_arr + alpha * heatmap_to_rgb(heatmap)
    return np.clip(blended * 255, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────
# Metrics charts
# ─────────────────────────────────────────────

def metrics_dataframe(patchcore, basic_clip, improved_clip) -> pd.DataFrame:
    rows = []
    methods = [
        ("PatchCore",     patchcore),
        ("Basic CLIP",    basic_clip),
        ("Improved CLIP", improved_clip),
    ]
    for method, results in methods:
        for category in CATEGORIES:
            cat_data = results.get(category, {})
            for key, label in METRICS.items():
                value = cat_data.get(key)
                if value is None:
                    continue
                rows.append({
                    "Method":   method,
                    "Category": category,
                    "Metric":   label,
                    "Value":    float(value),
                })
    return pd.DataFrame(rows)


def show_metric_charts(patchcore, basic_clip, improved_clip) -> None:
    df = metrics_dataframe(patchcore, basic_clip, improved_clip)
    if df.empty:
        st.info(
            "No dataset metrics found yet. Run the notebooks on Colab first "
            "to generate the JSON result files."
        )
        return

    for metric_label in METRICS.values():
        metric_df = df[df["Metric"] == metric_label]
        if metric_df.empty:
            continue

        # Check which methods have data for this metric
        methods_present = metric_df["Method"].unique().tolist()

        fig = px.bar(
            metric_df,
            x="Category",
            y="Value",
            color="Method",
            barmode="group",
            range_y=[0, 1],
            text=metric_df["Value"].map(lambda v: f"{v:.3f}"),
            title=metric_label,
            color_discrete_map=METHOD_COLORS,
            category_orders={"Method": ["PatchCore", "Basic CLIP", "Improved CLIP"]},
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(
            height=380,
            yaxis_title="Score",
            xaxis_title="",
            legend_title="",
            margin=dict(l=20, r=20, t=60, b=30),
        )

        # Note if Basic CLIP has no pixel-level metrics
        if metric_label in ("Pixel AUROC", "PRO") and "Basic CLIP" not in methods_present:
            st.caption(
                "Note: Pixel AUROC and PRO were not computed for the basic CLIP baseline "
                "because its 7x7 sliding window heatmap lacks the spatial resolution required "
                "for pixel-level evaluation. The improved version uses multi-scale overlapping "
                "crops specifically to address this."
            )

        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# Upload inference
# ─────────────────────────────────────────────

def display_uploaded_inference(
    uploaded_image: Image.Image,
    selected_category: str,
    patchcore_results: Dict,
    basic_clip_results: Dict,
    improved_clip_results: Dict,
) -> None:
    image_224    = resize_for_model(uploaded_image, cfg.IMG_SIZE)
    image_arr    = np.asarray(image_224).astype(np.float32) / 255.0
    detector, clip_error = get_clip_detector()

    # ── PatchCore ──────────────────────────────
    pc_score, pc_heatmap, pc_image = None, None, image_arr
    pc_predictor, pc_error = get_patchcore_predictor(selected_category)
    if pc_predictor:
        pc_score, pc_heatmap, pc_image = pc_predictor.predict(image_224)

    # ── Basic CLIP (no calibration) ────────────
    basic_score, basic_heatmap = None, None
    if detector:
        try:
            basic_score   = detector.image_score(image_224, selected_category, calibration=None)
            basic_heatmap = detector.heatmap(image_224, selected_category, calibration=None)
        except Exception as exc:
            clip_error = exc

    # ── Improved CLIP (with calibration) ───────
    improved_score, improved_heatmap = None, None
    if detector:
        try:
            cal = calibration_for(selected_category, improved_clip_results)
            improved_score   = detector.image_score(image_224, selected_category, calibration=cal)
            improved_heatmap = detector.heatmap(image_224, selected_category, calibration=cal)
        except Exception as exc:
            clip_error = exc

    # ── Scores row ─────────────────────────────
    score_cols = st.columns(3)
    with score_cols[0]:
        if pc_score is None:
            st.warning(f"PatchCore unavailable: {pc_error}")
        else:
            st.metric("PatchCore score", f"{pc_score:.4f}")
    with score_cols[1]:
        if basic_score is None:
            st.warning(f"Basic CLIP unavailable: {clip_error}")
        else:
            st.metric("Basic CLIP score", f"{basic_score:.4f}")
    with score_cols[2]:
        if improved_score is None:
            st.warning(f"Improved CLIP unavailable: {clip_error}")
        else:
            st.metric("Improved CLIP score", f"{improved_score:.4f}")

    # ── Heatmap row ────────────────────────────
    img_cols = st.columns(4)
    with img_cols[0]:
        st.image(image_224, caption="Uploaded image", use_container_width=True)
    with img_cols[1]:
        if pc_heatmap is not None:
            st.image(
                make_overlay(pc_image, pc_heatmap),
                caption="PatchCore heatmap",
                use_container_width=True,
            )
        else:
            st.info("Run `python train.py` first so PatchCore memory banks exist.")
    with img_cols[2]:
        if basic_heatmap is not None:
            norm = (basic_heatmap - basic_heatmap.min()) / max(
                basic_heatmap.max() - basic_heatmap.min(), 1e-8)
            st.image(
                make_overlay(image_arr, norm, assume_normalized=True),
                caption="Basic CLIP heatmap",
                use_container_width=True,
            )
        else:
            st.info("Install `open_clip_torch` to enable CLIP inference.")
    with img_cols[3]:
        if improved_heatmap is not None:
            norm = normalize_heatmap(improved_heatmap)
            st.image(
                make_overlay(image_arr, norm, assume_normalized=True),
                caption="Improved CLIP heatmap",
                use_container_width=True,
            )
        else:
            st.info("Install `open_clip_torch` to enable CLIP inference.")

    # ── Method comparison explanation ──────────
    with st.expander("How the three methods differ", expanded=False):
        st.markdown("""
**PatchCore** stores feature vectors of every patch from normal training images.
At test time it finds the nearest stored patch for each image patch.
Patches far from anything normal are flagged. This is the most accurate method.

**Basic CLIP** encodes the image and two text prompts ("a normal bottle",
"a broken bottle") and scores the image by how much closer it is to the defect prompt.
No training images are used at all. Simple and fast but uncalibrated.

**Improved CLIP** adds three things on top of the basic version:
(1) prompt ensembling using 5 specific prompts per defect type,
(2) calibration by running on 30 normal training images to set a reference baseline,
(3) multi-scale overlapping crops for sharper heatmaps.
Scores are more meaningful but still lower than PatchCore.
        """)


def suggest_category_ui(image: Image.Image) -> Tuple[str, Dict[str, float]]:
    detector, clip_error = get_clip_detector()
    if not detector:
        st.info(f"Automatic category suggestion is unavailable: {clip_error}")
        return CATEGORIES[0], {}
    suggested, scores = detector.suggest_category(resize_for_model(image, cfg.IMG_SIZE))
    return suggested, scores


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main() -> None:
    st.set_page_config(page_title="Industrial Defect Detector", layout="wide")
    st.title("Industrial Defect Detection: PatchCore vs Basic CLIP vs Improved CLIP")

    patchcore_results, basic_clip_results, improved_clip_results = load_results()

    upload_tab, metrics_tab = st.tabs(["Upload & Results", "Metrics Comparison"])

    with upload_tab:
        uploaded_file = st.file_uploader(
            "Upload a product image",
            type=["png", "jpg", "jpeg", "bmp"],
        )

        if uploaded_file is None:
            st.info("Upload a bottle, carpet, or screw image to compare all three methods.")
        else:
            uploaded_image = Image.open(uploaded_file).convert("RGB")
            suggested, suggestion_scores = suggest_category_ui(uploaded_image)

            category_index = CATEGORIES.index(suggested) if suggested in CATEGORIES else 0
            selected_category = st.selectbox(
                "Category",
                CATEGORIES,
                index=category_index,
                help="PatchCore memory banks are category-specific.",
            )

            if suggestion_scores:
                st.caption(
                    "CLIP category suggestion: "
                    + ", ".join(
                        f"{cat}={score:.3f}"
                        for cat, score in sorted(
                            suggestion_scores.items(),
                            key=lambda x: x[1],
                            reverse=True,
                        )
                    )
                )

            display_uploaded_inference(
                uploaded_image,
                selected_category,
                patchcore_results,
                basic_clip_results,
                improved_clip_results,
            )

    with metrics_tab:
        st.subheader("Dataset-level comparison")
        st.write(
            "These charts compare saved evaluation metrics for the full test set. "
            "They are not computed from the single uploaded image."
        )

        # Advantages and disadvantages table
        with st.expander("Method comparison summary", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**PatchCore**")
                st.markdown("+ Best accuracy across all categories")
                st.markdown("+ Precise pixel-level localization")
                st.markdown("+ Proven method (CVPR 2022)")
                st.markdown("- Needs good training images per category")
                st.markdown("- Requires memory bank storage")
                st.markdown("- Sensitive to rotation (screw)")
            with col2:
                st.markdown("**Basic CLIP**")
                st.markdown("+ Zero training images needed")
                st.markdown("+ Works on any product instantly")
                st.markdown("+ Simple to understand and explain")
                st.markdown("- Scores are uncalibrated")
                st.markdown("- Coarse heatmaps (7x7 grid)")
                st.markdown("- Fails badly on screw (0.548)")
            with col3:
                st.markdown("**Improved CLIP**")
                st.markdown("+ Calibrated, honest scores")
                st.markdown("+ Better heatmaps (multi-scale)")
                st.markdown("+ Specific prompts per defect type")
                st.markdown("- Needs 30 good images for calibration")
                st.markdown("- Still loses to PatchCore overall")
                st.markdown("- More complex to implement")

        show_metric_charts(patchcore_results, basic_clip_results, improved_clip_results)


if __name__ == "__main__":
    main()
