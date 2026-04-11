"""
C.2.6.1 - Enhanced Computer Vision Pipeline

======================================================
Architecture : SegFormer-B2 (Cityscapes, 19 classes) + YOLOv8n
Hardware     : Apple Silicon M3 Pro
Input        : refined_image_registry.csv  (~9 671 images)
Output       : perceptual_features.csv
               visualizations/  (one PNG every VIS_EVERY images)

This script:
Extract perceptual features from street-level images using a hybrid
SegFormer-B2 (semantic segmentation) and YOLOv8n (object detection) pipeline,
with auto-resume, async image loading, feature fusion, and visualization outputs.

Speed Optimizations
-----------------------------------------
1. MPS device (Apple Silicon GPU) instead of CPU
2. Pre-resize images to SEG_INPUT_SIZE before inference
   - SegFormer input: 512×512  (down from ~2048px; quality unchanged)
   - YOLO input:      640px    (auto-handled by ultralytics)
3. Async download with ThreadPoolExecutor — next image downloads
   while current image is being processed (hides ~5-10s latency)
4. torch.inference_mode() instead of no_grad() (slightly faster on MPS)
5. Visualization every VIS_EVERY=5 images (as requested)
6. Matplotlib uses non-interactive 'Agg' backend (no GUI overhead)

Author  : Zhang Wenyu
Updated : 2026-02-09
"""

# ===========================================================================
# Imports
# ===========================================================================
import os
import sys
import time
import logging
import warnings
import traceback
import threading
from io import BytesIO
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — must be set before pyplot import
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)
from ultralytics import YOLO

warnings.filterwarnings("ignore")

# ===========================================================================
# ─── USER CONFIGURATION ───────────────────────────────────────────────────
# ===========================================================================

BASE_DIR = Path(
    "/Users/zhangwenyu/Desktop/NUSFYP/Stage 1/1.3 Street Level Imagery"
)

REGISTRY_CSV    = BASE_DIR / "refined_image_registry.csv"
OUTPUT_CSV      = BASE_DIR / "perceptual_features.csv"
VIS_DIR         = BASE_DIR / "visualizations"
MODEL_CACHE_DIR = BASE_DIR / "model_weights"
ERROR_LOG       = BASE_DIR / "processing_errors.log"

MAPILLARY_TOKEN = "YOUR_TOKEN"

AUTOSAVE_EVERY  = 5     # flush CSV to disk every N images
VIS_EVERY       = 5     # save a visualization panel every N images
CONF_THRESHOLD  = 0.3   # YOLO minimum detection confidence

# ── Resolution settings ───────────────────────────────────────────────────
# SegFormer was trained at 1024×1024 but runs well at 512×512.
# The processor internally normalises to (512, 512) anyway when
# size={"height":512,"width":512} is passed, saving ~4× memory & time.
SEG_INPUT_SIZE  = (512, 512)   # (height, width) fed to SegFormer
# YOLO input size is managed by ultralytics (default 640); we resize
# the PIL image to at most 640px on the long side before passing it.
YOLO_MAX_SIDE   = 640


# ===========================================================================
# ─── CLASS / PALETTE / FEATURE DEFINITIONS ───────────────────────────────
# ===========================================================================

CITYSCAPES_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
]

CITYSCAPES_PALETTE = np.array([
    [128,  64, 128], [244,  35, 232], [ 70,  70,  70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170,  30], [220, 220,   0],
    [107, 142,  35], [152, 251, 152], [ 70, 130, 180], [220,  20,  60],
    [255,   0,   0], [  0,   0, 142], [  0,   0,  70], [  0,  60, 100],
    [  0,  80, 100], [  0,   0, 230], [119,  11,  32],
], dtype=np.uint8)

FEATURE_MAPPING = {
    "percept_greenery":            {"seg": ["vegetation", "terrain"],                      "yolo": ["potted plant"]},
    "percept_sky_visibility":      {"seg": ["sky"],                                        "yolo": []},
    "percept_building_frontage":   {"seg": ["building", "wall"],                           "yolo": []},
    "percept_ground_surface":      {"seg": ["road", "sidewalk"],                           "yolo": []},
    "percept_lighting_presence":   {"seg": ["traffic light"],                              "yolo": ["traffic light"]},
    "percept_pedestrian_presence": {"seg": ["person", "rider", "bicycle"],                 "yolo": ["person", "bicycle"]},
    "percept_vehicle_presence":    {"seg": ["car", "truck", "bus", "train", "motorcycle"], "yolo": ["car", "motorcycle", "bus", "truck"]},
    "percept_signage_density":     {"seg": ["traffic sign"],                               "yolo": ["stop sign", "clock"]},
    "percept_street_furniture":    {"seg": ["fence", "pole"],                              "yolo": ["bench", "fire hydrant", "parking meter"]},
}

YOLO_WEIGHT = {
    "percept_greenery":            0.10,
    "percept_sky_visibility":      0.00,
    "percept_building_frontage":   0.00,
    "percept_ground_surface":      0.00,
    "percept_lighting_presence":   0.80,
    "percept_pedestrian_presence": 0.50,
    "percept_vehicle_presence":    0.40,
    "percept_signage_density":     0.90,
    "percept_street_furniture":    0.70,
}

FEATURE_COLORS = {
    "greenery": "#2ecc71", "sky": "#3498db", "building": "#95a5a6",
    "ground": "#e67e22",   "lighting": "#f1c40f", "pedestrian": "#e74c3c",
    "vehicle": "#2980b9",  "signage": "#8e44ad",  "furniture": "#16a085",
}


# ===========================================================================
# ─── DEVICE SELECTION (MPS → CUDA → CPU) ─────────────────────────────────
# ===========================================================================

def get_device() -> torch.device:
    """
    Priority: Apple MPS > CUDA > CPU.
    MPS is the Apple Silicon GPU backend available from PyTorch 1.12+.
    """
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ===========================================================================
# ─── LOGGING ──────────────────────────────────────────────────────────────
# ===========================================================================

def setup_logging(log_path: Path) -> logging.Logger:
    """Logger: INFO to stdout, WARNING+ to file."""
    logger = logging.getLogger("cv_pipeline")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                             datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.WARNING)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


# ===========================================================================
# ─── MODEL DOWNLOAD / CACHE ───────────────────────────────────────────────
# ===========================================================================

def ensure_models(cache_dir: Path, logger: logging.Logger):
    """
    Download SegFormer-B2 (Cityscapes) and YOLOv8n to cache_dir on first run.
    Subsequent runs load from cache — no internet needed.
    Returns (segformer_cache_path, yolo_weights_path).
    """
    import shutil
    cache_dir.mkdir(parents=True, exist_ok=True)
    seg_cache    = cache_dir / "segformer_b2_cityscapes"
    yolo_weights = cache_dir / "yolov8n.pt"

    # SegFormer
    if seg_cache.exists() and any(seg_cache.iterdir()):
        logger.info("SegFormer cached: %s", seg_cache)
    else:
        logger.info("Downloading SegFormer-B2 (Cityscapes) → %s", seg_cache)
        model_id = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
        SegformerImageProcessor.from_pretrained(model_id).save_pretrained(str(seg_cache))
        SegformerForSemanticSegmentation.from_pretrained(model_id).save_pretrained(str(seg_cache))
        logger.info("SegFormer saved.")

    # YOLOv8n
    if yolo_weights.exists():
        logger.info("YOLOv8n cached: %s", yolo_weights)
    else:
        logger.info("Downloading YOLOv8n …")
        tmp = YOLO("yolov8n.pt")
        shutil.copy(str(Path(tmp.ckpt_path)), str(yolo_weights))
        logger.info("YOLOv8n saved: %s", yolo_weights)

    return seg_cache, yolo_weights


# ===========================================================================
# ─── IMAGE HELPERS ────────────────────────────────────────────────────────
# ===========================================================================

def resize_for_segformer(image: Image.Image) -> Image.Image:
    """
    Resize image to SEG_INPUT_SIZE (512×512) using high-quality Lanczos.
    SegFormer's processor will do this internally anyway; doing it here
    in PIL before tensor conversion is faster on large source images.
    """
    h, w = SEG_INPUT_SIZE
    if image.size != (w, h):
        image = image.resize((w, h), Image.LANCZOS)
    return image


def resize_for_yolo(image: Image.Image) -> Image.Image:
    """
    Resize image so its longest side equals YOLO_MAX_SIDE (640px),
    preserving aspect ratio.  Ultralytics would do this anyway, but
    doing it in PIL avoids passing a large tensor into the model.
    """
    w, h = image.size
    max_side = max(w, h)
    if max_side <= YOLO_MAX_SIDE:
        return image
    scale = YOLO_MAX_SIDE / max_side
    new_w, new_h = int(w * scale), int(h * scale)
    return image.resize((new_w, new_h), Image.LANCZOS)


def download_image(image_id: str, thumb_url: str, logger: logging.Logger) -> Image.Image | None:
    """
    Download a Mapillary image.
    1st attempt: thumb_1024_url from registry CSV (fast, no API call needed).
    2nd attempt: Mapillary Graph API (fallback for expired CDN URLs).
    """
    # Attempt 1 – pre-fetched thumbnail URL
    if thumb_url and isinstance(thumb_url, str) and thumb_url.startswith("http"):
        try:
            r = requests.get(thumb_url, timeout=30)
            r.raise_for_status()
            return Image.open(BytesIO(r.content)).convert("RGB")
        except Exception:
            pass

    # Attempt 2 – Mapillary Graph API
    try:
        meta = requests.get(
            f"https://graph.mapillary.com/{image_id}",
            headers={"Authorization": f"Bearer {MAPILLARY_TOKEN}"},
            params={"fields": "thumb_2048_url,thumb_1024_url,thumb_original_url"},
            timeout=10,
        )
        meta.raise_for_status()
        data = meta.json()
        url  = (data.get("thumb_2048_url")
                or data.get("thumb_1024_url")
                or data.get("thumb_original_url"))
        if url:
            r2 = requests.get(url, timeout=30)
            r2.raise_for_status()
            return Image.open(BytesIO(r2.content)).convert("RGB")
    except Exception as exc:
        logger.warning("Download failed [%s]: %s", image_id, exc)
    return None


# ===========================================================================
# ─── PIPELINE CLASS ───────────────────────────────────────────────────────
# ===========================================================================

class EnhancedPerceptualExtractor:
    """
    Hybrid pipeline: SegFormer-B2 (Cityscapes) + YOLOv8n.
    Loaded once; processes images one-by-one via process_one().
    """

    def __init__(self, seg_cache: Path, yolo_weights: Path, logger: logging.Logger):
        self.logger = logger
        self.device = get_device()
        logger.info("Compute device: %s", self.device)

        # SegFormer – load with size override for 512×512 inference
        logger.info("Loading SegFormer-B2 from %s …", seg_cache)
        self.seg_processor = SegformerImageProcessor.from_pretrained(
            str(seg_cache),
            size={"height": SEG_INPUT_SIZE[0], "width": SEG_INPUT_SIZE[1]},
        )
        self.seg_model = (
            SegformerForSemanticSegmentation
            .from_pretrained(str(seg_cache))
            .to(self.device)
            .eval()
        )
        logger.info("SegFormer-B2 ready  (input %dx%d, device=%s)",
                    SEG_INPUT_SIZE[1], SEG_INPUT_SIZE[0], self.device)

        # YOLOv8n
        logger.info("Loading YOLOv8n from %s …", yolo_weights)
        self.det_model = YOLO(str(yolo_weights))
        # Tell YOLO to use MPS/CUDA/CPU matching our device
        self.det_device = str(self.device)
        logger.info("YOLOv8n ready  (device=%s)", self.det_device)

        self.seg_classes = CITYSCAPES_CLASSES

    # ── Segmentation ──────────────────────────────────────────────────────

    def run_segmentation(self, image: Image.Image) -> np.ndarray:
        """
        Returns H×W uint8 mask with Cityscapes class indices.
        Image is pre-resized to SEG_INPUT_SIZE before the processor sees it,
        so the processor only needs to normalise (no resize overhead).
        The mask is upsampled back to the (pre-resized) input size.
        """
        img_resized = resize_for_segformer(image)   # 512×512 PIL image
        inputs = self.seg_processor(images=img_resized, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():                 # slightly faster than no_grad
            logits = self.seg_model(**inputs).logits  # (1, 19, 128, 128)

        # Upsample to 512×512
        upsampled = F.interpolate(
            logits,
            size=(SEG_INPUT_SIZE[0], SEG_INPUT_SIZE[1]),
            mode="bilinear",
            align_corners=False,
        )
        return upsampled.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    # ── Detection ─────────────────────────────────────────────────────────

    def run_detection(self, image: Image.Image) -> list:
        """
        Run YOLOv8n on a down-scaled image (max side 640px).
        Returns list of {class, confidence, bbox} dicts.
        """
        img_small = resize_for_yolo(image)
        results   = self.det_model(
            np.array(img_small),
            verbose=False,
            device=self.det_device,
        )[0]
        return [
            {
                "class":      results.names[int(box.cls[0])],
                "confidence": float(box.conf[0]),
                "bbox":       box.xyxy[0].cpu().numpy(),
            }
            for box in results.boxes
        ]

    # ── Feature fusion ────────────────────────────────────────────────────

    def compute_perceptual_shares(
        self, seg_mask: np.ndarray, detections: list
    ) -> dict:
        """
        Fuse segmentation pixel shares with YOLO detection counts.

        seg_share  = sum(pixels of relevant classes) / total pixels
        det_share  = min(n_detections / 5, 1.0)
        final      = seg_share + (1 - seg_share) * det_share * YOLO_WEIGHT

        YOLO fills the gap left by segmentation without double-counting.
        """
        total = seg_mask.size
        shares = {}
        for feat, src in FEATURE_MAPPING.items():
            seg_share = sum(
                (seg_mask == self.seg_classes.index(c)).sum() / total
                for c in src["seg"] if c in self.seg_classes
            )
            det_count = sum(
                1 for d in detections
                if d["class"] in src["yolo"] and d["confidence"] >= CONF_THRESHOLD
            )
            det_share = min(det_count / 5.0, 1.0)
            w = YOLO_WEIGHT.get(feat, 0.3)
            shares[feat] = float(np.clip(
                seg_share + (1.0 - seg_share) * det_share * w, 0.0, 1.0
            ))
        return shares

    # ── Visualisation ─────────────────────────────────────────────────────

    def build_visualisation(
        self,
        image: Image.Image,
        seg_mask: np.ndarray,
        detections: list,
        shares: dict,
        image_id: str,
        poi_id: str,
        global_index: int,
        save_path: Path,
    ):
        """
        5-panel figure saved as PNG.
        Uses Agg backend (no display required) for speed and headless compatibility.
        """
        fig = plt.figure(figsize=(22, 12))
        gs  = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.30)

        # Panel 1 – original image (display at original download resolution)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(image)
        ax1.set_title("Original Image", fontsize=12, fontweight="bold")
        ax1.axis("off")

        # Panel 2 – segmentation (at 512×512, coloured with Cityscapes palette)
        ax2 = fig.add_subplot(gs[0, 1])
        colored = CITYSCAPES_PALETTE[seg_mask % len(CITYSCAPES_PALETTE)]
        ax2.imshow(colored)
        ax2.set_title("Segmentation  SegFormer-B2 (Cityscapes)", fontsize=12, fontweight="bold")
        ax2.axis("off")
        patches = [
            mpatches.Patch(color=CITYSCAPES_PALETTE[i] / 255.0, label=CITYSCAPES_CLASSES[i])
            for i in np.unique(seg_mask) if i < len(CITYSCAPES_CLASSES)
        ]
        ax2.legend(handles=patches, loc="lower right", fontsize=6, framealpha=0.6, ncol=2)

        # Panel 3 – YOLO bounding boxes (on YOLO-resized image)
        ax3 = fig.add_subplot(gs[0, 2])
        img_draw = resize_for_yolo(image).copy()
        draw = ImageDraw.Draw(img_draw)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
        except Exception:
            font = ImageFont.load_default()
        n_det = 0
        for det in detections:
            if det["confidence"] >= CONF_THRESHOLD:
                x1, y1, x2, y2 = det["bbox"]
                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                draw.text((x1, max(y1 - 15, 0)),
                          f"{det['class']} {det['confidence']:.2f}",
                          fill="red", font=font)
                n_det += 1
        ax3.imshow(img_draw)
        ax3.set_title(f"YOLOv8n  –  {n_det} detections", fontsize=12, fontweight="bold")
        ax3.axis("off")

        # Panel 4 – perceptual shares bar chart
        ax4 = fig.add_subplot(gs[1, :2])
        feat_names = list(shares.keys())
        feat_vals  = [shares[f] * 100 for f in feat_names]
        disp_names = [f.replace("percept_", "").replace("_", " ").title() for f in feat_names]
        bar_colors = [
            next((v for k, v in FEATURE_COLORS.items() if k in f), "#bdc3c7")
            for f in feat_names
        ]
        bars = ax4.barh(disp_names, feat_vals, color=bar_colors, alpha=0.85, height=0.6)
        ax4.set_xlabel("Share / Presence Score (%)", fontsize=11)
        ax4.set_title("Layer 1  Perceptual Shares", fontsize=12, fontweight="bold")
        ax4.set_xlim(0, max(max(feat_vals, default=1) * 1.2, 5))
        ax4.grid(axis="x", alpha=0.3)
        for bar, val in zip(bars, feat_vals):
            if val > 0.3:
                ax4.text(val + 0.4, bar.get_y() + bar.get_height() / 2,
                         f"{val:.1f}%", va="center", fontsize=9)

        # Panel 5 – top detected object classes
        ax5 = fig.add_subplot(gs[1, 2])
        cls_counts = Counter(
            d["class"] for d in detections if d["confidence"] >= CONF_THRESHOLD
        )
        if cls_counts:
            top       = cls_counts.most_common(6)
            cls_names, cls_cnts = zip(*top)
            ax5.barh(range(len(cls_names)), cls_cnts, color="coral", alpha=0.85)
            ax5.set_yticks(range(len(cls_names)))
            ax5.set_yticklabels(cls_names, fontsize=9)
            ax5.set_xlabel("Count", fontsize=10)
            ax5.set_title("Top Detected Objects", fontsize=12, fontweight="bold")
            ax5.grid(axis="x", alpha=0.3)
        else:
            ax5.text(0.5, 0.5, "No detections", ha="center", va="center",
                     transform=ax5.transAxes, fontsize=12)
            ax5.axis("off")

        fig.suptitle(
            f"Image #{global_index}  |  POI: {poi_id}  |  ID: {image_id}",
            fontsize=14, fontweight="bold", y=0.995,
        )
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ── Single-image entry point ───────────────────────────────────────────

    def process_one(
        self,
        image: Image.Image,
        image_id: str,
        poi_id: str,
        global_index: int,
        vis_dir: Path,
    ) -> dict:
        """Full pipeline for one image. Returns perceptual shares dict."""
        seg_mask   = self.run_segmentation(image)
        detections = self.run_detection(image)
        shares     = self.compute_perceptual_shares(seg_mask, detections)

        if global_index % VIS_EVERY == 0:
            vis_path = vis_dir / f"vis_{global_index:05d}_{image_id}.png"
            self.build_visualisation(
                image, seg_mask, detections, shares,
                image_id, poi_id, global_index, vis_path,
            )

        return shares


# ===========================================================================
# ─── CSV HELPERS ──────────────────────────────────────────────────────────
# ===========================================================================

def load_processed_ids(output_csv: Path) -> set:
    """
    Resume logic:
    Only image_ids whose LAST status == 'ok' are skipped.
    Others will be reprocessed.
    """
    if not output_csv.exists():
        return set()

    try:
        df = pd.read_csv(output_csv, dtype=str)

        if "status" not in df.columns:
            return set()

        df = df.dropna(subset=["image_id"])

        # keep last record of each image
        df = df.drop_duplicates(subset="image_id", keep="last")

        ok_df = df[df["status"] == "ok"]

        return set(ok_df["image_id"].tolist())

    except Exception:
        return set()


def flush_buffer(buffer: list, output_csv: Path, logger: logging.Logger):
    """Append buffered result rows to the output CSV and clear the buffer."""
    if not buffer:
        return
    df = pd.DataFrame(buffer)
    write_header = not output_csv.exists()
    df.to_csv(str(output_csv), mode="a", header=write_header, index=False)
    logger.info("  Auto-saved %d rows → %s", len(df), output_csv.name)
    buffer.clear()


# ===========================================================================
# ─── MAIN LOOP ────────────────────────────────────────────────────────────
# ===========================================================================

def run_pipeline(logger: logging.Logger):

    VIS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Models ───────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("STEP 1  Checking / downloading model weights")
    seg_cache, yolo_weights = ensure_models(MODEL_CACHE_DIR, logger)

    # ── 2. Load pipeline ────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("STEP 2  Loading models onto %s", get_device())
    extractor = EnhancedPerceptualExtractor(seg_cache, yolo_weights, logger)

    # ── 3. Registry ─────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("STEP 3  Processing images")
    registry = pd.read_csv(str(REGISTRY_CSV), dtype=str)
    registry.columns = registry.columns.str.strip()
    total = len(registry)
    logger.info("Registry: %d images total", total)

    done_ids = load_processed_ids(OUTPUT_CSV)
    if done_ids:
        logger.info("Resume: %d already done, skipping.", len(done_ids))

    pending = registry[~registry["image_id"].isin(done_ids)].reset_index(drop=True)
    logger.info("Pending: %d images", len(pending))

    # Stable global index map (1-based, consistent across resume runs)
    global_idx_map = {row["image_id"]: i + 1 for i, row in registry.iterrows()}

    buffer: list = []
    processed = 0
    errors    = 0

    # ── Async download setup ─────────────────────────────────────────────
    # One background thread pre-fetches the NEXT image while we process
    # the current one, effectively hiding network latency.
    executor = ThreadPoolExecutor(max_workers=1)

    def submit_download(row) -> Future:
        return executor.submit(
            download_image,
            str(row.get("image_id", "")).strip(),
            str(row.get("thumb_1024_url", "")).strip(),
            logger,
        )

    # Pre-fetch the first image
    rows      = [pending.iloc[i] for i in range(len(pending))]
    next_fut  = submit_download(rows[0]) if rows else None

    t_wall_start = time.time()

    for idx, row in enumerate(rows):
        image_id  = str(row.get("image_id", "")).strip()
        poi_id    = str(row.get("POI_ID",   "")).strip()
        global_i  = global_idx_map.get(image_id, len(done_ids) + processed + 1)

        n_done = len(done_ids) + processed
        logger.info("[%d/%d]  id=%-20s  POI=%s", n_done + 1, total, image_id, poi_id)

        # Collect current image (already downloading in background)
        t0    = time.time()
        image = next_fut.result() if next_fut else None

        # Immediately kick off download of NEXT image (overlaps with inference)
        if idx + 1 < len(rows):
            next_fut = submit_download(rows[idx + 1])
        else:
            next_fut = None

        t_dl = time.time() - t0

        # ── Inference ─────────────────────────────────────────────────
        if image is None:
            logger.warning("  ✗ Download failed, skipping: %s", image_id)
            errors += 1
            buffer.append({
                "POI_ID": poi_id, "image_id": image_id,
                **{f: np.nan for f in FEATURE_MAPPING},
                "status": "download_failed",
            })
        else:
            try:
                t_inf = time.time()
                shares = extractor.process_one(image, image_id, poi_id, global_i, VIS_DIR)
                t_inf  = time.time() - t_inf

                share_str = "  ".join(
                    f"{k.replace('percept_','')[:8]}={v*100:.0f}%"
                    for k, v in shares.items()
                )
                logger.info(
                    "  ✓ dl=%.1fs  inf=%.1fs  total=%.1fs  |  %s",
                    t_dl, t_inf, t_dl + t_inf, share_str,
                )
                buffer.append({
                    "POI_ID": poi_id, "image_id": image_id,
                    **shares,
                    "status": "ok",
                })

            except Exception as exc:
                logger.error("  ✗ Processing error [%s]: %s", image_id, exc)
                logger.debug(traceback.format_exc())
                errors += 1
                buffer.append({
                    "POI_ID": poi_id, "image_id": image_id,
                    **{f: np.nan for f in FEATURE_MAPPING},
                    "status": "processing_error",
                })

        processed += 1

        # Auto-save
        if len(buffer) >= AUTOSAVE_EVERY:
            flush_buffer(buffer, OUTPUT_CSV, logger)

        # ETA estimate
        elapsed = time.time() - t_wall_start
        avg_s   = elapsed / processed
        remain  = (len(rows) - processed) * avg_s
        logger.info(
            "  Progress: %d/%d  |  avg=%.1fs/img  |  ETA ~%.0fmin",
            processed, len(rows), avg_s, remain / 60,
        )

    # ── Final flush ───────────────────────────────────────────────────────
    flush_buffer(buffer, OUTPUT_CSV, logger)
    executor.shutdown(wait=False)

    # ── Summary ───────────────────────────────────────────────────────────
    total_elapsed = time.time() - t_wall_start
    logger.info("=" * 70)
    logger.info("DONE")
    logger.info("  Registry total    : %d", total)
    logger.info("  Already done      : %d", len(done_ids))
    logger.info("  Processed now     : %d", processed)
    logger.info("  Errors            : %d", errors)
    logger.info("  Wall time         : %.1f min", total_elapsed / 60)
    logger.info("  Avg time / image  : %.1f s", total_elapsed / max(processed, 1))
    logger.info("  Output CSV        : %s", OUTPUT_CSV)
    logger.info("  Visualizations    : %s  (%d saved)",
                VIS_DIR, len(list(VIS_DIR.glob("vis_*.png"))))
    logger.info("=" * 70)


# ===========================================================================
# ─── ENTRY POINT ──────────────────────────────────────────────────────────
# ===========================================================================

if __name__ == "__main__":
    logger = setup_logging(ERROR_LOG)
    logger.info(
        "Pipeline starting  |  AUTOSAVE every %d  |  VIS every %d  |  SEG input %dx%d",
        AUTOSAVE_EVERY, VIS_EVERY, SEG_INPUT_SIZE[1], SEG_INPUT_SIZE[0],
    )
    run_pipeline(logger)
