"""
C.2.6.2 - Enhanced Computer Vision Pipeline - Additional Features Extraction (COMPLETE)
==============================================================================
Architecture : SegFormer-B2 (Cityscapes, 19 classes) + YOLOv8n
Hardware     : Apple Silicon M3 Pro
Input        : perceptual_features.csv (existing 9 features)
Output       : perceptual_features_complete.csv (9 + 3 new features)
               visualizations_additional/ (one PNG every 5 images)

NEW FEATURES COMPUTED
----------------------
1. percept_architectural_variety    - Building diversity (color entropy + edge complexity)
2. percept_activity_diversity       - Activity type richness (YOLO detection entropy)
3. percept_shading_coverage         - Shadow coverage (luminance analysis)

Author  : Zhang Wenyu
Updated : 2026-03-11
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
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import cv2
import torch
from PIL import Image
from ultralytics import YOLO

warnings.filterwarnings("ignore")

# ===========================================================================
# ─── USER CONFIGURATION ───────────────────────────────────────────────────
# ===========================================================================
BASE_DIR = Path(
    "/Users/zhangwenyu/Desktop/NUSFYP/Stage 1/1.3 Street Level Imagery"
)

# Input: existing features CSV (with 9 features)
INPUT_CSV = BASE_DIR / "perceptual_features.csv"

# Output: complete features CSV (12 features)
OUTPUT_CSV = BASE_DIR / "perceptual_features_complete.csv"

# New visualization directory
VIS_DIR = BASE_DIR / "visualizations_additional"

# Model cache
MODEL_CACHE_DIR = BASE_DIR / "model_weights"

ERROR_LOG = BASE_DIR / "processing_errors_additional.log"

MAPILLARY_TOKEN = "YOUR_TOKEN"

AUTOSAVE_EVERY = 5      # flush CSV every 5 images
VIS_EVERY = 5           # save visualization every 5 images
CONF_THRESHOLD = 0.3    # YOLO confidence threshold

# YOLO input resolution
YOLO_MAX_SIDE = 640

# ===========================================================================
# ─── FEATURE COLORS FOR VISUALIZATION ────────────────────────────────────
# ===========================================================================
FEATURE_COLORS = {
    # Original 9 features
    "greenery": "#2ecc71", "sky": "#3498db", "building": "#95a5a6",
    "ground": "#e67e22", "lighting": "#f1c40f", "pedestrian": "#e74c3c",
    "vehicle": "#2980b9", "signage": "#8e44ad", "furniture": "#16a085",
    
    # New 3 features
    "architectural": "#d35400",  # dark orange
    "activity": "#9b59b6",       # purple
    "shading": "#27ae60",        # dark green
}

# ===========================================================================
# ─── DEVICE SELECTION ─────────────────────────────────────────────────────
# ===========================================================================
def get_device() -> torch.device:
    """Priority: Apple MPS > CUDA > CPU."""
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
    logger = logging.getLogger("cv_additional")
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
# ─── MODEL CACHE ──────────────────────────────────────────────────────────
# ===========================================================================
def ensure_yolo_model(cache_dir: Path, logger: logging.Logger) -> Path:
    """Download YOLOv8n to cache on first run."""
    import shutil
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    yolo_weights = cache_dir / "yolov8n.pt"
    
    if yolo_weights.exists():
        logger.info("YOLOv8n cached: %s", yolo_weights)
    else:
        logger.info("Downloading YOLOv8n ...")
        tmp = YOLO("yolov8n.pt")
        shutil.copy(str(Path(tmp.ckpt_path)), str(yolo_weights))
        logger.info("YOLOv8n saved: %s", yolo_weights)
    
    return yolo_weights

# ===========================================================================
# ─── IMAGE HELPERS ────────────────────────────────────────────────────────
# ===========================================================================
def resize_for_yolo(image: Image.Image) -> Image.Image:
    """Resize image so longest side = YOLO_MAX_SIDE (640px)."""
    w, h = image.size
    max_side = max(w, h)
    if max_side <= YOLO_MAX_SIDE:
        return image
    scale = YOLO_MAX_SIDE / max_side
    new_w, new_h = int(w * scale), int(h * scale)
    return image.resize((new_w, new_h), Image.LANCZOS)

def download_image(image_id: str, thumb_url: str, logger: logging.Logger) -> Image.Image | None:
    """Download Mapillary image with fallback strategy."""
    # Attempt 1: pre-fetched thumbnail URL
    if thumb_url and isinstance(thumb_url, str) and thumb_url.startswith("http"):
        try:
            r = requests.get(thumb_url, timeout=30)
            r.raise_for_status()
            return Image.open(BytesIO(r.content)).convert("RGB")
        except Exception:
            pass
    
    # Attempt 2: Mapillary Graph API
    try:
        meta = requests.get(
            f"https://graph.mapillary.com/{image_id}",
            headers={"Authorization": f"Bearer {MAPILLARY_TOKEN}"},
            params={"fields": "thumb_2048_url,thumb_1024_url"},
            timeout=10,
        )
        meta.raise_for_status()
        data = meta.json()
        url = data.get("thumb_2048_url") or data.get("thumb_1024_url")
        if url:
            r2 = requests.get(url, timeout=30)
            r2.raise_for_status()
            return Image.open(BytesIO(r2.content)).convert("RGB")
    except Exception as exc:
        logger.warning("Download failed [%s]: %s", image_id, exc)
    
    return None

# ===========================================================================
# ─── NEW FEATURE COMPUTATION ──────────────────────────────────────────────
# ===========================================================================

class AdditionalFeatureExtractor:
    """
    Computes 3 new perceptual features from raw images.
    Includes YOLO re-inference for activity diversity.
    """
    
    def __init__(self, yolo_weights: Path, logger: logging.Logger):
        self.logger = logger
        self.device = get_device()
        
        # Load YOLOv8n
        logger.info("Loading YOLOv8n from %s ...", yolo_weights)
        self.yolo_model = YOLO(str(yolo_weights))
        self.yolo_device = str(self.device)
        logger.info("YOLOv8n ready (device=%s)", self.yolo_device)
    
    # ── Feature 1: Architectural Variety ──────────────────────────────────
    def compute_architectural_variety(self, image: np.ndarray) -> float:
        """
        Architectural Variety = 0.6 * color_entropy + 0.4 * edge_density
        
        Theory: Lynch (1960) - Visual diversity increases imageability
        
        Parameters:
        -----------
        image : np.ndarray
            RGB image as numpy array (H, W, 3)
            
        Returns:
        --------
        float : Architectural variety score [0, 1]
        """
        try:
            h, w = image.shape[:2]
            
            # 1. Color diversity (HSV hue entropy)
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            hue_channel = hsv[:, :, 0]
            
            # Histogram of hue values (18 bins)
            hue_hist, _ = np.histogram(hue_channel, bins=18, range=(0, 180))
            hue_hist = hue_hist / (hue_hist.sum() + 1e-10)
            
            # Shannon entropy
            color_entropy = -np.sum(hue_hist * np.log(hue_hist + 1e-10))
            color_entropy_norm = color_entropy / np.log(18)  # normalize to [0, 1]
            
            # 2. Texture complexity (edge density)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = edges.sum() / (h * w * 255)  # normalize by max possible
            edge_density_norm = np.clip(edge_density / 0.3, 0, 1)  # 0.3 = high density
            
            # 3. Combine
            variety = 0.6 * color_entropy_norm + 0.4 * edge_density_norm
            
            return float(np.clip(variety, 0, 1))
        
        except Exception as e:
            self.logger.warning("Architectural variety computation failed: %s", e)
            return 0.0
    
    # ── Feature 2: Activity Diversity ─────────────────────────────────────
    def run_yolo_detection(self, image: Image.Image) -> list:
        """
        Run YOLOv8n object detection.
        
        Returns:
        --------
        list : List of detection dicts {class, confidence, bbox}
        """
        img_resized = resize_for_yolo(image)
        img_array = np.array(img_resized)
        
        results = self.yolo_model(
            img_array,
            verbose=False,
            device=self.yolo_device,
        )[0]
        
        return [
            {
                "class": results.names[int(box.cls[0])],
                "confidence": float(box.conf[0]),
                "bbox": box.xyxy[0].cpu().numpy(),
            }
            for box in results.boxes
        ]
    
    def compute_activity_diversity(self, detections: list) -> float:
        """
        Activity Diversity = Shannon entropy of activity category distribution
        
        Theory: Jacobs (1961) - Mixed uses create vitality
        
        Parameters:
        -----------
        detections : list
            List of YOLO detection dicts
            
        Returns:
        --------
        float : Activity diversity score [0, 1]
        """
        # Activity category mapping (YOLO COCO 80 classes → 4 activity types)
        activity_mapping = {
            # Transport activity
            'bicycle': 'transport',
            'car': 'transport',
            'motorcycle': 'transport',
            'bus': 'transport',
            'truck': 'transport',
            
            # Leisure activity
            'person': 'leisure',
            'bench': 'leisure',
            'potted plant': 'leisure',
            'sports ball': 'leisure',
            'kite': 'leisure',
            'skateboard': 'leisure',
            'surfboard': 'leisure',
            
            # Commercial activity
            'stop sign': 'commercial',
            'traffic light': 'commercial',
            'parking meter': 'commercial',
            'fire hydrant': 'commercial',
            
            # Dining activity
            'dining table': 'dining',
            'chair': 'dining',
            'bottle': 'dining',
            'cup': 'dining',
            'fork': 'dining',
            'knife': 'dining',
            'spoon': 'dining',
            'bowl': 'dining',
        }
        
        # Count detections by activity category
        category_counts = {}
        
        for det in detections:
            if det['confidence'] >= CONF_THRESHOLD:
                obj_class = det['class']
                if obj_class in activity_mapping:
                    category = activity_mapping[obj_class]
                    category_counts[category] = category_counts.get(category, 0) + 1
        
        # If no detections, return 0
        if not category_counts:
            return 0.0
        
        # Calculate Shannon entropy
        total = sum(category_counts.values())
        probs = [count / total for count in category_counts.values()]
        entropy = -sum(p * np.log(p) for p in probs if p > 0)
        
        # Normalize by maximum entropy (log of number of categories)
        max_entropy = np.log(4)  # 4 activity categories
        diversity = entropy / max_entropy
        
        return float(np.clip(diversity, 0, 1))
    
    # ── Feature 3: Shading Coverage ───────────────────────────────────────
    def compute_shading_coverage(self, image: np.ndarray) -> float:
        """
        Shading Coverage = 0.7 * shadow_ratio + 0.3 * dark_ratio
        
        Theory: Mehta (2014) - Shade critical in tropical climates
        
        Parameters:
        -----------
        image : np.ndarray
            RGB image as numpy array
            
        Returns:
        --------
        float : Shading coverage score [0, 1]
        """
        try:
            # Convert to HSV for luminance analysis
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            value_channel = hsv[:, :, 2]  # V channel (brightness)
            
            # Shadow detection: pixels with very low brightness
            shadow_mask = value_channel < 80  # threshold for deep shadows
            shadow_ratio = shadow_mask.sum() / shadow_mask.size
            
            # Dark areas: broader threshold for shaded regions
            dark_mask = value_channel < 120
            dark_ratio = dark_mask.sum() / dark_mask.size
            
            # Combine: prioritize actual shadows, supplement with dark areas
            shading = 0.7 * shadow_ratio + 0.3 * dark_ratio
            
            return float(np.clip(shading, 0, 1))
        
        except Exception as e:
            self.logger.warning("Shading coverage computation failed: %s", e)
            return 0.0
    
    # ── Main Processing ───────────────────────────────────────────────────
    def process_one(self, image: Image.Image) -> dict:
        """
        Compute all 3 new features for one image.
        
        Returns:
        --------
        dict : {feature_name: value, 'yolo_detections': count}
        """
        img_array = np.array(image)
        
        # Run YOLO detection (needed for activity diversity)
        detections = self.run_yolo_detection(image)
        
        # Compute features
        features = {
            "percept_architectural_variety": self.compute_architectural_variety(img_array),
            "percept_activity_diversity": self.compute_activity_diversity(detections),
            "percept_shading_coverage": self.compute_shading_coverage(img_array),
            "n_yolo_detections": len([d for d in detections if d['confidence'] >= CONF_THRESHOLD]),
        }
        
        return features

# ===========================================================================
# ─── VISUALIZATION ────────────────────────────────────────────────────────
# ===========================================================================

def build_visualization(
    image: Image.Image,
    all_features: dict,
    image_id: str,
    poi_id: str,
    global_index: int,
    save_path: Path,
):
    """
    Create 2-panel visualization:
    1. Original image
    2. Bar chart of all 12 perceptual features (9 old + 3 new)
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Panel 1: Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image", fontsize=14, fontweight="bold")
    axes[0].axis("off")
    
    # Panel 2: All 12 features bar chart
    # Filter only perceptual features (exclude metadata like n_yolo_detections)
    feature_items = [(k, v) for k, v in all_features.items()
                     if k.startswith("percept_") and pd.notna(v)]
    
    if not feature_items:
        axes[1].text(0.5, 0.5, "No features available",
                    ha="center", va="center", fontsize=14)
        axes[1].axis("off")
    else:
        feature_names = [k for k, v in feature_items]
        feature_values = [float(v) * 100 for k, v in feature_items]
        
        # Shorten display names
        display_names = [
            f.replace("percept_", "").replace("_", " ").title()
            for f in feature_names
        ]
        
        # Color bars (highlight new 3 features)
        bar_colors = []
        for fname in feature_names:
            if "architectural" in fname:
                bar_colors.append(FEATURE_COLORS["architectural"])
            elif "activity" in fname:
                bar_colors.append(FEATURE_COLORS["activity"])
            elif "shading" in fname:
                bar_colors.append(FEATURE_COLORS["shading"])
            else:
                # Original features - use generic mapping
                for key in FEATURE_COLORS:
                    if key in fname:
                        bar_colors.append(FEATURE_COLORS[key])
                        break
                else:
                    bar_colors.append("#bdc3c7")
        
        bars = axes[1].barh(display_names, feature_values, color=bar_colors,
                           alpha=0.85, height=0.6)
        axes[1].set_xlabel("Share / Score (%)", fontsize=12, fontweight="bold")
        axes[1].set_title("All 12 Perceptual Features (9 Original + 3 New)",
                         fontsize=14, fontweight="bold")
        axes[1].set_xlim(0, max(max(feature_values, default=1) * 1.2, 5))
        axes[1].grid(axis="x", alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, feature_values):
            if val > 0.5:
                axes[1].text(val + 1, bar.get_y() + bar.get_height() / 2,
                            f"{val:.1f}%", va="center", fontsize=9)
        
        # Highlight new features with asterisk
        for i, fname in enumerate(feature_names):
            if any(x in fname for x in ["architectural", "activity", "shading"]):
                axes[1].text(-2, i, "★", fontsize=12, color="red",
                            ha="right", va="center", fontweight="bold")
    
    # Add metadata in title
    n_det = all_features.get("n_yolo_detections", 0)
    fig.suptitle(
        f"Image #{global_index}  |  POI: {poi_id}  |  ID: {image_id}  |  YOLO: {n_det} objects",
        fontsize=14, fontweight="bold", y=0.98
    )
    
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

# ===========================================================================
# ─── CSV HELPERS ──────────────────────────────────────────────────────────
# ===========================================================================

def load_processed_ids(output_csv: Path) -> set:
    """Return set of image_ids that already have all 12 features."""
    if not output_csv.exists():
        return set()
    
    try:
        df = pd.read_csv(output_csv, dtype=str)
        required_cols = [
            "percept_architectural_variety",
            "percept_activity_diversity",
            "percept_shading_coverage",
        ]
        
        # Only skip if all 3 new features exist and are not NaN
        if all(col in df.columns for col in required_cols):
            df = df.dropna(subset=["image_id"])
            complete = df.dropna(subset=required_cols)
            return set(complete["image_id"].tolist())
        
        return set()
    
    except Exception:
        return set()

def flush_buffer(buffer: list, output_csv: Path, logger: logging.Logger):
    """Append buffered rows to CSV."""
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
    
    # ── 1. Ensure YOLOv8n model ────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("STEP 1  Checking YOLOv8n model")
    yolo_weights = ensure_yolo_model(MODEL_CACHE_DIR, logger)
    
    # ── 2. Load existing features ──────────────────────────────────────
    logger.info("=" * 70)
    logger.info("STEP 2  Loading existing features from %s", INPUT_CSV.name)
    
    if not INPUT_CSV.exists():
        logger.error("Input CSV not found: %s", INPUT_CSV)
        return
    
    existing_df = pd.read_csv(str(INPUT_CSV))
    logger.info("Loaded %d rows with existing features", len(existing_df))
    
    # Check for required columns
    required = ["POI_ID", "image_id"]
    if not all(col in existing_df.columns for col in required):
        logger.error("Missing required columns: %s", required)
        return
    
    # ── 3. Initialize feature extractor ────────────────────────────────
    logger.info("=" * 70)
    logger.info("STEP 3  Initializing additional feature extractor")
    extractor = AdditionalFeatureExtractor(yolo_weights, logger)
    
    # ── 4. Determine pending images ────────────────────────────────────
    logger.info("=" * 70)
    logger.info("STEP 4  Processing images")
    
    done_ids = load_processed_ids(OUTPUT_CSV)
    if done_ids:
        logger.info("Resume: %d already complete, skipping.", len(done_ids))
    
    # Convert image_id to string for comparison
    existing_df['image_id'] = existing_df['image_id'].astype(str)
    
    pending = existing_df[~existing_df["image_id"].isin(done_ids)].reset_index(drop=True)
    logger.info("Pending: %d images", len(pending))
    
    if len(pending) == 0:
        logger.info("All images already processed!")
        return
    
    # Global index map
    global_idx_map = {str(row["image_id"]): i + 1
                     for i, row in existing_df.iterrows()}
    
    buffer = []
    processed = 0
    errors = 0
    
    # ── Async download setup ───────────────────────────────────────────
    executor = ThreadPoolExecutor(max_workers=1)
    
    def submit_download(row) -> Future:
        return executor.submit(
            download_image,
            str(row.get("image_id", "")).strip(),
            str(row.get("thumb_1024_url", "")).strip() if "thumb_1024_url" in row else "",
            logger,
        )
    
    rows = [pending.iloc[i] for i in range(len(pending))]
    next_fut = submit_download(rows[0]) if rows else None
    
    t_wall_start = time.time()
    
    for idx, row in enumerate(rows):
        image_id = str(row.get("image_id", "")).strip()
        poi_id = str(row.get("POI_ID", "")).strip()
        global_i = global_idx_map.get(image_id, len(done_ids) + processed + 1)
        
        n_done = len(done_ids) + processed
        logger.info("[%d/%d]  id=%-20s  POI=%s",
                   n_done + 1, len(existing_df), image_id, poi_id)
        
        # Download current image
        t0 = time.time()
        image = next_fut.result() if next_fut else None
        
        # Start downloading next image
        if idx + 1 < len(rows):
            next_fut = submit_download(rows[idx + 1])
        else:
            next_fut = None
        
        t_dl = time.time() - t0
        
        # ── Process image ──────────────────────────────────────────────
        if image is None:
            logger.warning("  ✗ Download failed: %s", image_id)
            errors += 1
            
            # Preserve existing features, add NaN for new 3
            result_row = row.to_dict()
            result_row.update({
                "percept_architectural_variety": np.nan,
                "percept_activity_diversity": np.nan,
                "percept_shading_coverage": np.nan,
                "n_yolo_detections": np.nan,
                "status": "download_failed",
            })
            buffer.append(result_row)
        
        else:
            try:
                t_proc = time.time()
                
                # Compute 3 new features (includes YOLO re-run)
                new_features = extractor.process_one(image)
                
                t_proc = time.time() - t_proc
                
                # Merge with existing features
                result_row = row.to_dict()
                result_row.update(new_features)
                result_row["status"] = "ok"
                
                buffer.append(result_row)
                
                # Build visualization every VIS_EVERY images
                if global_i % VIS_EVERY == 0:
                    # Combine all features for visualization
                    all_features = {
                        k: v for k, v in result_row.items()
                        if k.startswith("percept_") or k == "n_yolo_detections"
                    }
                    
                    vis_path = VIS_DIR / f"vis_{global_i:05d}_{image_id}.png"
                    build_visualization(
                        image, all_features, image_id, poi_id, global_i, vis_path
                    )
                
                # Log feature values
                feat_str = "  ".join(
                    f"{k.replace('percept_','')[:10]}={v*100:.0f}%"
                    for k, v in new_features.items()
                    if k.startswith("percept_")
                )
                logger.info(
                    "  ✓ dl=%.1fs  proc=%.1fs (YOLO+feat)  |  %s",
                    t_dl, t_proc, feat_str
                )
            
            except Exception as exc:
                logger.error("  ✗ Processing error [%s]: %s", image_id, exc)
                logger.debug(traceback.format_exc())
                errors += 1
                
                result_row = row.to_dict()
                result_row.update({
                    "percept_architectural_variety": np.nan,
                    "percept_activity_diversity": np.nan,
                    "percept_shading_coverage": np.nan,
                    "n_yolo_detections": np.nan,
                    "status": "processing_error",
                })
                buffer.append(result_row)
        
        processed += 1
        
        # Auto-save
        if len(buffer) >= AUTOSAVE_EVERY:
            flush_buffer(buffer, OUTPUT_CSV, logger)
        
        # ETA
        elapsed = time.time() - t_wall_start
        avg_s = elapsed / processed
        remain = (len(rows) - processed) * avg_s
        logger.info(
            "  Progress: %d/%d  |  avg=%.1fs/img  |  ETA ~%.0fmin",
            processed, len(rows), avg_s, remain / 60
        )
    
    # ── Final flush ────────────────────────────────────────────────────
    flush_buffer(buffer, OUTPUT_CSV, logger)
    executor.shutdown(wait=False)
    
    # ── Summary ────────────────────────────────────────────────────────
    total_elapsed = time.time() - t_wall_start
    logger.info("=" * 70)
    logger.info("DONE - ADDITIONAL FEATURES EXTRACTION")
    logger.info("=" * 70)
    logger.info("  Total images      : %d", len(existing_df))
    logger.info("  Already complete  : %d", len(done_ids))
    logger.info("  Processed now     : %d", processed)
    logger.info("  Errors            : %d", errors)
    logger.info("  Wall time         : %.1f min", total_elapsed / 60)
    logger.info("  Avg time / image  : %.1f s", total_elapsed / max(processed, 1))
    logger.info("  Output CSV        : %s", OUTPUT_CSV)
    logger.info("  Visualizations    : %s  (%d saved)",
               VIS_DIR, len(list(VIS_DIR.glob("vis_*.png"))))
    logger.info("=" * 70)
    
    # ── Verification ───────────────────────────────────────────────────
    if OUTPUT_CSV.exists():
        final_df = pd.read_csv(OUTPUT_CSV)
        logger.info("\nFINAL DATASET SUMMARY:")
        logger.info("  Total rows: %d", len(final_df))
        logger.info("  Complete (12 features): %d",
                   len(final_df.dropna(subset=[
                       "percept_architectural_variety",
                       "percept_activity_diversity",
                       "percept_shading_coverage"
                   ])))
        logger.info("\n  Feature completeness:")
        for feat in ["percept_architectural_variety",
                    "percept_activity_diversity",
                    "percept_shading_coverage"]:
            if feat in final_df.columns:
                non_null = final_df[feat].notna().sum()
                logger.info("    %s: %d / %d (%.1f%%)",
                          feat, non_null, len(final_df),
                          non_null / len(final_df) * 100)

# ===========================================================================
# ─── ENTRY POINT ──────────────────────────────────────────────────────────
# ===========================================================================

if __name__ == "__main__":
    logger = setup_logging(ERROR_LOG)
    logger.info(
        "Additional Features Pipeline (COMPLETE with YOLO)  |  "
        "AUTOSAVE every %d  |  VIS every %d",
        AUTOSAVE_EVERY, VIS_EVERY
    )
    run_pipeline(logger)
