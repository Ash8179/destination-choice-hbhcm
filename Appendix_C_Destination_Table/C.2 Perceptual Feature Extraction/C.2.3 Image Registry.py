"""
C.2.3 – Image Registry Construction

This script:
Query Mapillary API for street-level images around sampling points,
with auto-resume, logging, and batching. Constructs an image registry
with metadata and coverage statistics.

Author: Zhang Wenyu
Date: 2026-01-26
"""

import requests
import pandas as pd
import time
import os
from datetime import datetime

# -------------------------------------------------------------
# 1. Configuration
# -------------------------------------------------------------
MAPILLARY_TOKEN = "YOUR_TOKEN"

INPUT_SAMPLING_CSV = "/Users/zhangwenyu/Desktop/NUSFYP/Stage 1/1.3 Street Level Imagery/sampling_points_dedup copy.csv"
OUTPUT_REGISTRY_CSV = "/Users/zhangwenyu/Desktop/NUSFYP/Stage 1/1.3 Street Level Imagery/Image_registry.csv"
PROCESSED_POINTS_LOG = "/Users/zhangwenyu/Desktop/NUSFYP/Stage 1/1.3 Street Level Imagery/processed_points.txt"

SEARCH_RADIUS = 30
MAX_IMAGES = 20
REFERENCE_YEAR = 2026

AUTOSAVE_INTERVAL = 50
DEBUG_FIRST_N = 10

MAPILLARY_ENDPOINT = "https://graph.mapillary.com/images"
METERS_TO_DEGREES = SEARCH_RADIUS / 111000.0

# -------------------------------------------------------------
# 2. Helper: Mapillary API query
# -------------------------------------------------------------
def query_mapillary_images(lon, lat, debug=False):
    bbox_west = lon - METERS_TO_DEGREES
    bbox_south = lat - METERS_TO_DEGREES
    bbox_east = lon + METERS_TO_DEGREES
    bbox_north = lat + METERS_TO_DEGREES
    
    bbox_str = f"{bbox_west},{bbox_south},{bbox_east},{bbox_north}"
    
    params = {
        "access_token": MAPILLARY_TOKEN,
        "fields": (
            "id,"
            "captured_at,"
            "compass_angle,"
            "is_pano,"
            "sequence_id,"
            "geometry"
        ),
        "bbox": bbox_str,
        "limit": MAX_IMAGES
    }

    if debug:
        print(f"  [DEBUG] Querying bbox: {bbox_str}")

    try:
        response = requests.get(MAPILLARY_ENDPOINT, params=params, timeout=15)
        
        if response.status_code != 200:
            if debug:
                print(f"  [API ERROR] Status {response.status_code}: {response.text[:200]}")
            return None

        data = response.json().get("data", [])
        
        if debug:
            print(f"  [DEBUG] API returned {len(data)} images")
        
        return data
        
    except requests.exceptions.Timeout:
        if debug:
            print(f"  [TIMEOUT] Request timed out after 15s")
        return None
    except Exception as e:
        if debug:
            print(f"  [EXCEPTION] {type(e).__name__}: {e}")
        return None

# -------------------------------------------------------------
# 3. Auto-resume support
# -------------------------------------------------------------
processed_points = set()

# Step 1: Load points that have images (from registry CSV)
if os.path.exists(OUTPUT_REGISTRY_CSV):
    existing = pd.read_csv(OUTPUT_REGISTRY_CSV)
    points_with_images = set(existing["point_id"].unique())
    total_existing_records = len(existing)
    print(f"[RESUME] Found existing registry:")
    print(f"[RESUME]   - Total image records: {total_existing_records:,}")
    print(f"[RESUME]   - Points with images: {len(points_with_images):,}")
    processed_points.update(points_with_images)
else:
    points_with_images = set()
    total_existing_records = 0
    print("[START] No existing registry found")

# Step 2: Load ALL attempted points (from log file)
if os.path.exists(PROCESSED_POINTS_LOG):
    with open(PROCESSED_POINTS_LOG, 'r') as f:
        logged_points = set(line.strip() for line in f if line.strip())
    print(f"[RESUME]   - All attempted points (from log): {len(logged_points):,}")
    
    # Calculate points that were tried but had no images
    points_without_images = logged_points - points_with_images
    if points_without_images:
        print(f"[RESUME]   - Points without images (will skip): {len(points_without_images):,}")
    
    # Add ALL logged points to skip set
    processed_points.update(logged_points)
else:
    print("[START] No processed points log found")
    print("[ACTION] Will rebuild log from existing registry...")
    
    # Rebuild log file from registry if it doesn't exist
    if points_with_images:
        with open(PROCESSED_POINTS_LOG, 'w') as f:
            for point_id in points_with_images:
                f.write(f"{point_id}\n")
        print(f"[ACTION] Rebuilt log file with {len(points_with_images):,} points from registry")

print(f"[RESUME] Total points to skip: {len(processed_points):,}")

# -------------------------------------------------------------
# 4. Load sampling points
# -------------------------------------------------------------
print(f"\n[LOAD] Reading sampling points from CSV...")
samples = pd.read_csv(INPUT_SAMPLING_CSV)

total_points = len(samples)
remaining_points = total_points - len(processed_points)

print(f"[LOAD] Total sampling points in CSV: {total_points:,}")
print(f"[LOAD] Already processed: {len(processed_points):,}")
print(f"[LOAD] Remaining to process: {remaining_points:,}")
print(f"[LOAD] Unique POIs: {samples['POI_ID'].nunique():,}")

# Check for required columns
required_cols = ['point_id', 'POI_ID', 'lon', 'lat', 'source']
missing_cols = [col for col in required_cols if col not in samples.columns]
if missing_cols:
    print(f"[ERROR] Missing required columns: {missing_cols}")
    print(f"[ERROR] Available columns: {list(samples.columns)}")
    exit(1)

# Filter out already processed points
samples_to_process = samples[~samples['point_id'].isin(processed_points)].copy()
print(f"[LOAD] Filtered dataset size: {len(samples_to_process):,} points\n")

if len(samples_to_process) == 0:
    print("[COMPLETE] All points have been processed!")
    exit(0)

registry_buffer = []
processed_count = 0

# -------------------------------------------------------------
# 5. Statistics tracking
# -------------------------------------------------------------
empty_image_points = 0
api_error_points = 0
images_collected = 0
points_with_images_this_run = 0

print(f"{'='*70}")
print(f"STARTING IMAGE COLLECTION (radius={SEARCH_RADIUS}m, max_images={MAX_IMAGES})")
print(f"{'='*70}\n")

start_time = time.time()

# -------------------------------------------------------------
# 6. Main loop
# -------------------------------------------------------------
for idx, row in samples_to_process.iterrows():
    point_id = row["point_id"]
    poi_id = row["POI_ID"]
    lon, lat = row["lon"], row["lat"]
    source = row["source"]
    target_heading = row.get("target_heading", None)

    # Show progress
    show_debug = (processed_count < DEBUG_FIRST_N) or (processed_count % 100 == 0)
    
    if show_debug:
        overall_progress = len(processed_points) + processed_count + 1
        print(f"[{processed_count+1}/{remaining_points}] (Overall: {overall_progress}/{total_points}) point_id={point_id}, POI={poi_id}")
        print(f"  Location: ({lon:.6f}, {lat:.6f})")

    # Query Mapillary API
    images = query_mapillary_images(lon, lat, debug=show_debug)

    # Log the point as attempted (whether success, error, or no images)
    with open(PROCESSED_POINTS_LOG, 'a') as f:
        f.write(f"{point_id}\n")

    # Handle API errors
    if images is None:
        api_error_points += 1
        if show_debug:
            print(f"  [SKIP] API error, moving to next point\n")
        processed_count += 1
        continue

    # Handle empty results
    if len(images) == 0:
        empty_image_points += 1
        if show_debug:
            print(f"  [NO IMAGES] No street view images found in {SEARCH_RADIUS}m radius\n")
        processed_count += 1
        continue

    # Process each image
    point_image_count = 0
    for img in images:
        image_id = img.get("id")
        ts = img.get("captured_at")
        
        # Parse timestamp
        if ts:
            try:
                dt_object = datetime.fromtimestamp(ts / 1000.0)
                year = dt_object.year
                image_age = REFERENCE_YEAR - year
                captured_at_str = dt_object.isoformat()
            except (ValueError, OSError) as e:
                print(f"  [WARN] Invalid timestamp {ts} for image {image_id}: {e}")
                year, image_age, captured_at_str = None, None, None
        else:
            year, image_age, captured_at_str = None, None, None

        # Extract geometry
        geom = img.get("geometry", {}).get("coordinates", [None, None])

        # Add to registry buffer
        registry_buffer.append({
            "POI_ID": poi_id,
            "point_id": point_id,
            "image_id": image_id,
            "lon": geom[0],
            "lat": geom[1],
            "captured_at": captured_at_str,
            "image_year": year,
            "image_age": image_age,
            "camera_heading": img.get("compass_angle"),
            "target_heading": target_heading,
            "is_panorama": img.get("is_pano"),
            "sequence_id": img.get("sequence_id"),
            "source": source
        })
        
        point_image_count += 1

    # Update statistics
    images_collected += point_image_count
    points_with_images_this_run += 1
    processed_count += 1

    if show_debug:
        print(f"  [SUCCESS] Collected {point_image_count} images for this point")
        print(f"  [STATS] Total images in this run: {images_collected}\n")

    # Autosave logic - save based on processed_count, not just when there are images
    if processed_count % AUTOSAVE_INTERVAL == 0:
        elapsed = time.time() - start_time
        points_per_sec = processed_count / elapsed if elapsed > 0 else 0
        eta_seconds = (remaining_points - processed_count) / points_per_sec if points_per_sec > 0 else 0
        
        print(f"\n{'='*70}")
        print(f"[CHECKPOINT] New points processed: {processed_count}/{remaining_points} ({processed_count/remaining_points*100:.1f}%)")
        print(f"[CHECKPOINT] Overall progress: {len(processed_points) + processed_count}/{total_points} ({(len(processed_points) + processed_count)/total_points*100:.1f}%)")
        print(f"[STATS] Points with images: {points_with_images_this_run} ({points_with_images_this_run/processed_count*100:.1f}% of new points)")
        print(f"[STATS] Points without images: {empty_image_points} ({empty_image_points/processed_count*100:.1f}%)")
        print(f"[STATS] API errors: {api_error_points}")
        print(f"[STATS] Images collected this run: {images_collected}")
        print(f"[STATS] Total images in registry: {total_existing_records + len(registry_buffer):,}")
        if points_with_images_this_run > 0:
            print(f"[STATS] Avg images per successful point: {images_collected/points_with_images_this_run:.1f}")
        print(f"[SPEED] Processing rate: {points_per_sec:.2f} points/sec")
        print(f"[ETA] Estimated time remaining: {eta_seconds/60:.1f} minutes")
        
        # Save if there are images in buffer
        if registry_buffer:
            df_out = pd.DataFrame(registry_buffer)
            write_header = not os.path.exists(OUTPUT_REGISTRY_CSV)
            df_out.to_csv(
                OUTPUT_REGISTRY_CSV,
                mode="a",
                header=write_header,
                index=False
            )
            saved_count = len(df_out)
            registry_buffer.clear()
            print(f"[SAVE] Registry saved to disk ({saved_count} new records appended)")
        else:
            print(f"[INFO] No new images to save at this checkpoint (all points had no images or errors)")
        
        print(f"{'='*70}\n")

    # Rate limiting
    time.sleep(0.05)

# -------------------------------------------------------------
# 7. Final save
# -------------------------------------------------------------
if registry_buffer:
    print(f"\n[SAVE] Saving final batch of {len(registry_buffer)} records...")
    df_out = pd.DataFrame(registry_buffer)
    write_header = not os.path.exists(OUTPUT_REGISTRY_CSV)
    df_out.to_csv(
        OUTPUT_REGISTRY_CSV,
        mode="a",
        header=write_header,
        index=False
    )
    print(f"[SAVE] Final batch saved successfully")

# -------------------------------------------------------------
# 8. Final summary
# -------------------------------------------------------------
total_elapsed = time.time() - start_time

# Reload to get final count
if os.path.exists(OUTPUT_REGISTRY_CSV):
    final_registry = pd.read_csv(OUTPUT_REGISTRY_CSV)
    final_total_records = len(final_registry)
    final_unique_points = final_registry['point_id'].nunique()
else:
    final_total_records = 0
    final_unique_points = 0

print(f"\n{'='*70}")
print("IMAGE REGISTRY CONSTRUCTION COMPLETED")
print(f"{'='*70}")
print(f"Output file: {OUTPUT_REGISTRY_CSV}")
print(f"\nTHIS RUN STATISTICS:")
print(f"  New points processed: {processed_count:,}")
print(f"  Points with images: {points_with_images_this_run:,} ({points_with_images_this_run/processed_count*100:.1f}%)" if processed_count > 0 else "  Points with images: 0")
print(f"  Points without images: {empty_image_points:,} ({empty_image_points/processed_count*100:.1f}%)" if processed_count > 0 else "  Points without images: 0")
print(f"  API errors: {api_error_points:,}")
print(f"  Images collected this run: {images_collected:,}")
if points_with_images_this_run > 0:
    print(f"  Average images per successful point: {images_collected/points_with_images_this_run:.2f}")

print(f"\nOVERALL REGISTRY STATISTICS:")
print(f"  Total unique points: {final_unique_points:,} / {total_points:,} ({final_unique_points/total_points*100:.1f}%)")
print(f"  Total image records: {final_total_records:,}")
print(f"  Remaining points: {total_points - final_unique_points:,}")

print(f"\nPERFORMANCE:")
print(f"  Time for this run: {total_elapsed/60:.1f} minutes")
if processed_count > 0:
    print(f"  Processing rate: {processed_count/total_elapsed:.2f} points/second")
print(f"{'='*70}\n")

# Data quality check
if processed_count > 0:
    coverage_rate = points_with_images_this_run / processed_count
    if coverage_rate < 0.3:
        print("[WARNING] Low coverage rate (<30%). Possible issues:")
        print("  - Sampling points may be in areas with poor Mapillary coverage")
        print("  - Consider increasing SEARCH_RADIUS")
        print("  - Consider alternative data sources (Google Street View)")
    elif coverage_rate < 0.6:
        print("[CAUTION] Moderate coverage rate. Consider:")
        print("  - Verifying coverage by POI type")
        print("  - Hybrid approach for low-coverage areas")
    else:
        print("[SUCCESS] Good coverage rate! Data quality looks promising.")
