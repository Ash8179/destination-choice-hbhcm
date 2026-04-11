"""
C.2.1(2) – POI Preparation (network-constrained sampling)

This script:
Generates all the sampling points based on subtypes and categories, for POIs without a polygon geometry.

Author: Zhang Wenyu
Date: 2026-01-21
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import math
from shapely.geometry import Point
import os

# -------------------------------------------------------------
# 1. Configuration
# -------------------------------------------------------------
D_PATH = "/Users/zhangwenyu/Desktop/NUSFYP/Stage 1/1.3 Street Level Imagery/D.xlsx"
MISSING_POI_FINAL_PATH = "/Users/zhangwenyu/Desktop/NUSFYP/Stage 1/1.3 Street Level Imagery/missing_poi_id_final.csv"
OUTPUT_CSV = "/Users/zhangwenyu/Desktop/NUSFYP/Stage 1/1.3 Street Level Imagery/sampling_points_missing.csv"

SVY21 = "EPSG:3414"
WGS84 = "EPSG:4326"

AUTOSAVE_INTERVAL = 10  # save every 10 POIs processed

# Sampling radius by subtype (in meters)
RADIUS_BY_SUBTYPE = {
    # Retail
    "mall": 400,
    "lifestyle_street": 250,
    "hawker_centre": 200,
    "hawker centre": 200,  # handle space variant
    
    # Leisure
    "museum": 300,
    "theatre": 300,
    "historic_site": 200,
    "monument": 200,
    "momument": 200,  # handle typo
    "park": 250
}

# Landmark types that require directional perception
LANDMARK_TYPES = {"historic_site", "monument", "momument"}

# Radial sampling configuration
FALLBACK_RADII = [50, 100, 150, 200]  # concentric circles in meters
FALLBACK_POINTS_PER_RING = 8  # points per circle

# -------------------------------------------------------------
# 2. Helper functions
# -------------------------------------------------------------
def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculate bearing from point 1 to point 2"""
    # Validate inputs
    if any(pd.isna([lat1, lon1, lat2, lon2])):
        return None
    
    try:
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        d_lambda = math.radians(lon2 - lon1)

        y = math.sin(d_lambda) * math.cos(phi2)
        x = math.cos(phi1) * math.sin(phi2) - \
            math.sin(phi1) * math.cos(phi2) * math.cos(d_lambda)

        bearing = math.degrees(math.atan2(y, x))
        return (bearing + 360) % 360
    except (ValueError, TypeError):
        return None


def radial_fallback_sampling(center_svy21, radii, points_per_ring):
    """Generate radial sampling points in concentric circles"""
    points = []
    for r in radii:
        for i in range(points_per_ring):
            angle = (360 / points_per_ring) * i
            rad = math.radians(angle)
            x = center_svy21.x + r * math.cos(rad)
            y = center_svy21.y + r * math.sin(rad)
            points.append(Point(x, y))
    return points


def autosave(records, mode="a"):
    """Incrementally save records to CSV"""
    if not records:
        return
    df = pd.DataFrame(records)
    header = not os.path.exists(OUTPUT_CSV) or mode == "w"
    df.to_csv(OUTPUT_CSV, mode=mode, header=header, index=False)


# -------------------------------------------------------------
# 3. Load existing output (auto-resume)
# -------------------------------------------------------------
if os.path.exists(OUTPUT_CSV):
    existing_df = pd.read_csv(OUTPUT_CSV)
    processed_pois = set(existing_df["POI_ID"].unique())
    
    # Extract last point ID number
    point_ids = existing_df["point_id"].str.extract(r"MP_(\d+)", expand=False)
    point_ids = pd.to_numeric(point_ids, errors='coerce')
    last_id = point_ids.max()
    point_counter = int(last_id) if pd.notna(last_id) else 0
    
    print(f"Resuming from existing file:")
    print(f"   - {len(existing_df)} points already saved")
    print(f"   - {len(processed_pois)} POIs already completed")
    print(f"   - Last point_id: MP_{point_counter:07d}\n")
else:
    processed_pois = set()
    point_counter = 0
    existing_df = pd.DataFrame()
    print("No existing output found. Starting fresh.\n")


# -------------------------------------------------------------
# 4. Load data
# -------------------------------------------------------------
print("Loading POI data...")
df_all = pd.read_excel(D_PATH)

missing_ids = set(pd.read_csv(MISSING_POI_FINAL_PATH)["poi_id"])
df_missing = df_all[df_all["poi_id"].isin(missing_ids)].copy()

# Filter out already processed POIs
df_missing = df_missing[~df_missing["poi_id"].isin(processed_pois)].reset_index(drop=True)

if len(df_missing) == 0:
    print("All missing POIs have already been processed!")
    if len(existing_df) > 0:
        print(f"Total points in output: {len(existing_df)}")
    exit(0)

print(f"Remaining POIs to process: {len(df_missing)}")
print(f"   - Already completed: {len(processed_pois)}")

# Show breakdown by subtype
subtype_counts = df_missing['subtype'].value_counts()
print(f"\nBreakdown by subtype:")
for subtype, count in subtype_counts.items():
    print(f"   - {subtype}: {count}")
print()


# -------------------------------------------------------------
# 5. Main processing - Pure Hub-and-Spoke
# -------------------------------------------------------------
records = []
poi_count = 0
success_count = 0
failed_pois = []

for idx, row in df_missing.iterrows():
    poi_id = row["poi_id"]
    category = row["category"]
    subtype = str(row["subtype"]).strip().lower()

    # Centroid coordinates (WGS84)
    centroid_lon = row["centroid_y"]
    centroid_lat = row["centroid_x"]

    # Validate centroid data
    if pd.isna(centroid_lat) or pd.isna(centroid_lon):
        print(f"[{poi_count+1}/{len(df_missing)}]  SKIPPED {poi_id} ({subtype}) - Missing centroid")
        failed_pois.append({'poi_id': poi_id, 'reason': 'missing_centroid'})
        poi_count += 1
        continue
    
    # Convert to float
    try:
        centroid_lon = float(centroid_lon)
        centroid_lat = float(centroid_lat)
    except (ValueError, TypeError):
        print(f"[{poi_count+1}/{len(df_missing)}]  SKIPPED {poi_id} ({subtype}) - Invalid centroid")
        failed_pois.append({'poi_id': poi_id, 'reason': 'invalid_centroid'})
        poi_count += 1
        continue

    # Validate Singapore coordinates
    if not (1.0 <= centroid_lat <= 2.0 and 103.0 <= centroid_lon <= 105.0):
        print(f"[{poi_count+1}/{len(df_missing)}]  SKIPPED {poi_id} ({subtype}) - Out of Singapore range")
        print(f"   Coordinates: lat={centroid_lat}, lon={centroid_lon}")
        failed_pois.append({'poi_id': poi_id, 'reason': 'out_of_range'})
        poi_count += 1
        continue

    print(f"[{poi_count+1}/{len(df_missing)}] Processing {poi_id} ({subtype})")

    # Determine sampling radius
    radius = RADIUS_BY_SUBTYPE.get(subtype, 200)

    # ---------------------------------------------------------
    # Pure Hub-and-Spoke Radial Sampling
    # ---------------------------------------------------------
    try:
        # Create point in WGS84
        center_point_wgs84 = Point(centroid_lon, centroid_lat)
        center_gdf = gpd.GeoDataFrame(
            [{"geometry": center_point_wgs84}],
            crs=WGS84
        )
        
        # Convert to SVY21 for metric buffering
        center_svy21 = center_gdf.to_crs(SVY21).iloc[0]["geometry"]
        
        # Generate radial sampling points
        sampled_pts = radial_fallback_sampling(
            center_svy21,
            FALLBACK_RADII,
            FALLBACK_POINTS_PER_RING
        )
        
        source = "hub_spoke_radial"
        print(f"   ✓ Hub-and-Spoke: {len(sampled_pts)} points (4 rings × 8 directions)")
        
        # Store center for bearing calculation
        centroid_for_bearing_lat = centroid_lat
        centroid_for_bearing_lon = centroid_lon
        
        success_count += 1
        
    except Exception as e:
        print(f"Hub-and-Spoke failed: {e}")
        failed_pois.append({'poi_id': poi_id, 'reason': f'sampling_error: {str(e)}'})
        sampled_pts = []
        source = "failed"
        centroid_for_bearing_lat = centroid_lat
        centroid_for_bearing_lon = centroid_lon

    # ---------------------------------------------------------
    # Save sampled points
    # ---------------------------------------------------------
    for pt in sampled_pts:
        point_counter += 1

        # Convert point to WGS84
        pt_wgs84 = gpd.GeoSeries([pt], crs=SVY21).to_crs(WGS84).iloc[0]

        # Calculate bearing for landmarks
        target_heading = None
        if subtype in LANDMARK_TYPES:
            target_heading = calculate_bearing(
                pt_wgs84.y, pt_wgs84.x,
                centroid_for_bearing_lat, centroid_for_bearing_lon
            )

        records.append({
            "POI_ID": poi_id,
            "point_id": f"MP_{point_counter:07d}",
            "lon": pt_wgs84.x,
            "lat": pt_wgs84.y,
            "target_heading": target_heading,
            "source": source,
            "subtype": subtype
        })

    poi_count += 1

    # Autosave every N POIs
    if poi_count % AUTOSAVE_INTERVAL == 0:
        autosave(records)
        records.clear()
        print(f"Autosaved at POI #{poi_count}\n")


# ---------------------------------------------------------
# 6. Final save and statistics
# ---------------------------------------------------------
autosave(records)  # save remaining records

# Reload complete dataset
final_df = pd.read_csv(OUTPUT_CSV)
before_dedup = len(final_df)

# Deduplicate by coordinates
final_df = final_df.drop_duplicates(subset=["lon", "lat"], keep="first")
final_df.to_csv(OUTPUT_CSV, index=False)

duplicates_removed = before_dedup - len(final_df)

# Calculate new points
if len(existing_df) > 0:
    new_points_this_run = len(final_df) - len(existing_df)
else:
    new_points_this_run = len(final_df)

# -------------------------------------------------------------
# 7. Summary
# -------------------------------------------------------------
print("\n" + "=" * 70)
print("Final Missing POI Sampling Completed!")
print("=" * 70)
print(f"This Run Statistics:")
print(f"   - POIs attempted: {poi_count}")
print(f"   - Successfully sampled: {success_count}")
print(f"   - Failed: {len(failed_pois)}")
print(f"   - New points added: {new_points_this_run}")
print(f"   - Duplicates removed: {duplicates_removed}")

print(f"\nFinal Dataset:")
print(f"   - Total sampling points: {len(final_df)}")
print(f"   - Total POIs with points: {final_df['POI_ID'].nunique()}")

if len(failed_pois) > 0:
    print(f"\n  Failed POIs ({len(failed_pois)}):")
    for fail in failed_pois[:10]:  # Show first 10
        print(f"   - {fail['poi_id']}: {fail['reason']}")
    if len(failed_pois) > 10:
        print(f"   ... and {len(failed_pois) - 10} more")
    
    # Save failed POIs
    fail_df = pd.DataFrame(failed_pois)
    fail_path = OUTPUT_CSV.replace('.csv', '_failed_pois.csv')
    fail_df.to_csv(fail_path, index=False)
    print(f"\n   Saved failed POIs to: {fail_path}")

print(f"\nOutput saved to:")
print(f"   {OUTPUT_CSV}")
print("=" * 70)
