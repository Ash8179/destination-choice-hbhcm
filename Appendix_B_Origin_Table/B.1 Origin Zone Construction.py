"""
B.1 – Origin Zone Construction
URA Master Plan 2019 Subzone Boundary (No Sea)

This script:
1. Loads URA MP2019 Subzone GeoJSON (WGS84)
2. Reprojects geometries to SVY21 / EPSG:3414
3. Computes area (square meters)
4. Computes representative centroids (point-on-surface)
5. Saves processed layers for downstream accessibility and OD analysis

Author: Zhang Wenyu
Date: 2025-12-10
"""

import geopandas as gpd
from pathlib import Path
import os

# -------------------------------------------------------------------
# 0. File paths
# -------------------------------------------------------------------

RAW_DATA_PATH = Path("/Users/zhangwenyu/Desktop/NUSFYP/Stage 1/1.1 Define Origins and Destinations/MasterPlan2019SubzoneBoundaryNoSeaGEOJSON.geojson")
INPUT_DIR = RAW_DATA_PATH.parent
OUTPUT_PATH = INPUT_DIR / "Processed_Output"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------
# 1. Check if file exists before loading
# -------------------------------------------------------------------

print(f"Looking for file at: {RAW_DATA_PATH}")
print(f"File exists: {RAW_DATA_PATH.exists()}")

if not RAW_DATA_PATH.exists():
    print("ERROR: File not found! Please check:")
    print(f"1. File path: {RAW_DATA_PATH}")
    print(f"2. Check if filename is correct: MasterPlan2019SubzoneBoundaryNoSeaGEOJSON.geojson")
    print(f"3. Current working directory: {os.getcwd()}")
    
    # List files in the directory to see what's available
    try:
        files_in_dir = list(INPUT_DIR.glob("*.geojson")) + list(INPUT_DIR.glob("*.json"))
        print(f"\nAvailable GeoJSON/JSON files in directory:")
        for file in files_in_dir:
            print(f"  - {file.name}")
    except:
        print("Could not list directory contents")
    
    exit(1)

# -------------------------------------------------------------------
# 2. Load GeoJSON
# -------------------------------------------------------------------

try:
    subzones = gpd.read_file(RAW_DATA_PATH)
    print(f"Successfully loaded file: {len(subzones)} subzones found")
except Exception as e:
    print(f"Error loading file: {e}")
    exit(1)

# Inspect coordinate reference system (CRS)
print("\nOriginal CRS:", subzones.crs)
print("Columns:", list(subzones.columns))

# Expected: EPSG:4326 (WGS84 – longitude/latitude)
# If CRS is missing, explicitly define it
if subzones.crs is None:
    subzones = subzones.set_crs(epsg=4326)
    print("CRS was missing, set to EPSG:4326")

# -------------------------------------------------------------------
# 3. Reproject to Singapore SVY21 (EPSG:3414)
# -------------------------------------------------------------------

"""
Project to EPSG:3414:
- Official projected CRS for Singapore
- Unit: meters
- Required for correct area, distance, buffering, OD analysis
"""

subzones = subzones.to_crs(epsg=3414)
print("\nReprojected CRS:", subzones.crs)

# -------------------------------------------------------------------
# 4. Geometry-derived attributes
# -------------------------------------------------------------------

# Area in square meters
subzones["area_sqm"] = subzones.geometry.area

# Representative centroid (guaranteed inside polygon)
subzones["centroid_geom"] = subzones.geometry.representative_point()

# Extract centroid coordinates
subzones["centroid_x"] = subzones["centroid_geom"].x
subzones["centroid_y"] = subzones["centroid_geom"].y

# -------------------------------------------------------------------
# 5. Select and rename key attributes (clean schema)
# -------------------------------------------------------------------

# First, let's see what columns we actually have
print("\nAvailable columns in the data:")
for col in subzones.columns:
    print(f"  - {col}")

column_mapping = {}

# Check for common URA column name patterns
if "SUBZONE_NO" in subzones.columns:
    column_mapping["SUBZONE_NO"] = "subzone_no"
elif "SZ" in subzones.columns:
    column_mapping["SZ"] = "subzone_no"
elif "SUBZONE" in subzones.columns:
    column_mapping["SUBZONE"] = "subzone_no"

if "SUBZONE_N" in subzones.columns:
    column_mapping["SUBZONE_N"] = "subzone_name"
elif "SUBZONE_NAME" in subzones.columns:
    column_mapping["SUBZONE_NAME"] = "subzone_name"
elif "NAME" in subzones.columns:
    column_mapping["NAME"] = "subzone_name"

# Select columns to keep (including our computed ones)
keep_columns = list(column_mapping.keys()) + ["area_sqm", "centroid_x", "centroid_y", "geometry"]
subzones_clean = subzones[keep_columns].copy()

# Rename columns
subzones_clean = subzones_clean.rename(columns=column_mapping)

# -------------------------------------------------------------------
# 6. Create centroid GeoDataFrame (for OD / accessibility)
# -------------------------------------------------------------------

centroids = subzones_clean.copy()
centroids = centroids.set_geometry(
    gpd.points_from_xy(
        centroids["centroid_x"],
        centroids["centroid_y"],
        crs=subzones_clean.crs
    )
)

# -------------------------------------------------------------------
# 7. Save processed outputs
# -------------------------------------------------------------------

# Save polygons
polygon_output = OUTPUT_PATH / "subzones.gpkg"
subzones_clean.to_file(polygon_output, layer="subzones", driver="GPKG")
print(f"\nSaved subzone polygons to: {polygon_output}")

# Save centroids
centroid_output = OUTPUT_PATH / "subzone_centroids.gpkg"
centroids.to_file(centroid_output, layer="centroids", driver="GPKG")
print(f"Saved subzone centroids to: {centroid_output}")

# Save as GeoJSON for easy inspection
subzones_clean.to_file(OUTPUT_PATH / "subzones.geojson", driver="GeoJSON")
centroids.to_file(OUTPUT_PATH / "subzone_centroids.geojson", driver="GeoJSON")

print("\nSubzone polygons and centroids successfully processed and saved.")
print(f"Total zones processed: {len(subzones_clean)}")
