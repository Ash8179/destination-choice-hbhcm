"""
C.2.1 – POI Preparation (network-constrained sampling)

This script:
Generates all the sampling points based on subtypes and categories.

Author: Zhang Wenyu
Date: 2026-01-21
"""

# geom_polygon is assumed to be in SVY21 (EPSG:3414)
# centroid_x / centroid_y are WGS84 and used only for bearing calculation

import pandas as pd
import geopandas as gpd
import osmnx as ox
import numpy as np
import math
import os
from shapely import wkt

# -------------------------------------------------------------
# 1. Configuration & Paths
# -------------------------------------------------------------
INPUT_PATH = "/Users/zhangwenyu/Desktop/NUSFYP/Stage 1/1.3 Street Level Imagery/D.xlsx"
SAMPLING_OUTPUT_CSV = "/Users/zhangwenyu/Desktop/NUSFYP/Sampling_points_v2.csv"

SVY21 = "EPSG:3414"
WGS84 = "EPSG:4326"

AUTOSAVE_INTERVAL = 2  # write to disk every 2 new points

# -------------------------------------------------------------
# 2. Helper Functions
# -------------------------------------------------------------
def calculate_bearing(lat1, lon1, lat2, lon2):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_lambda = math.radians(lon2 - lon1)

    y = math.sin(delta_lambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - \
        math.sin(phi1) * math.cos(phi2) * math.cos(delta_lambda)

    bearing = math.degrees(math.atan2(y, x))
    return (bearing + 360) % 360


def get_network_samples(polygon_svy21, spacing):
    try:
        polygon_wgs84 = gpd.GeoSeries(
            [polygon_svy21], crs=SVY21
        ).to_crs(WGS84).iloc[0]

        graph = ox.graph_from_polygon(
            polygon_wgs84,
            network_type="walk",
            retain_all=True
        )

        graph_proj = ox.project_graph(graph, to_crs=SVY21)
        _, edges = ox.graph_to_gdfs(graph_proj)

        points = []
        for _, edge in edges.iterrows():
            line = edge.geometry
            if line.length < 5:
                continue

            for d in np.arange(0, line.length, spacing):
                points.append(line.interpolate(d))

        return points

    except Exception:
        return []


def safe_wkt_loads(val):
    if isinstance(val, str):
        try:
            return wkt.loads(val)
        except Exception:
            return None
    return None


def autosave(records, mode="a"):
    if not records:
        return
    df = pd.DataFrame(records)
    header = not os.path.exists(SAMPLING_OUTPUT_CSV) or mode == "w"
    df.to_csv(SAMPLING_OUTPUT_CSV, mode=mode, header=header, index=False)


# -------------------------------------------------------------
# 3. Load existing output (auto-resume)
# -------------------------------------------------------------
if os.path.exists(SAMPLING_OUTPUT_CSV):
    existing_df = pd.read_csv(SAMPLING_OUTPUT_CSV)
    processed_pois = set(existing_df["POI_ID"].unique())
    last_id = existing_df["point_id"].str.extract(r"P_(\d+)").astype(int).max()[0]
    point_counter = int(last_id)
    print(f"Resuming from existing file: {point_counter} points, "
          f"{len(processed_pois)} POIs completed")
else:
    processed_pois = set()
    point_counter = 0
    print("No existing output found. Starting fresh.")

# -------------------------------------------------------------
# 4. Load and prepare POIs
# -------------------------------------------------------------
df = pd.read_excel(INPUT_PATH)
df["geometry"] = df["geom_polygon"].apply(safe_wkt_loads)
df = df[df["geometry"].notnull()].copy()

gdf_poi = gpd.GeoDataFrame(df, geometry="geometry", crs=SVY21)

# -------------------------------------------------------------
# 5. Main Processing Loop
# -------------------------------------------------------------
sampling_buffer = []

for _, row in gdf_poi.iterrows():
    poi_id = row["poi_id"]

    if poi_id in processed_pois:
        continue

    subtype = row["subtype"]
    geom = row["geometry"]
    centroid_lat = row["centroid_y"]
    centroid_lon = row["centroid_x"]

    print(f"Processing {poi_id} ({subtype})")

    if subtype in ["historic_site", "monument"]:
        sampling_zone = geom.centroid.buffer(150)
        spacing = 30
        source = "landmark_radial"

    elif subtype == "park":
        sampling_zone = geom.buffer(50)
        spacing = 50
        source = "park_near_edge"

    elif subtype in ["mall", "hawker", "museum", "theatre"]:
        sampling_zone = geom.buffer(300)
        spacing = 40
        source = "network_accessibility"

    elif subtype == "lifestyle_street":
        sampling_zone = geom.buffer(80)
        spacing = 25
        source = "street_corridor"

    else:
        continue

    sampled_points = get_network_samples(sampling_zone, spacing)

    for pt in sampled_points:
        point_counter += 1
        pt_wgs84 = gpd.GeoSeries([pt], crs=SVY21).to_crs(WGS84).iloc[0]

        target_heading = None
        if source == "landmark_radial":
            target_heading = calculate_bearing(
                pt_wgs84.y, pt_wgs84.x,
                centroid_lat, centroid_lon
            )

        sampling_buffer.append({
            "POI_ID": poi_id,
            "point_id": f"P_{point_counter:07d}",
            "lon": pt_wgs84.x,
            "lat": pt_wgs84.y,
            "target_heading": target_heading,
            "source": source
        })

        if len(sampling_buffer) >= AUTOSAVE_INTERVAL:
            autosave(sampling_buffer)
            sampling_buffer.clear()

    # autosave at POI boundary
    autosave(sampling_buffer)
    sampling_buffer.clear()

# -------------------------------------------------------------
# 6. Final cleanup (dedup once)
# -------------------------------------------------------------
final_df = pd.read_csv(SAMPLING_OUTPUT_CSV)
final_df = final_df.drop_duplicates(subset=["lon", "lat"])
final_df.to_csv(SAMPLING_OUTPUT_CSV, index=False)

print("-" * 40)
print("Sampling completed successfully.")
print(f"Total sampling points: {len(final_df)}")
print(f"Output saved to: {SAMPLING_OUTPUT_CSV}")
