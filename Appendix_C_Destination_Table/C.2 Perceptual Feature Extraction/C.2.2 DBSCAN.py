"""
C.2.2 – Spatial Deduplication via DBSCAN

This script:
Perform POI-level spatial deduplication of sampling points using DBSCAN,
with source-specific distance thresholds and centroid-based representative selection.
Outputs deduplicated points and before–after summary statistics.

Author: Zhang Wenyu
Date: 2026-01-24
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from sklearn.cluster import DBSCAN

# --------------------------------------------------
# Configuration
# --------------------------------------------------
INPUT_CSV = "/Users/zhangwenyu/Desktop/NUSFYP/Stage 1/1.3 Street Level Imagery/sampling_points.csv"
OUTPUT_DEDUP_CSV = "/Users/zhangwenyu/Desktop/NUSFYP/Stage 1/1.3 Street Level Imagery/sampling_points_dedup.csv"
OUTPUT_SUMMARY_CSV = "/Users/zhangwenyu/Desktop/NUSFYP/Stage 1/1.3 Street Level Imagery/dedup_summary_by_poi.csv"

# Distance threshold (meters) by perceptual source
EPS_BY_SOURCE = {
    "network_accessibility": 25,
    "landmark_radial": 10,
    "park_near_edge": 20
}

# --------------------------------------------------
# Load data
# --------------------------------------------------
df = pd.read_csv(INPUT_CSV)
df = df.dropna(subset=["lon", "lat", "POI_ID", "source"])

gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df.lon, df.lat),
    crs="EPSG:4326"
).to_crs("EPSG:3414")  # metric CRS for clustering

# --------------------------------------------------
# Deduplication function (per POI + per source)
# --------------------------------------------------
def spatial_dedup_per_group(gdf_group):
    """
    Perform spatial deduplication within a single POI_ID.
    Clustering is further separated by perceptual source.
    """
    kept_indices = []

    for source, sub in gdf_group.groupby("source"):
        eps = EPS_BY_SOURCE.get(source, 20)

        coords = np.vstack([
            sub.geometry.x.values,
            sub.geometry.y.values
        ]).T

        if len(coords) == 0:
            continue

        db = DBSCAN(
            eps=eps,
            min_samples=1,
            metric="euclidean"
        )

        sub = sub.copy()
        sub["cluster_id"] = db.fit_predict(coords)

        # Select representative point per cluster
        for _, cluster in sub.groupby("cluster_id"):
            centroid = cluster.geometry.union_all().centroid
            distances = cluster.geometry.distance(centroid)
            kept_indices.append(distances.idxmin())

    return gdf_group.loc[kept_indices]

# --------------------------------------------------
# Run deduplication POI by POI
# --------------------------------------------------
dedup_gdf = (
    gdf
    .groupby("POI_ID", group_keys=False)
    .apply(spatial_dedup_per_group)
)

# --------------------------------------------------
# Convert back to WGS84
# --------------------------------------------------
dedup_gdf = dedup_gdf.to_crs("EPSG:4326")
dedup_gdf["lon"] = dedup_gdf.geometry.x
dedup_gdf["lat"] = dedup_gdf.geometry.y

# --------------------------------------------------
# Save deduplicated points
# --------------------------------------------------
cols_to_keep = [
    "point_id", "POI_ID", "lon", "lat", "source", "target_heading"
]

dedup_gdf[cols_to_keep].to_csv(OUTPUT_DEDUP_CSV, index=False)

# --------------------------------------------------
# Summary statistics (before vs after)
# --------------------------------------------------
summary_before = (
    gdf.groupby(["POI_ID", "source"])
    .size()
    .rename("count_before")
    .reset_index()
)

summary_after = (
    dedup_gdf.groupby(["POI_ID", "source"])
    .size()
    .rename("count_after")
    .reset_index()
)

summary = pd.merge(
    summary_before,
    summary_after,
    on=["POI_ID", "source"],
    how="left"
).fillna(0)

summary["count_after"] = summary["count_after"].astype(int)
summary["reduction_ratio"] = 1 - summary["count_after"] / summary["count_before"]

summary.to_csv(OUTPUT_SUMMARY_CSV, index=False)

print("Deduplication completed.")
print(f"Original points: {len(gdf)}")
print(f"After deduplication: {len(dedup_gdf)}")
