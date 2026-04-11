"""
C.1.5 – Destination Footprint Matching

Pipeline:
1. Point-in-Polygon (PIP)
2. Fuzzy name match within 50 m
3. Largest polygon fallback (category filtered)
4. Union fallback (lowest confidence)

Author: Zhang Wenyu
Date: 2025-12-24
"""

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union
from rapidfuzz import fuzz
from collections import Counter

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------

POI_PATH = (
    "/Users/zhangwenyu/Desktop/NUSFYP/Stage 1/"
    "1.1 Define Origins and Destinations/Destinations/Destination.xlsx"
)

OSM_PATH = (
    "/Users/zhangwenyu/Desktop/NUSFYP/Stage 1/"
    "1.1 Define Origins and Destinations/Destinations/"
    "Singapore_OSM_Complete.geojson"
)

OUTPUT_PATH = (
    "/Users/zhangwenyu/Desktop/NUSFYP/Stage 1/"
    "1.1 Define Origins and Destinations/Destinations/"
    "Destination_with_Footprints.xlsx"
)

WGS84 = "EPSG:4326"
SVY21 = "EPSG:3414"

FUZZY_RADIUS_M = 50
FUZZY_THRESHOLD = 70

# ------------------------------------------------------------------
# CATEGORY FILTERING RULES
# ------------------------------------------------------------------

CATEGORY_RULES = {
    "mall": {
        "shop": ["mall"],
        "building": ["retail", "commercial"]
    },
    "hawker centre": {
        "amenity": ["food_court", "hawker_centre", "marketplace"]
    },
    "park": {
        "leisure": ["park"]
    }
}

# ------------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------------

print("Loading POI data...")
poi_df = pd.read_excel(POI_PATH)

print("Loading OSM polygons...")
osm_gdf = gpd.read_file(OSM_PATH).to_crs(SVY21)

# Ensure required columns
required_cols = ["poi_id", "name", "centroid_x", "centroid_y", "subtype"]
missing = set(required_cols) - set(poi_df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Convert POIs to GeoDataFrame
poi_gdf = gpd.GeoDataFrame(
    poi_df,
    geometry=gpd.points_from_xy(
        poi_df["centroid_y"],
        poi_df["centroid_x"],
        crs=WGS84
    )
).to_crs(SVY21)

# ------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------

def category_filter(gdf, subtype):
    """Apply strict semantic filtering based on POI subtype."""
    rules = CATEGORY_RULES.get(subtype.lower())
    if not rules:
        return gdf

    mask = False
    for col, values in rules.items():
        if col in gdf.columns:
            mask |= gdf[col].isin(values)

    return gdf[mask]


def fuzzy_score(a, b):
    return fuzz.token_set_ratio(a.lower(), b.lower())


# ------------------------------------------------------------------
# MATCHING PIPELINE
# ------------------------------------------------------------------

results = []
match_counter = Counter()

print("\nStarting footprint matching...\n")

for idx, poi in poi_gdf.iterrows():
    poi_id = poi["poi_id"]
    name = str(poi["name"])
    subtype = str(poi["subtype"])
    point = poi.geometry

    print(f"POI {poi_id} | {name}")

    # --------------------------------------------------------------
    # Step 1: Point-in-Polygon (PIP)
    # --------------------------------------------------------------
    pip_matches = osm_gdf[osm_gdf.contains(point)]

    pip_matches = category_filter(pip_matches, subtype)

    if not pip_matches.empty:
        geom = unary_union(pip_matches.geometry)
        area = geom.area
        match_type = (
            "pip_union" if len(pip_matches) > 1 else "pip_single"
        )

        print(f"PIP match | polygons={len(pip_matches)} | area={area:.1f} sqm")

    else:
        # ----------------------------------------------------------
        # Step 2: Fuzzy name match within 50 m
        # ----------------------------------------------------------
        buffer = point.buffer(FUZZY_RADIUS_M)
        nearby = osm_gdf[osm_gdf.intersects(buffer)]
        nearby = category_filter(nearby, subtype)

        if not nearby.empty:
            nearby = nearby.copy()
            nearby["fuzzy"] = nearby["name"].apply(
                lambda x: fuzzy_score(name, x) if x else 0
            )
            best = nearby[nearby["fuzzy"] >= FUZZY_THRESHOLD]

            if not best.empty:
                geom = unary_union(best.geometry)
                area = geom.area
                match_type = (
                    "name_match_union" if len(best) > 1 else "name_match_single"
                )

                print(
                    f"Name match | polygons={len(best)} | "
                    f"best_score={best['fuzzy'].max()} | area={area:.1f} sqm"
                )

            else:
                # --------------------------------------------------
                # Step 3: Largest polygon fallback
                # --------------------------------------------------
                nearby["area"] = nearby.geometry.area
                largest = nearby.sort_values("area", ascending=False).head(1)

                if not largest.empty:
                    geom = largest.geometry.iloc[0]
                    area = geom.area
                    match_type = "largest_polygon"

                    print(
                        f"Largest polygon fallback | "
                        f"area={area:.1f} sqm"
                    )
                else:
                    # --------------------------------------------------
                    # Step 4: Union fallback
                    # --------------------------------------------------
                    geom = unary_union(nearby.geometry)
                    area = geom.area if not geom.is_empty else 0.0
                    match_type = "union_all"

                    print(f"Union fallback | area={area:.1f} sqm")

        else:
            geom = None
            area = 0.0
            match_type = "no_match"
            print("No candidates found")

    match_counter[match_type] += 1

    results.append({
        "poi_id": poi_id,
        "name": name,
        "geom_polygon": geom.wkt if geom else None,
        "footprint_area": area,
        "match_type": match_type
    })

# ------------------------------------------------------------------
# SAVE OUTPUT
# ------------------------------------------------------------------

result_df = pd.DataFrame(results)
result_df.to_excel(OUTPUT_PATH, index=False)

# ------------------------------------------------------------------
# FINAL SUMMARY
# ------------------------------------------------------------------

print("\n" + "=" * 60)
print("FINAL MATCHING SUMMARY")
print("=" * 60)

total = sum(match_counter.values())
for k, v in match_counter.items():
    print(f"{k:20s}: {v:5d} ({v / total:.1%})")

print("=" * 60)
print(f"Total POIs processed: {total}")
print(f"Output saved to:\n{OUTPUT_PATH}")
print("=" * 60)
