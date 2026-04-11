"""
C.1.6 – Interactive Destination Footprint Resolution

Improvements:
- Show OSM tags (building, amenity, shop) for better decision making
- Display progress counter
- Save incremental results (auto-save every 10 POIs)
- Better visualization of candidates
- Add 'a' option to accept largest polygon quickly
- Show area reasonableness warnings
- Directly update the original file

Author: Zhang Wenyu
Date: 2025-12-24
"""

import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union
from collections import Counter
import os
import shutil
from datetime import datetime

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------

POI_PATH = (
    "/Users/zhangwenyu/Desktop/NUSFYP/Stage 1/"
    "1.1 Define Origins and Destinations/Destinations/Destination.xlsx"
)

RESULT_PATH = (
    "/Users/zhangwenyu/Desktop/NUSFYP/Stage 1/"
    "1.1 Define Origins and Destinations/Destinations/"
    "Destination_with_Footprints.xlsx"
)

OSM_PATH = (
    "/Users/zhangwenyu/Desktop/NUSFYP/Stage 1/"
    "1.1 Define Origins and Destinations/Destinations/"
    "Singapore_OSM_Complete.geojson"
)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
BACKUP_PATH = (
    "/Users/zhangwenyu/Desktop/NUSFYP/Stage 1/"
    "1.1 Define Origins and Destinations/Destinations/"
    f"Destination_with_Footprints_backup_{timestamp}.xlsx"
)

WGS84 = "EPSG:4326"
SVY21 = "EPSG:3414"

BUFFER_M = 100
AUTO_SAVE_INTERVAL = 10

# Area constraints for warnings
AREA_CONSTRAINTS = {
    "mall": {"min": 1000, "max": 300000},
    "hawker centre": {"min": 500, "max": 15000},
    "park": {"min": 500, "max": 2000000},
    "default": {"min": 100, "max": 500000}
}

# ------------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------------

print("Loading POIs...")
poi_df = pd.read_excel(POI_PATH)

print("Loading existing matching results...")
result_df = pd.read_excel(RESULT_PATH)

print(f"Creating backup: {BACKUP_PATH}")
shutil.copy2(RESULT_PATH, BACKUP_PATH)

print("Loading OSM polygons...")
osm_gdf = gpd.read_file(OSM_PATH).to_crs(SVY21)
osm_gdf["geometry"] = osm_gdf.geometry.buffer(0)
osm_gdf["area_sqm"] = osm_gdf.geometry.area

# Merge POI info with results
df = poi_df.merge(
    result_df[["poi_id", "match_type", "footprint_area", "geom_polygon"]],
    on="poi_id",
    how="left"
)

# Convert to GeoDataFrame
poi_gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(
        df["centroid_y"], df["centroid_x"], crs=WGS84
    )
).to_crs(SVY21)

# Only process no_match
targets = poi_gdf[poi_gdf["match_type"] == "no_match"].copy()

print(f"\n{'='*70}")
print(f"POIs requiring manual resolution: {len(targets)}")
print(f"{'='*70}\n")

if len(targets) == 0:
    print("No POIs need manual resolution!")
    exit()

# ------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------

def area_reasonable(area, subtype):
    """Check if area is reasonable and return warning if not"""
    constraints = AREA_CONSTRAINTS.get(subtype.lower(), AREA_CONSTRAINTS["default"])
    
    if area < constraints["min"]:
        return False, f"WARNING: Area too small (< {constraints['min']} sqm)"
    elif area > constraints["max"]:
        return False, f"WARNING: Area too large (> {constraints['max']} sqm)"
    else:
        return True, "Area reasonable"


def format_tags(row):
    """Format relevant OSM tags for display"""
    tags = []
    
    tag_cols = ['building', 'amenity', 'shop', 'leisure', 'landuse']
    for col in tag_cols:
        if col in row.index and pd.notna(row[col]):
            tags.append(f"{col}={row[col]}")
    
    return " | ".join(tags) if tags else "no tags"


def save_progress(poi_gdf, path):
    """Save current progress to the original file"""
    try:
        output_df = poi_gdf.drop(columns="geometry")
        output_df.to_excel(path, index=False)
        print(f"Progress saved")
    except Exception as e:
        print(f"Failed to save: {e}")


# ------------------------------------------------------------------
# INTERACTIVE SELECTION FUNCTION
# ------------------------------------------------------------------

def interactive_select(poi_row, candidates, progress_str):
    print("\n" + "=" * 70)
    print(f"{progress_str}")
    print(f"POI: {poi_row['name']}")
    print(f"Type: {poi_row['subtype']}")
    print(f"ID: {poi_row['poi_id']}")
    print("=" * 70)

    candidates = candidates.copy()
    candidates["area"] = candidates.geometry.area
    candidates["dist"] = candidates.geometry.distance(poi_row.geometry)

    # Sort by distance, then by area
    candidates = candidates.sort_values(["dist", "area"], ascending=[True, False])

    # Show top 30 candidates
    display_limit = min(30, len(candidates))
    
    for i, (idx, row) in enumerate(candidates.head(display_limit).iterrows(), start=1):
        name_str = row.get('name', '∅') or '∅'
        tags_str = format_tags(row)
        
        # Area warning
        reasonable, warning = area_reasonable(row["area"], poi_row['subtype'])
        warning_icon = "" if reasonable else "⚠"
        
        print(
            f"[{i:2d}] {warning_icon} "
            f"dist={row['dist']:5.1f}m | "
            f"area={row['area']:8.1f} sqm | "
            f"name={name_str[:25]:<25s} | "
            f"{tags_str}"
        )
    
    if len(candidates) > display_limit:
        print(f"... and {len(candidates) - display_limit} more candidates")

    print("\n" + "-" * 70)
    print("Options:")
    print("  [numbers]  - Select by index (e.g. 1,3,4 or just 1)")
    print("  a          - Accept largest/closest polygon automatically")
    print("  u          - Union all candidates")
    print("  s          - Skip this POI")
    print("  q          - Quit and save progress")
    print("-" * 70)

    choice = input("Your choice: ").strip().lower()

    if choice == "q":
        raise KeyboardInterrupt

    if choice == "s":
        return None, "manual_skip"

    if choice == "a":
        # Accept the first (closest) polygon
        best = candidates.iloc[0]
        geom = best.geometry
        area = best["area"]
        
        reasonable, warning = area_reasonable(area, poi_row['subtype'])
        print(f"\nAccepting: {best.get('name', 'N/A')} | area={area:.1f} sqm")
        print(warning)
        
        return geom, "manual_accept"

    if choice == "u":
        geom = unary_union(candidates.geometry)
        area = geom.area
        
        reasonable, warning = area_reasonable(area, poi_row['subtype'])
        print(f"\nUnion of {len(candidates)} polygons | total area={area:.1f} sqm")
        print(warning)
        
        confirm = input("Confirm union? (y/n): ").strip().lower()
        if confirm == "y":
            return geom, "manual_union_all"
        else:
            return None, "manual_skip"

    try:
        # Parse index selection
        idxs = [int(x.strip()) - 1 for x in choice.replace(",", " ").split()]
        
        if not all(0 <= i < len(candidates) for i in idxs):
            print("Invalid index. Skipping.")
            return None, "manual_error"
        
        selected = candidates.iloc[idxs]
        
        if len(selected) == 1:
            geom = selected.geometry.iloc[0]
            area = geom.area
            match_type = "manual_single"
        else:
            geom = unary_union(selected.geometry)
            area = geom.area
            match_type = "manual_union"
        
        reasonable, warning = area_reasonable(area, poi_row['subtype'])
        print(f"\nSelected {len(selected)} polygon(s) | total area={area:.1f} sqm")
        print(warning)
        
        return geom, match_type
        
    except Exception as e:
        print(f"Invalid input: {e}. Skipping.")
        return None, "manual_error"


# ------------------------------------------------------------------
# MAIN LOOP
# ------------------------------------------------------------------

counter = Counter()
processed_count = 0

print("Starting manual resolution...\n")
print("TIP: Press Ctrl+C at any time to save and quit\n")

try:
    for i, (idx, poi) in enumerate(targets.iterrows(), start=1):
        point = poi.geometry
        progress_str = f"Progress: {i}/{len(targets)} ({i/len(targets):.1%})"

        buffer = point.buffer(BUFFER_M)
        candidates = osm_gdf[osm_gdf.intersects(buffer)]

        if candidates.empty:
            print(f"\n{progress_str}")
            print(f"POI {poi['poi_id']} | {poi['name']}")
            print("✗ No nearby OSM polygons within 100m")
            counter["manual_no_candidates"] += 1
            continue

        geom, mtype = interactive_select(poi, candidates, progress_str)

        if geom:
            area = geom.area
            poi_gdf.at[idx, "geom_polygon"] = geom.wkt
            poi_gdf.at[idx, "footprint_area"] = area
            poi_gdf.at[idx, "match_type"] = mtype

            print(f"Saved | method={mtype}")
        else:
            print("Skipped")

        counter[mtype] += 1
        processed_count += 1
        
        # Auto-save every N POIs to original file
        if processed_count % AUTO_SAVE_INTERVAL == 0:
            save_progress(poi_gdf, RESULT_PATH)

except KeyboardInterrupt:
    print("\n\n" + "=" * 70)
    print("Manual resolution interrupted by user")
    print("=" * 70)

# ------------------------------------------------------------------
# SAVE FINAL OUTPUT
# ------------------------------------------------------------------

print("\nSaving final results to original file...")
save_progress(poi_gdf, RESULT_PATH)

print("\n" + "=" * 70)
print("MANUAL RESOLUTION SUMMARY")
print("=" * 70)

if counter:
    total_processed = sum(counter.values())
    for k, v in sorted(counter.items(), key=lambda x: -x[1]):
        print(f"{k:25s}: {v:4d} ({v/total_processed:.1%})")
else:
    print("No POIs were processed in this session.")

print("=" * 70)

# Calculate final overall match rate
final_df = pd.read_excel(RESULT_PATH)
total_pois = len(final_df)
no_match_final = len(final_df[final_df['match_type'] == 'no_match'])
match_rate = (total_pois - no_match_final) / total_pois

print(f"\nFINAL OVERALL STATISTICS:")
print(f"  Total POIs:           {total_pois}")
print(f"  Successfully matched: {total_pois - no_match_final} ({match_rate:.1%})")
print(f"  Still unmatched:      {no_match_final} ({no_match_final/total_pois:.1%})")

print("=" * 70)
print(f"Results saved to:\n{RESULT_PATH}")
print(f"\nBackup file created:\n{BACKUP_PATH}")
print("=" * 70)
