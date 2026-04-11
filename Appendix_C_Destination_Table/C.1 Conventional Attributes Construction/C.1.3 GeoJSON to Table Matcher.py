"""
C.1.3 – GeoJSON to Table Matcher

This script matches POI (Point of Interest) data from an Excel table with Singapore's 
Master Plan 2019 subzone boundaries from a GeoJSON file.

Input:
- Excel table with columns: poi_id, centroid_x (longitude), centroid_y (latitude)
- GeoJSON file containing Singapore's subzone polygons

Output:
- Enhanced Excel file with matched planning_area, subzone, and subzone_code
- Console output showing unmatched poi_ids for verification

Method:
- Uses point-in-polygon containment check to find which subzone each POI belongs to
- Only exact matches are considered (no approximation)

Author: Zhang Wenyu
Date: 2025-12-23
"""

import json
import pandas as pd
from shapely.geometry import shape, Point

# Define CBD and central area
CORE_CBD = ["DOWNTOWN CORE", "STRAITS VIEW"]
CENTRAL_AREA = [
    "DOWNTOWN CORE", "STRAITS VIEW", "ORCHARD", "ROCHOR", "OUTRAM",
    "SINGAPORE RIVER", "MUSEUM", "NEWTON", "RIVER VALLEY",
    "MARINA SOUTH", "MARINA EAST"
]

def load_geojson(geojson_path):
    """Load GeoJSON file"""
    with open(geojson_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def find_matching_zone(lat, lon, geojson_data):
    """Point-in-polygon match using shapely"""
    point = Point(lon, lat)  # Point expects (x, y) = (lon, lat)
    for feature in geojson_data['features']:
        poly = shape(feature['geometry'])
        if poly.contains(point):
            return feature
    return None

def match_table_to_geojson(table_path, geojson_path, output_path):
    print("Loading data...")
    df = pd.read_excel(table_path)
    geojson_data = load_geojson(geojson_path)
    
    # Create new columns
    df['planning_area'] = ''
    df['subzone'] = ''
    df['cbd_flag'] = 0
    df['central_area_flag'] = 0
    
    unmatched_poi_ids = []

    print("Matching data...")
    for idx, row in df.iterrows():
        lat = row['centroid_x']
        lon = row['centroid_y']
        
        matched_feature = find_matching_zone(lat, lon, geojson_data)
        if matched_feature:
            props = matched_feature['properties']
            pa = props.get('PLN_AREA_N', '').upper()
            sz = props.get('SUBZONE_N', '')
            
            df.at[idx, 'planning_area'] = pa
            df.at[idx, 'subzone'] = sz
            
            # Flags
            df.at[idx, 'cbd_flag'] = 1 if pa in CORE_CBD else 0
            df.at[idx, 'central_area_flag'] = 1 if pa in CENTRAL_AREA else 0
        else:
            unmatched_poi_ids.append(row['poi_id'])
        
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx+1}/{len(df)} records")

    print(f"Saving results to {output_path}...")
    df.to_excel(output_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"Matching completed! Matched: {len(df) - len(unmatched_poi_ids)}/{len(df)}")
    print(f"Unmatched: {len(unmatched_poi_ids)}")
    if unmatched_poi_ids:
        print("Sample unmatched POI IDs:", unmatched_poi_ids[:10])
    
    return df, unmatched_poi_ids

# Usage
if __name__ == "__main__":
    TABLE_FILE = "/Users/zhangwenyu/Desktop/NUSFYP/Stage 1/1.1 Define Origins and Destinations/Destinations/Destination.xlsx"
    GEOJSON_FILE = "/Users/zhangwenyu/Desktop/NUSFYP/Stage 1/1.1 Define Origins and Destinations/Origins/geojson/MasterPlan2019SubzoneBoundaryNoSeaGEOJSON.geojson"
    OUTPUT_FILE = "/Users/zhangwenyu/Desktop/NUSFYP/Stage 1/1.1 Define Origins and Destinations/Destinations/Destination_matched.xlsx"
    
    result_df, unmatched_ids = match_table_to_geojson(TABLE_FILE, GEOJSON_FILE, OUTPUT_FILE)
    print("Sample results:")
    print(result_df[['poi_id','planning_area','subzone','cbd_flag','central_area_flag']].head(10))
