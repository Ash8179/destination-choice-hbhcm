"""
C.1.4 – Singapore OSM Data Downloader (Comprehensive)

Strategy: Download ALL buildings and leisure spaces in Singapore, then filter
during matching. This prevents missing destinations due to inconsistent OSM tagging.

Author: Zhang Wenyu
Date: 2025-12-23
"""

import geopandas as gpd
import requests
import time
import json
from shapely.geometry import shape, Point
from datetime import datetime
import random
import os

# --- CONFIGURATION ---
OUTPUT_PATH = r"/Users/zhangwenyu/Desktop/NUSFYP/Stage 1/1.1 Define Origins and Destinations/Destinations/Singapore_OSM_Complete.geojson"

# Singapore divided into 12 smaller chunks to avoid timeout
SINGAPORE_CHUNKS = [
    {'name': 'North_1', 'bbox': (1.35, 103.6, 1.38, 103.75)},
    {'name': 'North_2', 'bbox': (1.38, 103.6, 1.41, 103.75)},
    {'name': 'North_3', 'bbox': (1.35, 103.75, 1.38, 103.9)},
    {'name': 'North_4', 'bbox': (1.38, 103.75, 1.41, 103.9)},
    
    {'name': 'Central_1', 'bbox': (1.25, 103.75, 1.28, 103.87)},
    {'name': 'Central_2', 'bbox': (1.28, 103.75, 1.31, 103.87)},
    {'name': 'Central_3', 'bbox': (1.25, 103.87, 1.28, 104.0)},
    {'name': 'Central_4', 'bbox': (1.28, 103.87, 1.31, 104.0)},
    
    {'name': 'East_1', 'bbox': (1.25, 104.0, 1.28, 104.05)},
    {'name': 'East_2', 'bbox': (1.28, 104.0, 1.31, 104.05)},
    
    {'name': 'West_1', 'bbox': (1.15, 103.6, 1.25, 103.725)},
    {'name': 'West_2', 'bbox': (1.25, 103.6, 1.35, 103.725)}
]

# Rotate between public Overpass instances
OVERPASS_INSTANCES = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter"
]

LOCAL_CRS = "EPSG:3414"  # Singapore SVY21
WGS84_CRS = "EPSG:4326"

def build_comprehensive_query(bbox):
    """
    Download EVERYTHING that could possibly be a destination.
    Filter during matching, not during download.
    """
    
    south, west, north, east = bbox
    bbox_str = f"{south},{west},{north},{east}"
    
    return f"""
    [out:json][timeout:180][bbox:{bbox_str}];
    (
      // ALL buildings
      way["building"];
      relation["building"];
      
      // ALL leisure spaces
      way["leisure"];
      relation["leisure"];
      
      // ALL tourism
      way["tourism"];
      relation["tourism"];
      
      // ALL shops
      way["shop"];
      relation["shop"];
      
      // ALL amenities
      way["amenity"];
      relation["amenity"];
      
      // Historic sites
      way["historic"];
      relation["historic"];
      
      // Specific hawker centre nodes
      node["amenity"~"food_court|hawker_centre|marketplace"];
      
      // Commercial complexes
      way["landuse"="commercial"]["name"];
      relation["landuse"="commercial"]["name"];
    );
    out geom;
    """

def safe_request(url, query, attempt=1, max_attempts=3):
    """
    Make API request with exponential backoff and error handling.
    """
    
    try:
        response = requests.post(url, data={'data': query}, timeout=200)
        
        # Rate limiting
        if response.status_code == 429:
            if attempt < max_attempts:
                wait_time = (2 ** attempt) * 30
                print(f"Rate limited. Waiting {wait_time}s...")
                time.sleep(wait_time)
                return safe_request(url, query, attempt + 1, max_attempts)
            else:
                raise Exception("Max retry attempts due to rate limiting")
        
        # Server errors
        if response.status_code in [503, 504]:
            if attempt < max_attempts:
                print(f"Server timeout. Waiting 60s...")
                time.sleep(60)
                return safe_request(url, query, attempt + 1, max_attempts)
            else:
                raise Exception("Server consistently timing out")
        
        if not response.headers.get("Content-Type", "").startswith("application/json"):
            raise Exception("Non-JSON response received")
        
        response.raise_for_status()
        return response.json()
    
    except Exception as e:
        if attempt < max_attempts:
            print(f"Error: {str(e)}. Trying different server...")
            time.sleep(30)
            next_url = OVERPASS_INSTANCES[(OVERPASS_INSTANCES.index(url) + 1) % len(OVERPASS_INSTANCES)]
            return safe_request(next_url, query, attempt + 1, max_attempts)
        raise

def parse_osm_elements(elements):
    """
    Convert OSM elements to features with all tags preserved.
    """
    
    features = []
    
    for el in elements:
        try:
            geom = None
            tags = el.get('tags', {})
            
            if not tags:
                continue
            
            # Ways (closed polygons)
            if el['type'] == 'way' and 'geometry' in el:
                coords = [(p['lon'], p['lat']) for p in el['geometry']]
                if len(coords) >= 4 and coords[0] == coords[-1]:
                    geom = shape({"type": "Polygon", "coordinates": [coords]})
            
            # Relations (multipolygons)
            elif el['type'] == 'relation':
                outer_rings = []
                for member in el.get('members', []):
                    if member.get('role') == 'outer' and 'geometry' in member:
                        coords = [(p['lon'], p['lat']) for p in member['geometry']]
                        if len(coords) >= 4:
                            outer_rings.append(coords)
                
                if outer_rings:
                    if len(outer_rings) == 1:
                        geom = shape({"type": "Polygon", "coordinates": [outer_rings[0]]})
                    else:
                        geom = shape({"type": "MultiPolygon", "coordinates": [[[ring]] for ring in outer_rings]})
            
            # Nodes (point features)
            elif el['type'] == 'node' and 'lat' in el and 'lon' in el:
                point = Point(el['lon'], el['lat'])
                geom = point.buffer(0.0002)  # ~20m radius
            
            if geom and geom.is_valid:
                feature = {
                    'geometry': geom,
                    'osm_id': el.get('id'),
                    'osm_type': el.get('type'),
                    'name': tags.get('name', ''),
                    'amenity': tags.get('amenity', ''),
                    'leisure': tags.get('leisure', ''),
                    'tourism': tags.get('tourism', ''),
                    'shop': tags.get('shop', ''),
                    'building': tags.get('building', '')
                }
                features.append(feature)
        
        except Exception as e:
            print(f"Skipped element {el.get('id')}: {str(e)}")
            continue
    
    return features

def download_chunk(chunk_info):
    """
    Download one geographic chunk with progress tracking.
    """
    name = chunk_info['name']
    bbox = chunk_info['bbox']
    
    print(f"\n{'='*70}")
    print(f"Downloading {name} Region")
    print(f"   Bbox: {bbox}")
    print(f"{'='*70}")
    
    query = build_comprehensive_query(bbox)
    time.sleep(random.uniform(1, 3))
    server_url = random.choice(OVERPASS_INSTANCES)
    
    data = safe_request(server_url, query)
    elements = data.get('elements', [])
    print(f"Retrieved {len(elements)} raw elements")
    
    features = parse_osm_elements(elements)
    print(f"Parsed {len(features)} valid features")
    
    return features

def download_singapore_data():
    """
    Main orchestrator - downloads all chunks and merges.
    """
    
    all_features = []
    
    for i, chunk in enumerate(SINGAPORE_CHUNKS, 1):
        try:
            chunk_features = download_chunk(chunk)
            all_features.extend(chunk_features)
            
            print(f"Chunk complete. Total features so far: {len(all_features)}")
            if i < len(SINGAPORE_CHUNKS):
                wait = random.randint(10, 15)
                time.sleep(wait)
        except Exception as e:
            print(f"Failed to download {chunk['name']}: {str(e)}")
            continue
    
    if not all_features:
        raise Exception("No features extracted!")
    
    gdf = gpd.GeoDataFrame(all_features, crs=WGS84_CRS)
    gdf = gdf.drop_duplicates(subset=['osm_id'])
    
    gdf_projected = gdf.to_crs(LOCAL_CRS)
    gdf['area_sqm'] = gdf_projected.geometry.area
    
    gdf.to_file(OUTPUT_PATH, driver='GeoJSON')
    print(f"Saved {len(gdf)} features to {OUTPUT_PATH}")

if __name__ == "__main__":
    try:
        download_singapore_data()
    except KeyboardInterrupt:
        print("\nDownload interrupted by user")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("\nFatal Error. Check network or retry later.")
