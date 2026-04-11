"""
D.1 – OD PT Routing Data Collection (OneMap API)

This script:
Fetch public transit routes between origin–destination pairs using
OneMap API, with coordinate transformation, rate limiting, parallel
requests, and per-origin result storage with progress tracking.

Author: Zhang Wenyu
Date: 2026-01-06
"""

import pandas as pd
import requests
import json
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pyproj import Transformer
import threading

# ================= CONFIG =================
O_PATH = "/Users/zhangwenyu/Desktop/NUSFYP/O.xlsx"
D_PATH = "/Users/zhangwenyu/Desktop/NUSFYP/D.xlsx"
BASE_OUTPUT_DIR = "/Users/zhangwenyu/Desktop/NUSFYP/Stage 1/1.2 Accessibility (Skims)"

ONEMAP_TOKEN = "YOUR_TOKEN"
BASE_URL = "https://www.onemap.gov.sg/api/public/routingsvc/route"

DATE = "12-13-2025"
TIME = "14:00:00"
MODE = "TRANSIT"

MAX_WORKERS = 6
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
MIN_REQUEST_INTERVAL = 0.18  # ~280 req/min

# ================= RATE LIMITER =================
rate_lock = threading.Lock()
last_request_ts = 0.0

def rate_limit_wait():
    global last_request_ts
    with rate_lock:
        now = time.time()
        wait = MIN_REQUEST_INTERVAL - (now - last_request_ts)
        if wait > 0:
            time.sleep(wait)
        last_request_ts = time.time()

# ================= COORD TRANSFORM =================
transformer = Transformer.from_crs("EPSG:3414", "EPSG:4326", always_xy=True)

def svy21_to_wgs84(x, y):
    """Convert SVY21 to WGS84 (lat, lon)"""
    lon, lat = transformer.transform(x, y)
    return lat, lon

# ================= FETCH ROUTE =================
def fetch_route(origin_id, start_coord, dest_id, dest_lat, dest_lon, output_dir):
    """Fetch route from OneMap API"""
    out_path = os.path.join(output_dir, f"O{origin_id}_D{dest_id}_PT.json")
    if os.path.exists(out_path):
        return "skipped"

    # OneMap API expects "lat,lon" format
    end_coord = f"{dest_lat},{dest_lon}"

    params = {
        "start": start_coord,
        "end": end_coord,
        "routeType": "pt",
        "date": DATE,
        "time": TIME,
        "mode": MODE,
        "maxWalkDistance": "1000",
        "numItineraries": "1",
    }

    headers = {"Authorization": ONEMAP_TOKEN}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            rate_limit_wait()
            response = requests.get(BASE_URL, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
            
            if response.status_code == 200:
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(response.json(), f, ensure_ascii=False, indent=2)
                return "success"

            # Print error details
            print(f"\n[ERROR] O{origin_id}_D{dest_id}: Status {response.status_code}")
            print(f"  Start: {start_coord}")
            print(f"  End: {end_coord}")
            print(f"  Response: {response.text[:200]}")
            
            if response.status_code >= 429:
                time.sleep(2 ** attempt)
                continue
            break

        except Exception as e:
            print(f"\n[EXCEPTION] O{origin_id}_D{dest_id}: {str(e)}")
            if attempt == MAX_RETRIES:
                return "exception"
            time.sleep(2 ** attempt)

    # Save error file
    error_path = os.path.join(output_dir, f"ERROR_O{origin_id}_D{dest_id}.json")
    with open(error_path, "w", encoding="utf-8") as f:
        json.dump({
            "status": getattr(response, "status_code", None),
            "response": getattr(response, "text", None),
            "start": start_coord,
            "end": end_coord
        }, f, indent=2)

    return "error"

# ================= PROCESS ORIGIN =================
def process_origin(origin_row, destinations):
    """Process all routes from one origin to all destinations"""
    origin_id = origin_row["origin_id"]
    output_dir = os.path.join(BASE_OUTPUT_DIR, str(origin_id))
    os.makedirs(output_dir, exist_ok=True)

    summary_path = os.path.join(output_dir, "origin_summary.json")
    if os.path.exists(summary_path):
        print(f"O{origin_id} already done, skipping.")
        return

    # Convert origin coordinates from SVY21 to WGS84
    o_lat, o_lon = svy21_to_wgs84(origin_row["centroid_x"], origin_row["centroid_y"])
    start_coord = f"{o_lat},{o_lon}"

    # Build task list
    # Based on actual values: centroid_x has values ~1.x (latitude), centroid_y has ~103.x (longitude)
    tasks = []
    for _, row in destinations.iterrows():
        # Swap the assignments based on actual data values
        dest_lat = row["centroid_x"]  # Actually contains latitude (1.2...)
        dest_lon = row["centroid_y"]  # Actually contains longitude (103.8...)
        
        tasks.append((
            origin_id,
            start_coord,
            row["poi_id"],
            dest_lat,  # Latitude (1.2...)
            dest_lon,  # Longitude (103.8...)
            output_dir
        ))

    success = error = exception = skipped = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(fetch_route, *task) for task in tasks]

        with tqdm(total=len(futures), desc=f"O{origin_id}", unit="dest") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result == "success":
                    success += 1
                elif result == "error":
                    error += 1
                elif result == "exception":
                    exception += 1
                else:
                    skipped += 1
                pbar.update(1)
                pbar.set_postfix({"ok": success, "skip": skipped, "err": error, "exc": exception})

    # Save summary
    summary = {
        "origin_id": origin_id,
        "success": success,
        "skipped": skipped,
        "error": error,
        "exception": exception,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(BASE_OUTPUT_DIR, "progress_autosave.json"), "a", encoding="utf-8") as f:
        f.write(json.dumps(summary) + "\n")

# ================= MAIN =================
def main():
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    
    # Load data
    origins = pd.read_excel(O_PATH)
    destinations = pd.read_excel(D_PATH)

    # Process each origin
    for _, origin_row in origins.iterrows():
        process_origin(origin_row, destinations)

if __name__ == "__main__":
    main()
