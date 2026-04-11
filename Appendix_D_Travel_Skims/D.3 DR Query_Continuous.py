"""
D.3 – OD Drive Routing Data Collection (OneMap API)

This script:
Fetch drive routes between origin–destination pairs using
OneMap API, with coordinate transformation, rate limiting, parallel
requests, and per-origin result storage with progress tracking.

Author: Zhang Wenyu
Date: 2026-01-12
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

ROUTE_TYPE = "drive"  # Changed from transit to drive

MAX_WORKERS = 6
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
MIN_REQUEST_INTERVAL = 0.2  # 300 requests/min = 1 request every 0.2 seconds

# ================= RATE LIMITER =================
rate_lock = threading.Lock()
last_request_ts = 0.0

def rate_limit_wait():
    """Enforce rate limit of 300 requests per minute"""
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
    """Convert SVY21 (Singapore coordinate system) to WGS84 (lat, lon)"""
    lon, lat = transformer.transform(x, y)
    return lat, lon

# ================= FETCH ROUTE =================
def fetch_route(origin_id, start_coord, dest_id, dest_lat, dest_lon, output_dir):
    """Fetch driving route from OneMap API"""
    # Update filename to reflect drive mode
    out_path = os.path.join(output_dir, f"O{origin_id}_D{dest_id}_DRIVE.json")
    if os.path.exists(out_path):
        return "skipped"

    # OneMap API expects "lat,lon" format
    end_coord = f"{dest_lat},{dest_lon}"

    # Simplified parameters for drive route (no transit-specific params needed)
    params = {
        "start": start_coord,
        "end": end_coord,
        "routeType": ROUTE_TYPE,  # "drive"
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

            # Log error details for debugging
            print(f"\n[ERROR] O{origin_id}_D{dest_id}: Status {response.status_code}")
            print(f"  Start: {start_coord}")
            print(f"  End: {end_coord}")
            print(f"  Response: {response.text[:200]}")
            
            # Retry on rate limit errors
            if response.status_code >= 429:
                time.sleep(2 ** attempt)
                continue
            break

        except Exception as e:
            print(f"\n[EXCEPTION] O{origin_id}_D{dest_id}: {str(e)}")
            if attempt == MAX_RETRIES:
                return "exception"
            time.sleep(2 ** attempt)

    # Save error details to file for later analysis
    error_path = os.path.join(output_dir, f"ERROR_O{origin_id}_D{dest_id}.json")
    with open(error_path, "w", encoding="utf-8") as f:
        json.dump({
            "status": getattr(response, "status_code", None),
            "response": getattr(response, "text", None),
            "start": start_coord,
            "end": end_coord,
            "route_type": ROUTE_TYPE
        }, f, indent=2)

    return "error"

# ================= PROCESS ORIGIN =================
def process_origin(origin_row, destinations):
    """Process all driving routes from one origin to all destinations"""
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

    # Build task list for all destinations
    # Note: D.xlsx has swapped column names - centroid_x contains latitude, centroid_y contains longitude
    tasks = []
    for _, row in destinations.iterrows():
        dest_lat = row["centroid_x"]  # Actually contains latitude (~1.2...)
        dest_lon = row["centroid_y"]  # Actually contains longitude (~103.8...)
        
        tasks.append((
            origin_id,
            start_coord,
            row["poi_id"],
            dest_lat,
            dest_lon,
            output_dir
        ))

    # Track results
    success = error = exception = skipped = 0

    # Execute requests in parallel with rate limiting
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

    # Save summary statistics for this origin
    summary = {
        "origin_id": origin_id,
        "route_type": ROUTE_TYPE,
        "success": success,
        "skipped": skipped,
        "error": error,
        "exception": exception,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Append to global progress log
    with open(os.path.join(BASE_OUTPUT_DIR, "progress_autosave.json"), "a", encoding="utf-8") as f:
        f.write(json.dumps(summary) + "\n")

# ================= MAIN =================
def main():
    """Main execution function"""
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    
    # Load origin and destination data
    origins = pd.read_excel(O_PATH)
    destinations = pd.read_excel(D_PATH)

    print(f"Loaded {len(origins)} origins and {len(destinations)} destinations")
    print(f"Route type: {ROUTE_TYPE}")
    print(f"Rate limit: {int(60/MIN_REQUEST_INTERVAL)} requests/min")
    print(f"Max workers: {MAX_WORKERS}")
    print(f"Output directory: {BASE_OUTPUT_DIR}\n")

    # Process each origin sequentially
    for idx, origin_row in origins.iterrows():
        print(f"\n[{idx+1}/{len(origins)}] Processing origin {origin_row['origin_id']}...")
        process_origin(origin_row, destinations)

    print("\n✓ All origins processed!")

if __name__ == "__main__":
    main()
