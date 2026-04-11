"""
D.4 - Drive Table Construction

Purpose
-------
This script builds a Driving (CAR) skim table from OneMap routing JSON files.

Key features:
1. Traverses all origin folders (O_XXXX) under the skim directory
2. Parses both successful DRIVE routing JSONs and ERROR JSONs
3. Extracts origin_id and destination_id from file names using regex
4. Joins origin and destination names from O.xlsx and D.xlsx
5. Normalizes units:
   - Travel time is converted from seconds to minutes
   - Distance is converted from meters to kilometers
6. Handles errors robustly:
   - 404 / 400 / other HTTP errors are retained
   - All numeric skim fields for error cases are filled with 0
7. Outputs:
   - A full Driving skim table (OD-level)
   - A detailed summary table counting OK vs error cases
8. Includes progress indicators for large batch processing

The resulting skim table is directly usable for accessibility analysis.

Author: Zhang Wenyu
Date: 2026-01-15
"""

import os
import re
import json
import pandas as pd
from tqdm import tqdm

# =========================
# Paths
# =========================
BASE_DIR = "/Users/zhangwenyu/Desktop/NUSFYP/Stage 1/1.2 Accessibility (Skims)"
ORIGIN_XLSX = "/Users/zhangwenyu/Desktop/NUSFYP/O.xlsx"
DEST_XLSX = "/Users/zhangwenyu/Desktop/NUSFYP/D.xlsx"

OUTPUT_SKIMS = os.path.join(BASE_DIR, "drive_skims_full.csv")
OUTPUT_SUMMARY = os.path.join(BASE_DIR, "drive_skims_summary.csv")

# =========================
# Load origin / destination lookup tables
# =========================
origin_df = pd.read_excel(ORIGIN_XLSX)
dest_df = pd.read_excel(DEST_XLSX)

origin_map = dict(zip(origin_df["origin_id"], origin_df["subzone_name"]))
dest_map = dict(zip(dest_df["poi_id"], dest_df["name"]))

# =========================
# Regex patterns
# =========================
# Normal DRIVE file:
# OO_0313_1_ML_0001_DRIVE.json
DRIVE_PATTERN = re.compile(
    r"O+_(\d+).*?(\d+_[A-Z]+_\d+)_DRIVE\.json"
)

# ERROR file:
# ERROR_OO_0313_D9_PK_0354.json
ERROR_PATTERN = re.compile(
    r"ERROR_O+_(\d+).*?(\d+_[A-Z]+_\d+)\.json"
)

# =========================
# Containers
# =========================
rows = []
summary_records = []

# =========================
# Traverse origin folders
# =========================
origin_folders = [
    f for f in os.listdir(BASE_DIR)
    if os.path.isdir(os.path.join(BASE_DIR, f)) and f.startswith("O_")
]

print(f"\nFound {len(origin_folders)} origin folders to process")

for origin_folder in tqdm(origin_folders, desc="Processing origins"):
    origin_path = os.path.join(BASE_DIR, origin_folder)

    for fname in os.listdir(origin_path):
        if not fname.endswith(".json"):
            continue

        # Only process DRIVE-related files
        if "_DRIVE" not in fname and not fname.startswith("ERROR_"):
            continue

        fpath = os.path.join(origin_path, fname)

        drive_match = DRIVE_PATTERN.search(fname)
        err_match = ERROR_PATTERN.search(fname)

        origin_id = None
        dest_id = None

        if drive_match:
            origin_num = drive_match.group(1)
            origin_id = f"O_{origin_num}"
            dest_id = drive_match.group(2)
        elif err_match:
            origin_num = err_match.group(1)
            origin_id = f"O_{origin_num}"
            dest_id = err_match.group(2)
        else:
            continue

        origin_name = origin_map.get(origin_id)
        dest_name = dest_map.get(dest_id)

        # Default record (used for all error cases)
        record = {
            "origin_id": origin_id,
            "origin_name": origin_name,
            "dest_id": dest_id,
            "dest_name": dest_name,
            "mode": "drive",
            "travel_time_min": 0,
            "distance_km": 0,
            "accessible_flag": 0,
            "status_code": None,
            "error_type": None
        }

        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)

            # -------------------------
            # Check if this is an ERROR JSON (from ERROR_ prefixed files)
            # -------------------------
            if fname.startswith("ERROR_"):
                # ERROR files contain HTTP error status
                status = data.get("status", "UNKNOWN")
                record["status_code"] = status
                record["error_type"] = "HTTP_ERROR"

                rows.append(record)
                summary_records.append({
                    "status": f"HTTP_{status}"
                })
                continue

            # -------------------------
            # Normal DRIVE JSON
            # -------------------------
            # Check routing API status (0 = success, non-zero = routing error)
            routing_status = data.get("status")
            
            if routing_status == 0:
                # SUCCESS: Valid route found
                route_summary = data.get("route_summary")

                if route_summary is None:
                    # Unexpected: status=0 but no route_summary
                    record["status_code"] = "NO_ROUTE_SUMMARY"
                    record["error_type"] = "INVALID_JSON"

                    rows.append(record)
                    summary_records.append({
                        "status": "NO_ROUTE_SUMMARY"
                    })
                    continue

                # Extract and normalize fields
                record["travel_time_min"] = (route_summary.get("total_time", 0) or 0) / 60
                record["distance_km"] = (route_summary.get("total_distance", 0) or 0) / 1000

                record["accessible_flag"] = 1
                record["status_code"] = 200
                record["error_type"] = None

                rows.append(record)
                summary_records.append({
                    "status": "OK"
                })
                
            else:
                # ROUTING ERROR: status != 0 (e.g., no route found)
                status_msg = data.get("status_message", "Unknown routing error")
                record["status_code"] = routing_status
                record["error_type"] = f"ROUTING_ERROR: {status_msg}"

                rows.append(record)
                summary_records.append({
                    "status": f"ROUTING_ERROR_{routing_status}"
                })

        except Exception as e:
            record["status_code"] = "EXCEPTION"
            record["error_type"] = type(e).__name__

            rows.append(record)
            summary_records.append({
                "status": f"EXCEPTION_{type(e).__name__}"
            })

# =========================
# Create output DataFrames
# =========================
skims_df = pd.DataFrame(rows)
summary_df = pd.DataFrame(summary_records)

# =========================
# Generate summary statistics
# =========================
print("\n" + "=" * 60)
print("PROCESSING SUMMARY (DRIVE)")
print("=" * 60)

status_summary = (
    summary_df
    .groupby("status")
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)

print("\nOverall Status Summary:")
print(status_summary.to_string(index=False))

final_summary = pd.DataFrame([
    {"category": "Total Records", "count": len(summary_df)},
    {"category": "Successful Routes (OK)", "count": (summary_df["status"] == "OK").sum()},
    {"category": "HTTP 404 Errors", "count": (summary_df["status"] == "HTTP_404").sum()},
    {"category": "HTTP 400 Errors", "count": (summary_df["status"] == "HTTP_400").sum()},
    {"category": "Routing Errors", "count": summary_df["status"].str.contains("ROUTING_ERROR", na=False).sum()},
    {"category": "Exceptions", "count": summary_df["status"].str.contains("EXCEPTION", na=False).sum()},
])

# Add other statuses
other_status = summary_df[
    ~summary_df["status"].isin(["OK", "HTTP_404", "HTTP_400"]) &
    ~summary_df["status"].str.contains("EXCEPTION", na=False) &
    ~summary_df["status"].str.contains("ROUTING_ERROR", na=False)
]

for s in other_status["status"].unique():
    final_summary = pd.concat([
        final_summary,
        pd.DataFrame([{
            "category": f"Other Status: {s}",
            "count": (summary_df["status"] == s).sum()
        }])
    ], ignore_index=True)

print("\nDetailed Summary Table:")
print(final_summary.to_string(index=False))

# =========================
# Save output files
# =========================
skims_df.to_csv(OUTPUT_SKIMS, index=False)
status_summary.to_csv(OUTPUT_SUMMARY, index=False)

print("\n" + "=" * 60)
print("OUTPUT FILES")
print("=" * 60)
print(f"Driving skim table saved to: {OUTPUT_SKIMS}")
print(f"Summary table saved to: {OUTPUT_SUMMARY}")
print(f"Total rows in skim table: {len(skims_df)}")
print("=" * 60)

# =========================
# Additional validation output
# =========================
print("\n" + "=" * 60)
print("DATA QUALITY CHECKS")
print("=" * 60)

accessible_count = (skims_df["accessible_flag"] == 1).sum()
inaccessible_count = (skims_df["accessible_flag"] == 0).sum()

print(f"Accessible OD pairs: {accessible_count:,}")
print(f"Inaccessible OD pairs: {inaccessible_count:,}")

if accessible_count > 0:
    print(f"\nAccessible routes statistics:")
    accessible_routes = skims_df[skims_df["accessible_flag"] == 1]
    print(f"  Mean travel time: {accessible_routes['travel_time_min'].mean():.2f} min")
    print(f"  Mean distance: {accessible_routes['distance_km'].mean():.2f} km")
    print(f"  Max travel time: {accessible_routes['travel_time_min'].max():.2f} min")
    print(f"  Max distance: {accessible_routes['distance_km'].max():.2f} km")

print("=" * 60)
