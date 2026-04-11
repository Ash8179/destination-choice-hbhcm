"""
D.2 - PT Table Construction

Purpose
-------
This script builds a Public Transport (PT) skim table from OneMap routing JSON files.

Key features:
1. Traverses all origin folders (O_XXXX) under the skim directory
2. Parses both successful PT routing JSONs and ERROR JSONs
3. Extracts origin_id and destination_id from file names using regex
4. Joins origin and destination names from O.xlsx and D.xlsx
5. Normalizes units:
   - All time variables are converted from seconds to minutes
   - Distance is converted from meters to kilometers
6. Handles errors robustly:
   - 404 / ERROR / NO_ITINERARY / EXCEPTION entries are retained
   - All numeric skim fields for these cases are filled with 0
7. Outputs:
   - A full PT skim table (long format, OD-level)
   - A detailed summary table counting OK vs error cases
8. Includes progress indicators for large batch processing

The resulting skim table is directly usable for accessibility analysis.

Author: Zhang Wenyu
Date: 2026-01-09
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

OUTPUT_SKIMS = os.path.join(BASE_DIR, "pt_skims_full.csv")
OUTPUT_SUMMARY = os.path.join(BASE_DIR, "pt_skims_summary.csv")

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
# Match normal files: OO_0002_1_ML_0010_PT.json
# Note: File names have double O (OO_) prefix
PT_PATTERN = re.compile(r"O+_(\d+).*?(\d+_[A-Z]+_\d+)_PT\.json")

# Match error files: ERROR_OO_0072_3_HC_0118.json
# Extract the origin number and destination ID
ERROR_PATTERN = re.compile(r"ERROR_O+_(\d+).*?(\d+_[A-Z]+_\d+)\.json")

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

        fpath = os.path.join(origin_path, fname)

        # Try to match normal PT files first
        pt_match = PT_PATTERN.search(fname)
        # Try to match ERROR files
        err_match = ERROR_PATTERN.search(fname)

        origin_id = None
        dest_id = None

        if pt_match:
            # Normal file: construct origin_id as "O_XXXX"
            origin_num = pt_match.group(1)
            origin_id = f"O_{origin_num}"
            dest_id = pt_match.group(2)  # e.g., "1_ML_0087"
        elif err_match:
            # Error file: construct origin_id as "O_XXXX"
            origin_num = err_match.group(1)
            origin_id = f"O_{origin_num}"
            dest_id = err_match.group(2)  # e.g., "3_HC_0118"
        else:
            # File name doesn't match expected patterns, skip
            continue

        origin_name = origin_map.get(origin_id)
        dest_name = dest_map.get(dest_id)

        # Default values (for all error cases)
        record = {
            "origin_id": origin_id,
            "origin_name": origin_name,
            "dest_id": dest_id,
            "dest_name": dest_name,
            "total_time_min": 0,
            "iv_time_min": 0,
            "walk_time_min": 0,
            "wait_time_min": 0,
            "transfers": 0,
            "distance_km": 0,
            "pt_fare_proxy": 0,
            "accessible_flag": 0,
            "error_code": None,
            "fare_status": None  # Track fare availability status
        }

        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)

            # -------------------------
            # ERROR JSON (contains "status" field, typically HTTP errors)
            # -------------------------
            if "status" in data:
                status_code = data.get("status")
                record["error_code"] = f"HTTP_{status_code}"
                rows.append(record)
                summary_records.append({
                    "origin_id": origin_id,
                    "dest_id": dest_id,
                    "status": f"HTTP_{status_code}",
                    "fare_status": None
                })
                continue

            # -------------------------
            # Normal PT JSON (contains routing plan)
            # -------------------------
            itineraries = data.get("plan", {}).get("itineraries", [])

            if len(itineraries) == 0:
                record["error_code"] = "NO_ITINERARY"
                rows.append(record)
                summary_records.append({
                    "origin_id": origin_id,
                    "dest_id": dest_id,
                    "status": "NO_ITINERARY",
                    "fare_status": None
                })
                continue

            # Extract first itinerary (best route)
            itin = itineraries[0]

            # Convert time from seconds to minutes
            record["total_time_min"] = (itin.get("duration", 0) or 0) / 60
            record["iv_time_min"] = (itin.get("transitTime", 0) or 0) / 60
            record["walk_time_min"] = (itin.get("walkTime", 0) or 0) / 60
            record["wait_time_min"] = (itin.get("waitingTime", 0) or 0) / 60

            record["transfers"] = itin.get("transfers", 0) or 0

            # Convert distance from meters to kilometers
            record["distance_km"] = sum(
                leg.get("distance", 0) for leg in itin.get("legs", [])
            ) / 1000

            # Handle fare field - distinguish between numeric values and "info unavailable"
            fare = itin.get("fare")
            if fare is None:
                record["pt_fare_proxy"] = 0
                record["fare_status"] = "NULL"
            elif isinstance(fare, str):
                if "unavailable" in fare.lower():
                    record["pt_fare_proxy"] = 0
                    record["fare_status"] = "INFO_UNAVAILABLE"
                else:
                    try:
                        record["pt_fare_proxy"] = float(fare)
                        record["fare_status"] = "OK"
                    except ValueError:
                        record["pt_fare_proxy"] = 0
                        record["fare_status"] = f"PARSE_ERROR_{fare}"
            else:
                record["pt_fare_proxy"] = float(fare)
                record["fare_status"] = "OK"

            record["accessible_flag"] = 1
            record["error_code"] = "OK"

            rows.append(record)
            summary_records.append({
                "origin_id": origin_id,
                "dest_id": dest_id,
                "status": "OK",
                "fare_status": record["fare_status"]
            })

        except Exception as e:
            # Catch any parsing exceptions
            record["error_code"] = f"EXCEPTION_{type(e).__name__}"
            rows.append(record)
            summary_records.append({
                "origin_id": origin_id,
                "dest_id": dest_id,
                "status": f"EXCEPTION_{type(e).__name__}",
                "fare_status": None
            })

# =========================
# Create output DataFrames
# =========================
skims_df = pd.DataFrame(rows)
summary_df = pd.DataFrame(summary_records)

# =========================
# Generate detailed statistics
# =========================
print("\n" + "="*60)
print("PROCESSING SUMMARY")
print("="*60)

# 1. Overall status summary
status_summary = (
    summary_df
    .groupby("status")
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)
print("\n1. Overall Status Summary:")
print(status_summary.to_string(index=False))

# 2. Fare status summary (for successful routes only)
fare_summary = (
    summary_df[summary_df["status"] == "OK"]
    .groupby("fare_status")
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)
print("\n2. Fare Status Summary (for successful routes only):")
if len(fare_summary) > 0:
    print(fare_summary.to_string(index=False))
else:
    print("No successful routes found")

# 3. Combined summary table
combined_summary = (
    summary_df
    .groupby(["status", "fare_status"])
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)

# 4. Generate final summary table
final_summary = pd.DataFrame([
    {"category": "Total Records", "count": len(summary_df)},
    {"category": "Successful Routes (OK)", "count": (summary_df["status"] == "OK").sum()},
    {"category": "HTTP 404 Errors", "count": (summary_df["status"] == "HTTP_404").sum()},
    {"category": "No Itinerary Found", "count": (summary_df["status"] == "NO_ITINERARY").sum()},
    {"category": "Fare Info Unavailable", "count": (summary_df["fare_status"] == "INFO_UNAVAILABLE").sum()},
    {"category": "Fare Info Available", "count": (summary_df["fare_status"] == "OK").sum()},
    {"category": "Exceptions", "count": summary_df["status"].str.contains("EXCEPTION", na=False).sum()},
])

# Add counts for other error codes
other_errors = summary_df[
    ~summary_df["status"].isin(["OK", "HTTP_404", "NO_ITINERARY"]) &
    ~summary_df["status"].str.contains("EXCEPTION", na=False)
]
if len(other_errors) > 0:
    for status in other_errors["status"].unique():
        count = (summary_df["status"] == status).sum()
        final_summary = pd.concat([
            final_summary,
            pd.DataFrame([{"category": f"Other Error: {status}", "count": count}])
        ], ignore_index=True)

print("\n3. Detailed Summary Table:")
print(final_summary.to_string(index=False))

# =========================
# Save output files
# =========================
skims_df.to_csv(OUTPUT_SKIMS, index=False)
combined_summary.to_csv(OUTPUT_SUMMARY, index=False)

print("\n" + "="*60)
print("OUTPUT FILES")
print("="*60)
print(f"PT skim table saved to: {OUTPUT_SKIMS}")
print(f"Summary table saved to: {OUTPUT_SUMMARY}")
print(f"\nTotal rows in skim table: {len(skims_df)}")
print("="*60)
