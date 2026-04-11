import json
import pandas as pd
import uuid

# =========================
# File paths
# =========================
JSON_PATH = "/Users/zhangwenyu/Desktop/NUSFYP/Stage 1/1.1 Define Origins and Destinations/Destinations/Data/json/onemap_museum.json"
EXCEL_PATH = "/Users/zhangwenyu/Desktop/NUSFYP/Stage 1/1.1 Define Origins and Destinations/Destinations/Destination.xlsx"

# =========================
# Load data
# =========================
with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

df_dest = pd.read_excel(EXCEL_PATH)

# =========================
# Parse POIs (skip metadata)
# =========================
records = []

for item in data["SrchResults"]:
    # Skip metadata rows
    if "NAME" not in item or "LatLng" not in item:
        continue

    # Parse LatLng string: "lat,lon"
    lat_str, lon_str = item["LatLng"].split(",")
    lat = float(lat_str)
    lon = float(lon_str)

    record = {
        # ---------- Group A ----------
        "poi_id": f"MU_{uuid.uuid4().hex[:8]}",
        "name": item.get("NAME"),
        "category": "leisure",
        "subtype": "museum",

        # ---------- Group B ----------
        "centroid_x": lat,
        "centroid_y": lon,

        # ---------- Group G ----------
        "experiential_flag": 1,
        "heritage_flag": 1,
        "night_only_flag": 0
    }

    records.append(record)

df_new = pd.DataFrame(records)

# =========================
# Append to destination table
# =========================
df_updated = pd.concat([df_dest, df_new], ignore_index=True)

# =========================
# Save
# =========================
df_updated.to_excel(EXCEL_PATH, index=False)

print("Museums successfully added to destination table.")
print(f"Number of museums added: {len(df_new)}")
