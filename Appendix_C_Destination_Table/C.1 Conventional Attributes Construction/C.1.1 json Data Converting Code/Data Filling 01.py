import json
import pandas as pd
import uuid

# =========================
# File paths
# =========================
GEOJSON_PATH = "/Users/zhangwenyu/Desktop/NUSFYP/Stage 1/1.1 Define Origins and Destinations/Destinations/Data/HawkerCentresGEOJSON.geojson"
EXCEL_PATH = "/Users/zhangwenyu/Desktop/NUSFYP/Stage 1/1.1 Define Origins and Destinations/Destinations/Destination.xlsx"

# =========================
# Load data
# =========================
with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
    geojson = json.load(f)

df_dest = pd.read_excel(EXCEL_PATH)

# =========================
# Prepare new records
# =========================
records = []

for feature in geojson["features"]:
    props = feature["properties"]
    coords = feature["geometry"]["coordinates"]

    record = {
        # ---------- Group A ----------
        "poi_id": f"HC_{uuid.uuid4().hex[:8]}",
        "name": props.get("NAME"),
        "category": "community retail",
        "subtype": "hawker centre",

        # ---------- Group B ----------
        # GeoJSON format: [longitude, latitude]
        "centroid_x": coords[1],  # latitude
        "centroid_y": coords[0],  # longitude

        # ---------- Group C ----------
        "fnb_count": props.get("NUMBER_OF_COOKED_FOOD_STALLS"),
        "retail_count": 0,
        "entertainment_count": 0,
        "has_foodcourt": 1,

        # ---------- Group G ----------
        "experiential_flag": 1,
        "heritage_flag": 0,
        "night_only_flag": 0
    }

    records.append(record)

df_new = pd.DataFrame(records)

# =========================
# Merge into destination table
# =========================
# Only update matching columns, keep all others unchanged
df_updated = pd.concat([df_dest, df_new], ignore_index=True)

# =========================
# Save back to Excel
# =========================
df_updated.to_excel(EXCEL_PATH, index=False)

print("Hawker centres successfully added to destination table.")
print(f"Number of hawker centres added: {len(df_new)}")
