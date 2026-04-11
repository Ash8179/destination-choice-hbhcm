"""
E.2 – Postal Accessibility Mapping (JSON Export)

This script:
Merge postal sectors with planning-area accessibility metrics and
export a compact JSON mapping for lookup and downstream use.

Author: Zhang Wenyu
Date: 2026-03-09
"""

import pandas as pd
import json
import numpy as np

# =========================
# 1. Directory
# =========================
postal_file = "/Users/zhangwenyu/Desktop/NUSFYP/Stage 2/Postal Code.xlsx"
p20_file = "/Users/zhangwenyu/Desktop/NUSFYP/Stage 1/1.2 Accessibility (Skims)/P20.csv"

output_json = "/Users/zhangwenyu/Desktop/NUSFYP/Stage 2/postal_accessibility_compact.json"


# =========================
# 2. Read Data
# =========================
postal_df = pd.read_excel(postal_file)
p20_df = pd.read_csv(p20_file)


# =========================
# 3. Unified Row Names
# =========================
postal_df = postal_df.rename(columns={
    "Postal Sector": "postal_sector",
    "Representative Planning Area": "planning_area"
})


# =========================
# 4. Standardize planning_area
# =========================
def clean_name(x):
    if pd.isna(x):
        return x
    return str(x).strip().upper()

postal_df["planning_area"] = postal_df["planning_area"].apply(clean_name)
p20_df["planning_area"] = p20_df["planning_area"].apply(clean_name)


# =========================
# 5. Merge
# =========================
merged_df = pd.merge(
    postal_df,
    p20_df,
    on="planning_area",
    how="left"
)


# =========================
# 6. JSON
# =========================

# planning_area → accessibility
planning_area_data = {}

for _, row in p20_df.iterrows():

    pa = row["planning_area"]

    planning_area_data[pa] = {
        "car_time": None if pd.isna(row["P20_Car_travel_time"]) else float(row["P20_Car_travel_time"]),
        "car_cost": None if pd.isna(row["P20_total_cost_proxy"]) else float(row["P20_total_cost_proxy"]),
        "pt_time": None if pd.isna(row["P20_PT_total_time"]) else float(row["P20_PT_total_time"]),
        "pt_transfers": None if pd.isna(row["P20_PT_transfers"]) else float(row["P20_PT_transfers"]),
        "pt_fare": None if pd.isna(row["P20_PT_fare"]) else float(row["P20_PT_fare"])
    }


# postal_sector → planning_area
postal_map = {}

for _, row in postal_df.iterrows():
    sector = str(row["postal_sector"]).zfill(2)
    postal_map[sector] = row["planning_area"]


result = {
    "postal_to_planning_area": postal_map,
    "planning_area_accessibility": planning_area_data
}


# =========================
# 7. Save JSON
# =========================
with open(output_json, "w") as f:
    json.dump(result, f, indent=4)


# =========================
# 8. Summary
# =========================
missing = merged_df["P20_Car_travel_time"].isna().sum()

print("\n===== SUMMARY =====")

print("Postal sectors:", len(postal_map))
print("Planning areas:", len(planning_area_data))
print("Missing accessibility:", missing)

print("\nExample postal mapping:")
print(list(postal_map.items())[:5])

print("\nExample planning area data:")
print(list(planning_area_data.items())[:3])

print("\nJSON saved to:")
print(output_json)
