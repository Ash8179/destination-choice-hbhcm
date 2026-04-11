"""
D.5 - Cost Proxy Construction

Drive Cost Proxy Construction
=============================

Purpose
-------
This script constructs a generalized monetary cost proxy for car travel
using the Singapore drive skim matrix. The original skim dataset contains
travel time and distance but does not include monetary cost components such
as ERP tolls or parking fees. To approximate realistic driving costs,
a composite proxy is constructed.

Notation
--------
Let i denote the origin zone and j denote the destination zone.

Each zone is classified into three hierarchical spatial categories:

Z = 2 : Central Business District (CBD)
Z = 1 : Central Area excluding CBD
Z = 0 : Outside Central Area

The zone level is derived from binary flags:

CBD_z : CBD indicator for zone z
CA_z  : Central Area indicator for zone z

The hierarchical classification rule is:

Z_z =
    2 if CBD_z = 1
    1 if CA_z = 1 and CBD_z = 0
    0 otherwise


Generalized Cost Proxy
----------------------

The generalized driving monetary cost between origin i and destination j
is defined as:

C_ij = C_dist_ij + C_erp_ij + C_park_j


1. Distance-based vehicle operating cost

C_dist_ij = α * d_ij

where:
d_ij = travel distance in km
α = 0.25 SGD/km

This parameter represents fuel consumption, vehicle maintenance,
and depreciation.


2. ERP proxy (Electronic Road Pricing)

ERP is approximated based on the spatial transition between origin and
destination zones.

Origin zone -> Destination zone -> ERP proxy

0 -> 1 : 2.0 SGD
0 -> 2 : 3.0 SGD
1 -> 2 : 1.5 SGD
2 -> 2 : 1.0 SGD
1 -> 1 : 0.5 SGD
otherwise : 0

This structure approximates congestion pricing when entering the
central area and CBD.


3. Parking fee proxy

Parking cost depends on the destination zone.

Destination zone -> Parking proxy

2 (CBD)            : 6 SGD
1 (Central Area)   : 4 SGD
0 (Outside CA)     : 1.5 SGD


Final Cost Proxy
----------------

The final proxy used in the drive skim matrix is:

C_ij = 0.25 * distance_km + ERP_proxy + Parking_proxy


Input Files
-----------

O_with_flags.xlsx
    Origin zones with CBD and Central Area flags

D.xlsx
    Destination zones with CBD and Central Area flags

drive_skims_full.csv
    Drive skim matrix containing travel time and distance


Output
------

drive_skims_with_cost_proxy.csv

The output dataset includes the following additional fields:

origin_zone
dest_zone
distance_cost
erp_proxy
park_fee_proxy
total_cost_proxy

Author: Zhang Wenyu
Date: 2026-01-16
"""

import pandas as pd
from pathlib import Path

# ==============================
# File paths
# ==============================

base = Path("/Users/zhangwenyu/Desktop/NUSFYP/Stage 1/1.2 Accessibility (Skims)")

origin_file = base / "O_with_flags.xlsx"
dest_file = base / "D.xlsx"
skim_file = base / "Drive/drive_skims_full.csv"

output_file = base / "Drive/drive_skims_with_cost_proxy.csv"

# ==============================
# Load datasets
# ==============================

skims = pd.read_csv(skim_file)
orig = pd.read_excel(origin_file)
dest = pd.read_excel(dest_file)

# Standardize column names
orig.columns = orig.columns.str.strip().str.lower()
dest.columns = dest.columns.str.strip().str.lower()

# ==============================
# Merge origin flags
# ==============================

orig_flags = orig[['origin_id','cbd_flag','central_area_flag']].rename(
    columns={
        'cbd_flag':'origin_cbd_flag',
        'central_area_flag':'origin_central_area_flag'
    }
)

skims = skims.merge(orig_flags, on="origin_id", how="left")

# ==============================
# Merge destination flags (poi_id -> dest_id)
# ==============================

dest_flags = dest[['poi_id','cbd_flag','central_area_flag']].rename(
    columns={
        'poi_id':'dest_id',
        'cbd_flag':'dest_cbd_flag',
        'central_area_flag':'dest_central_area_flag'
    }
)

skims = skims.merge(dest_flags, on="dest_id", how="left")

# ==============================
# Zone classification
# ==============================

def zone(cbd, central):
    if cbd == 1:
        return 2
    elif central == 1:
        return 1
    else:
        return 0

skims["origin_zone"] = skims.apply(
    lambda r: zone(r.origin_cbd_flag, r.origin_central_area_flag), axis=1
)

skims["dest_zone"] = skims.apply(
    lambda r: zone(r.dest_cbd_flag, r.dest_central_area_flag), axis=1
)

# ==============================
# Distance operating cost
# ==============================

skims["distance_cost"] = skims["distance_km"] * 0.25

# ==============================
# ERP proxy
# ==============================

def erp_cost(o, d):

    if o == 0 and d == 1:
        return 2.0
    if o == 0 and d == 2:
        return 3.0
    if o == 1 and d == 2:
        return 1.5
    if o == 2 and d == 2:
        return 1.0
    if o == 1 and d == 1:
        return 0.5

    return 0.0

skims["erp_proxy"] = skims.apply(
    lambda r: erp_cost(r.origin_zone, r.dest_zone), axis=1
)

# ==============================
# Parking proxy
# ==============================

def parking_cost(d):

    if d == 2:
        return 6
    if d == 1:
        return 4
    return 1.5

skims["park_fee_proxy"] = skims["dest_zone"].apply(parking_cost)

# ==============================
# Total cost proxy
# ==============================

skims["total_cost_proxy"] = (
    skims["distance_cost"]
    + skims["erp_proxy"]
    + skims["park_fee_proxy"]
)

# ==============================
# Save output
# ==============================

skims.to_csv(output_file, index=False)

print("Finished.")
print("Output saved to:", output_file)
print("Total rows:", len(skims))
