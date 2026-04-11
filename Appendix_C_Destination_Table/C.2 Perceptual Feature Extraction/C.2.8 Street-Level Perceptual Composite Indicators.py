"""
=====================================================================
C.2.8 – Street-Level Perceptual Composite Indicators
=====================================================================

Input
-----
perceptual_features_with_meta.csv

Goal
----
Compute five perceptual urban design indicators using weighted
linear combinations of existing perceptual features.

---------------------------------------------------------------------
Dimension 1: Urban Vibrancy
Theory: Jacobs (1961); Montgomery (1998)

vibrancy =
    0.45 * pedestrian_presence +
    0.30 * building_frontage +
    0.20 * signage_density +
    0.05 * activity_diversity

---------------------------------------------------------------------
Dimension 2: Pleasantness
Theory: Kaplan Preference Matrix

pleasantness =
    0.40 * greenery +
    0.30 * sky_visibility +
    0.20 * architectural_variety -
    0.30 * vehicle_presence

---------------------------------------------------------------------
Dimension 3: Walkability
Theory: Speck (2012)

walkability =
    0.35 * ground_surface +
    0.40 * shading_coverage +
    0.15 * greenery -
    0.10 * vehicle_presence

---------------------------------------------------------------------
Dimension 4: Safety
Theory: CPTED

safety =
    0.30 * lighting_presence +
    0.30 * pedestrian_presence +
    0.20 * sky_visibility +
    0.20 * street_furniture

---------------------------------------------------------------------
Dimension 5: Experiential Richness
Theory: Lynch + Mehta

experiential =
    0.45 * architectural_variety +
    0.10 * activity_diversity +
    0.25 * signage_density +
    0.20 * street_furniture

Output
------
perceptual_features_with_dimensions.csv
=====================================================================

Author: Zhang Wenyu
Date: 2026-03-12
"""

# ==========================================================
# Imports
# ==========================================================
import pandas as pd

# ==========================================================
# File paths
# ==========================================================
INPUT_FILE = "/Users/zhangwenyu/Desktop/NUSFYP/Stage 1/1.3 Street Level Imagery/perceptual_features_with_meta.csv"

OUTPUT_FILE = "/Users/zhangwenyu/Desktop/NUSFYP/Stage 1/1.3 Street Level Imagery/perceptual_features_with_dimensions.csv"

# ==========================================================
# Load data
# ==========================================================
df = pd.read_csv(INPUT_FILE)

# ==========================================================
# Dimension 1: Vibrancy
# ==========================================================
df["vibrancy"] = (
    0.45 * df["percept_pedestrian_presence"] +
    0.30 * df["percept_building_frontage"] +
    0.20 * df["percept_signage_density"] +
    0.05 * df["percept_activity_diversity"]
)

# ==========================================================
# Dimension 2: Pleasantness
# ==========================================================
df["pleasantness"] = (
    0.40 * df["percept_greenery"] +
    0.30 * df["percept_sky_visibility"] +
    0.20 * df["percept_architectural_variety"] -
    0.30 * df["percept_vehicle_presence"]
)

# ==========================================================
# Dimension 3: Walkability
# ==========================================================
df["walkability"] = (
    0.35 * df["percept_ground_surface"] +
    0.40 * df["percept_shading_coverage"] +
    0.15 * df["percept_greenery"] -
    0.10 * df["percept_vehicle_presence"]
)

# ==========================================================
# Dimension 4: Safety
# ==========================================================
df["safety"] = (
    0.30 * df["percept_lighting_presence"] +
    0.30 * df["percept_pedestrian_presence"] +
    0.20 * df["percept_sky_visibility"] +
    0.20 * df["percept_street_furniture"]
)

# ==========================================================
# Dimension 5: Experiential Richness
# ==========================================================
df["experiential"] = (
    0.45 * df["percept_architectural_variety"] +
    0.10 * df["percept_activity_diversity"] +
    0.25 * df["percept_signage_density"] +
    0.20 * df["percept_street_furniture"]
)

# ==========================================================
# STANDARDIZATION — Min-Max to [0,1]
# ==========================================================
dimensions = ["vibrancy", "pleasantness", "walkability", "safety", "experiential"]

for col in dimensions:
    min_val = df[col].min()
    max_val = df[col].max()
    
    # Avoid division by zero
    if max_val - min_val == 0:
        df[col + "_std"] = 0
    else:
        df[col + "_std"] = (df[col] - min_val) / (max_val - min_val)

# ==========================================================
# Save output
# ==========================================================
df.to_csv(OUTPUT_FILE, index=False)

print("Finished. File saved to:")
print(OUTPUT_FILE)
