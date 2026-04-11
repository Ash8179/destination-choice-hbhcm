"""
B.8 - Origin Table Construction with Missing Value Reporting

This script constructs the Origin Table for all subzones, integrating:
- Spatial info from subzone_centroids.geojson
- Population data (subzone-level) from Population.xlsx
- Planning-area-level socioeconomic & Car Proxy PCA data from car_proxy_pca.csv
- Household structure from Table 109.csv

It calculates additional variables:
- population_density
- PCA_Car_Proxy scaled to 0-1
- Missing value reporting (overall + by group + critical keys)

Author: Zhang Wenyu
Date: 2025-12-12
"""

import pandas as pd
import json
import numpy as np
import os

# ======================================================
# CONFIGURATION: FILE PATHS
# ======================================================
BASE_DIR = "/Users/zhangwenyu/Desktop/NUSFYP/Stage 1/1.1 Define Origins and Destinations"
GEOJSON_PATH = os.path.join(BASE_DIR, "geojson/Processed_Output/subzone_centroids.geojson")
POP_PATH = os.path.join(BASE_DIR, "Tables/Population.xlsx")
CAR_PCA_PATH = os.path.join(BASE_DIR, "Tables/car_proxy_pca.csv")
HH_PATH = os.path.join(BASE_DIR, "Tables/Table 109.csv")
OUTPUT_PATH = "/Users/zhangwenyu/Desktop/NUSFYP/Origins.csv"

# ======================================================
# 1. LOAD GEOJSON (SPATIAL BASE)
# ======================================================
print("="*70)
print("1. LOADING GEOJSON")
print("="*70)
with open(GEOJSON_PATH, "r") as f:
    gj = json.load(f)

# Extract properties from features to create the base DataFrame
geo_records = [feat["properties"] for feat in gj["features"]]
geo = pd.DataFrame(geo_records)

# Ensure required columns exist in GeoJSON
print(f"GeoJSON columns: {list(geo.columns)}")
print(f"GeoJSON shape: {geo.shape}")

if 'subzone_no' in geo.columns:
    geo.rename(columns={'subzone_no': 'subzone_id'}, inplace=True)
    print("Renamed 'subzone_no' to 'subzone_id'")

# Select only necessary columns from GeoJSON for Group A
required_geo_columns = ['subzone_id', 'subzone_name', 'area_sqm', 'centroid_x', 'centroid_y']
geo = geo[[col for col in required_geo_columns if col in geo.columns]].copy()
print(f"Selected GeoJSON columns: {list(geo.columns)}")

# ======================================================
# 2. LOAD OTHER DATASETS
# ======================================================
print("\n" + "="*70)
print("2. LOADING TABULAR DATA")
print("="*70)

# 2.1 Load Population data
print("Loading Population data...")
population = pd.read_excel(POP_PATH, sheet_name="Population")
population.columns = [c.strip() for c in population.columns]
print(f"Population data shape: {population.shape}")
print(f"Population columns: {list(population.columns)}")

# Standardize subzone names (convert to uppercase)
population['subzone'] = population['subzone'].str.upper().str.strip()
geo['subzone_name'] = geo['subzone_name'].str.upper().str.strip()

# 2.2 Load Car Proxy PCA data
print("\nLoading Car Proxy PCA data...")
car_pca = pd.read_csv(CAR_PCA_PATH)
print(f"Car PCA shape: {car_pca.shape}")
print(f"Car PCA columns: {list(car_pca.columns)}")

# 2.3 Load Household data
print("\nLoading Household data...")
hh = pd.read_csv(HH_PATH)
print(f"Household data shape: {hh.shape}")
print(f"Household columns: {list(hh.columns)}")

# ======================================================
# 3. PREPROCESS HOUSEHOLD DATA
# ======================================================
print("\n" + "="*70)
print("3. PREPROCESSING HOUSEHOLD DATA")
print("="*70)

# Rename columns
hh_rename_map = {
    "Planning Area of Residence": "planning_area",
    "Total": "total_hh_count",
    "1 Person": "hh_1p",
    "2 Persons": "hh_2p",
    "3 Persons": "hh_3p",
    "4 Persons": "hh_4p",
    "5 Persons": "hh_5p",
    "6 Persons": "hh_6p",
    "7 Persons": "hh_7p",
    "8 or More Persons": "hh_8p_plus",
    "Total Number of population": "hh_total_population",
    "avg_hh_size": "avg_hh_size"
}
hh = hh.rename(columns=hh_rename_map)

# Clean numeric columns (remove commas, convert to numbers)
for col in ['hh_1p', 'hh_2p', 'hh_3p', 'hh_4p', 'hh_5p', 'hh_6p', 'hh_7p',
            'hh_8p_plus', 'total_hh_count', 'hh_total_population', 'avg_hh_size']:
    if col in hh.columns:
        hh[col] = hh[col].astype(str).str.replace(',', '').str.replace(' ', '')
        hh[col] = pd.to_numeric(hh[col], errors='coerce')

print(f"Household columns after processing: {list(hh.columns)}")

# ======================================================
# 4. MERGE DATASETS
# ======================================================
print("\n" + "="*70)
print("4. MERGING DATASETS")
print("="*70)

# 4.1 First merge Population data to GeoJSON
df = geo.copy()
print(f"Initial dataframe shape: {df.shape}")

# Merge Population data (based on subzone_name)
df = df.merge(
    population[["subzone", "planning_area", "pop_total"]],
    left_on="subzone_name",
    right_on="subzone",
    how="left"
)

# Check merge results
print(f"After population merge: {df.shape}")
print(f"Rows with planning_area: {df['planning_area'].notna().sum()}/{len(df)}")
print(f"Rows with pop_total: {df['pop_total'].notna().sum()}/{len(df)}")

# Remove redundant column
if 'subzone' in df.columns:
    df = df.drop(columns=['subzone'])

# 4.2 Calculate population density
df["population_density"] = df["pop_total"] / df["area_sqm"]

# 4.3 Merge Car PCA data (based on planning_area)
# Standardize planning area names (all uppercase)
df['planning_area'] = df['planning_area'].str.upper().str.strip()
car_pca['planning_area'] = car_pca['planning_area'].str.upper().str.strip()

df = df.merge(car_pca, on="planning_area", how="left")
print(f"After Car PCA merge: {df.shape}")
print(f"Rows with Car PCA data: {df['PCA_Car_Proxy'].notna().sum()}/{len(df)}")

# 4.4 Merge Household data (based on planning_area)
# Standardize planning area names
hh['planning_area'] = hh['planning_area'].str.upper().str.strip()

# Select only household structure columns
hh_columns_to_merge = ['planning_area'] + [col for col in ['hh_1p', 'hh_2p', 'hh_3p', 'hh_4p', 'hh_5p',
                                                           'hh_6p', 'hh_7p', 'hh_8p_plus', 'avg_hh_size']
                                           if col in hh.columns]

df = df.merge(hh[hh_columns_to_merge], on="planning_area", how="left")
print(f"After Household merge: {df.shape}")
print(f"Rows with Household data: {df['avg_hh_size'].notna().sum()}/{len(df)}")

# ======================================================
# 5. FEATURE ENGINEERING & COLUMN RENAMING
# ======================================================
print("\n" + "="*70)
print("5. FEATURE ENGINEERING")
print("="*70)

# 5.1 Rename VehicleExp_p column if it has a different name
if 'VehicleExp_p (Bucket Value)' in df.columns and 'VehicleExp_p' not in df.columns:
    df = df.rename(columns={'VehicleExp_p (Bucket Value)': 'VehicleExp_p'})
    print("Renamed 'VehicleExp_p (Bucket Value)' to 'VehicleExp_p'")

# 5.2 Calculate PCA_Car_Proxy_01 (Min-Max Scaling)
pca_col = "PCA_Car_Proxy"
if pca_col in df.columns:
    if df[pca_col].notna().any():
        pca_min = df[pca_col].min()
        pca_max = df[pca_col].max()
        if pca_max != pca_min:  # Avoid division by zero
            df["PCA_Car_Proxy_01"] = (df[pca_col] - pca_min) / (pca_max - pca_min)
            print(f"Created PCA_Car_Proxy_01 (range: {pca_min:.4f} to {pca_max:.4f})")
        else:
            df["PCA_Car_Proxy_01"] = 0.5  # Set to middle if all values are same
            print("All PCA_Car_Proxy values are identical, set PCA_Car_Proxy_01 to 0.5")
    else:
        print(f"PCA_Car_Proxy column exists but all values are NaN")
        df["PCA_Car_Proxy_01"] = None
else:
    print(f"{pca_col} not found in dataframe")

# =======================================================
# 6. FINALIZE COLUMNS (REORDER & SELECT)
# =======================================================
print("\n" + "="*70)
print("6. FINAL COLUMNS")
print("="*70)

# Define the exact column order based on Groups A-F
final_columns = [
    # Group A: Spatial
    "subzone_id", "subzone_name", "planning_area", "area_sqm", "centroid_x", "centroid_y",
    # Group B: Population
    "pop_total", "population_density",
    # Group C: Income
    "Mean_Monthly_Income", "Median_Monthly_Income", "Median_Monthly_Income_Percentile",
    # Group D: Expenditure
    "Interpolated_Expenditure", "VehicleExp_p", "Interpolated_VehicleExp_p",
    # Group E: Car Proxies
    "Car Proxy 1 (Ownership%)", "Car Proxy 2 (Economic Parameter)", "PCA_Car_Proxy", "PCA_Car_Proxy_01",
    # Group F: Household
    "hh_1p", "hh_2p", "hh_3p", "hh_4p", "hh_5p", "hh_6p", "hh_7p", "hh_8p_plus", "avg_hh_size"
]

print(f"Current dataframe columns: {list(df.columns)}")

# Check for missing columns
missing_cols = set(final_columns) - set(df.columns)
available_cols = [c for c in final_columns if c in df.columns]

print(f"\nAvailable columns: {len(available_cols)}/{len(final_columns)}")
print(f"Missing columns: {missing_cols}")

# Create final DataFrame with ordered columns
final_df = df[available_cols].copy()

# =====================================================
# 7. MISSING VALUE REPORT
# =====================================================
print("\n" + "="*70)
print("7. MISSING VALUE REPORT")
print("="*70)

if not final_df.empty:
    # Overall missing values
    missing_counts = final_df.isna().sum()
    missing_percentage = (missing_counts / len(final_df) * 100).round(2)
    
    print(f"DataFrame shape: {final_df.shape}")
    print(f"\nMissing values by column (count and %):")
    print("-" * 60)
    
    # Group columns by category for better reporting
    column_groups = {
        "Group A: Spatial": ["subzone_id", "subzone_name", "planning_area", "area_sqm", "centroid_x", "centroid_y"],
        "Group B: Population": ["pop_total", "population_density"],
        "Group C: Income": ["Mean_Monthly_Income", "Median_Monthly_Income", "Median_Monthly_Income_Percentile"],
        "Group D: Expenditure": ["Interpolated_Expenditure", "VehicleExp_p", "Interpolated_VehicleExp_p"],
        "Group E: Car Proxies": ["Car Proxy 1 (Ownership%)", "Car Proxy 2 (Economic Parameter)", "PCA_Car_Proxy", "PCA_Car_Proxy_01"],
        "Group F: Household": ["hh_1p", "hh_2p", "hh_3p", "hh_4p", "hh_5p", "hh_6p", "hh_7p", "hh_8p_plus", "avg_hh_size"]
    }
    
    for group_name, group_cols in column_groups.items():
        print(f"\n{group_name}:")
        for col in group_cols:
            if col in final_df.columns:
                count = missing_counts[col]
                percent = missing_percentage[col]
                status = "✓" if count == 0 else f"{count} ({percent}%)"
                print(f"  {col:<35} {status}")
            else:
                print(f"  {col:<35} Column not found!")
    
    # Summary statistics
    print(f"\nSUMMARY STATISTICS:")
    print(f"Total rows: {len(final_df)}")
    total_columns = len(final_df.columns)
    complete_columns = (missing_counts == 0).sum()
    print(f"Complete columns (0% missing): {complete_columns}/{total_columns} ({complete_columns/total_columns*100:.1f}%)")
    print(f"Partial columns (1-50% missing): {(missing_percentage.between(1, 50)).sum()}")
    print(f"Mostly missing columns (51-99% missing): {(missing_percentage.between(51, 99)).sum()}")
    print(f"Completely missing columns (100% missing): {(missing_percentage == 100).sum()}")
    
    # Show first few rows with data
    print(f"\nSAMPLE DATA (first 3 rows with available data):")
    # Find columns that have some data
    cols_with_data = [col for col in final_df.columns if final_df[col].notna().any()]
    if cols_with_data:
        sample_df = final_df[cols_with_data].head(3)
        print(sample_df.to_string())
    else:
        print("No data available in any columns!")
        
else:
    print("DataFrame is empty!")

# =====================================================
# 8. SAVE OUTPUT
# =====================================================
print("\n" + "="*70)
print("8. SAVING OUTPUT")
print("="*70)

try:
    final_df.to_csv(OUTPUT_PATH, index=False, float_format="%.15f")
    print(f"Processing Complete. Origin Table saved to:\n{OUTPUT_PATH}")
    print(f"Final Shape: {final_df.shape}")
    
    # Also save a debug version with all columns
    debug_path = OUTPUT_PATH.replace(".csv", "_FULL.csv")
    df.to_csv(debug_path, index=False)
    print(f"Full version with all columns saved to:\n{debug_path}")
    
    # Save missing value report
    report_path = OUTPUT_PATH.replace(".csv", "_MISSING_REPORT.csv")
    missing_report = pd.DataFrame({
        'column': missing_counts.index,
        'missing_count': missing_counts.values,
        'missing_percentage': missing_percentage.values,
        'data_type': final_df.dtypes.values
    })
    missing_report = missing_report.sort_values('missing_percentage', ascending=False)
    missing_report.to_csv(report_path, index=False)
    print(f"Missing value report saved to:\n{report_path}")
    
except Exception as e:
    print(f"Error saving output: {e}")

print("="*70)
