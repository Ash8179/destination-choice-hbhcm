"""
B.3 – Subzone Household Size Allocation (Final Corrected Version)

Workflow:
1. Load Table 109 (Households) and Population (Subzones).
2. CLEANING: Remove newline characters ('\n') and extra spaces from Planning Area names 
   in both files to ensure "Bukit \nPanjang" matches "Bukit Panjang".
3. MATCHING: Identify which Planning Areas exist in Table 109. 
   - If a PA exists in Table 109, use specific data.
   - If a PA is in Population data but NOT in Table 109, assign it to "Others".
4. CALCULATION:
   - Weight = (Subzone Population) / (Total Population of that Group).
   - "Group" is either the specific Planning Area or the aggregated "Others" group.
5. OUTPUT: Save the result with clean names (no truncation).

Author: Zhang Wenyu
Date: 2025-12-11
"""

import pandas as pd
import numpy as np
import re

# --------------------------
# Helper Function: Name Cleaner
# --------------------------
def clean_pa_name(name):
    """
    Removes newlines (\n), replaces them with spaces, 
    and removes multiple spaces to ensure standard formatting.
    Example: "Bukit \nPanjang" -> "Bukit Panjang"
    """
    if pd.isna(name):
        return ""
    # Convert to string just in case
    name = str(name)
    # Replace newlines with a single space
    name = name.replace('\n', ' ')
    # Replace multiple spaces with a single space and strip leading/trailing whitespace
    clean_name = re.sub(' +', ' ', name).strip()
    return clean_name

# --------------------------
# Step 1: Load Input Files
# --------------------------
pop_path = "/Users/zhangwenyu/Desktop/NUSFYP/Stage 1/1.1 Define Origins and Destinations/Population.xlsx"
table109_path = "/Users/zhangwenyu/Desktop/NUSFYP/Stage 1/1.1 Define Origins and Destinations/Table 109.csv"

print("----------Loading data----------")

# Load Population Data
# Using read_excel as per your description of .xlsx
pop_df = pd.read_excel(pop_path)
# Normalize column headers to lowercase/snake_case for internal processing
pop_df.columns = [c.strip().lower() for c in pop_df.columns]
# Ensure population is numeric
pop_df['pop_total'] = pd.to_numeric(pop_df['pop_total'], errors='coerce').fillna(0)

# Load Table 109
table109 = pd.read_csv(table109_path)

# --------------------------
# Step 2: Clean and Normalize Planning Area Names
# --------------------------
print("----------Cleaning Planning Area names----------")

# 1. Clean Population DF Planning Areas
# "Bukit \nPanjang" -> "Bukit Panjang" and "Choa Chu \nKang" -> "Choa Chu Kang"
pop_df['planning_area'] = pop_df['planning_area'].apply(clean_pa_name)

# 2. Clean Table 109 Planning Areas
# Rename the first column to a standard key for merging
original_cols = table109.columns.tolist()
pa_col_name = original_cols[0]
table109.rename(columns={pa_col_name: 'PA_Cleaned'}, inplace=True)
table109['PA_Cleaned'] = table109['PA_Cleaned'].apply(clean_pa_name)

# --------------------------
# Step 3: Identify "Others" vs Specific Areas
# --------------------------
# Get list of Planning Areas explicitly listed in Table 109 (excluding the row named "Others")
explicit_pas_in_109 = set(table109[table109['PA_Cleaned'].str.lower() != 'others']['PA_Cleaned'].unique())

# Define Logic: How to link Population Data to Table 109
def get_match_key(pa_name):
    # If the PA exists in Table 109 (e.g., "Ang Mo Kio", "Bukit Panjang"), use it.
    if pa_name in explicit_pas_in_109:
        return pa_name
    # If not (e.g., "Simpang", "Western Water Catchment"), map it to "Others"
    else:
        return 'Others'

# Apply the matching logic
pop_df['Match_Key'] = pop_df['planning_area'].apply(get_match_key)

# Identify which areas were actually grouped into "Others" for verification
others_list = pop_df[pop_df['Match_Key'] == 'Others']['planning_area'].unique()
print(f"\nThe following {len(others_list)} areas were mapped to 'Others':")
print(others_list)

# --------------------------
# Step 4: Merge and Calculate Weights
# --------------------------
# Clean numeric columns in Table 109 (remove commas)
value_cols = [c for c in table109.columns if c != 'PA_Cleaned']
for col in value_cols:
    table109[col] = pd.to_numeric(table109[col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)

# Merge Population data with Table 109 based on the Match_Key
merged_df = pd.merge(pop_df, table109, left_on='Match_Key', right_on='PA_Cleaned', how='left')

# Drop rows where matching failed (should not happen if "Others" row exists in Table 109)
if merged_df['Total'].isnull().any():
    print("Some rows could not be matched.")
    merged_df = merged_df.dropna(subset=['Total'])

# Calculate Group Totals
# If Match_Key is "Ang Mo Kio", sum is total pop of AMK.
# If Match_Key is "Others", sum is total pop of (Simpang + Western Water Catchment + ... etc.)
merged_df['group_pop_sum'] = merged_df.groupby('Match_Key')['pop_total'].transform('sum')

# Calculate Weight
merged_df['weight'] = np.where(
    merged_df['group_pop_sum'] > 0,
    merged_df['pop_total'] / merged_df['group_pop_sum'],
    0
)

# Allocate Households (Vectorized multiplication)
for col in value_cols:
    merged_df[col] = merged_df[col] * merged_df['weight']

# --------------------------
# Step 5: Format and Save Output
# --------------------------
# Select final columns: Planning Area (Original Cleaned Name), Subzone, and Calculated Values
final_cols = ['planning_area', 'subzone'] + value_cols
result_df = merged_df[final_cols].copy()

# Sort for readability
result_df = result_df.sort_values(by=['planning_area', 'subzone'])

# Rounding
result_df[value_cols] = result_df[value_cols].round(4)

print("\nFirst 5 rows of result:")
print(result_df.head())

# Save to CSV
output_path = "/Users/zhangwenyu/Desktop/NUSFYP/Stage 1/1.1 Define Origins and Destinations/subzone_household_size_allocated.csv"
result_df.to_csv(output_path, index=False)

print(f"\nFile saved to: {output_path}")
