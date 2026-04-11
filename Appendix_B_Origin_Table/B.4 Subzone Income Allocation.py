"""
B.4 – Subzone Income Distribution Allocation

Description:
This script allocates household income data (Table 110) to subzones based on population weights.
It handles "Others" grouping and fixes text formatting issues (newlines/truncation).

Workflow:
1. Load Population data and Table 110.
2. CLEANING: Remove '\n' and extra spaces from Planning Area names.
3. MATCHING: Map specific PAs to Table 110; map missing PAs to "Others".
4. CALCULATION: Weight = Subzone Pop / Group Total Pop.
5. OUTPUT: Subzone-level income distribution csv.

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
    Removes newlines (\\n), replaces them with spaces, 
    and removes multiple spaces to ensure standard formatting.
    Example: "Bukit \\nPanjang" -> "Bukit Panjang"
    """
    if pd.isna(name):
        return ""
    name = str(name)
    name = name.replace('\n', ' ')
    clean_name = re.sub(' +', ' ', name).strip()
    return clean_name

# --------------------------
# Step 1: Load Input Files
# --------------------------
pop_path = "/Users/zhangwenyu/Desktop/NUSFYP/Stage 1/1.1 Define Origins and Destinations/Population.xlsx"
table110_path = "/Users/zhangwenyu/Desktop/NUSFYP/Stage 1/1.1 Define Origins and Destinations/Table 110.csv"

print("----------Loading data----------")

# Load Population Data
pop_df = pd.read_excel(pop_path)
pop_df.columns = [c.strip().lower() for c in pop_df.columns]
pop_df['pop_total'] = pd.to_numeric(pop_df['pop_total'], errors='coerce').fillna(0)

# Load Table 110
table110 = pd.read_csv(table110_path)

# --------------------------
# Step 2: Clean and Normalize Data
# --------------------------
print("----------Cleaning Planning Area names and Income values----------")

# 1. Clean Population DF Names
pop_df['planning_area'] = pop_df['planning_area'].apply(clean_pa_name)

# 2. Clean Table 110 Names
# Rename first column to 'PA_Cleaned' for merging
original_cols = table110.columns.tolist()
table110.rename(columns={original_cols[0]: 'PA_Cleaned'}, inplace=True)
table110['PA_Cleaned'] = table110['PA_Cleaned'].apply(clean_pa_name)

# 3. Clean Numeric Values in Table 110
# Identify income columns (all columns except the first one 'PA_Cleaned')
income_cols = table110.columns[1:].tolist()

for col in income_cols:
    # Remove commas and convert to numeric
    # Example: "1,563" -> 1563
    table110[col] = pd.to_numeric(table110[col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)

# --------------------------
# Step 3: Identify "Others" vs Specific Areas
# --------------------------
# Find PAs that explicitly exist in Table 110 (excluding the "Others" row itself)
explicit_pas_in_110 = set(table110[table110['PA_Cleaned'].str.lower() != 'others']['PA_Cleaned'].unique())

# Logic: Map Pop Data to Table 110 Keys
def get_match_key(pa_name):
    if pa_name in explicit_pas_in_110:
        return pa_name
    else:
        return 'Others'

pop_df['Match_Key'] = pop_df['planning_area'].apply(get_match_key)

# --------------------------
# Step 4: Merge and Calculate Weights
# --------------------------
# Merge Population data with Table 110
merged_df = pd.merge(pop_df, table110, left_on='Match_Key', right_on='PA_Cleaned', how='left')

# Check for merge errors
if merged_df['Total'].isnull().any():
    print("Some rows could not be matched.")
    merged_df = merged_df.dropna(subset=['Total'])

# Calculate Group Totals (Denominator for weight)
merged_df['group_pop_sum'] = merged_df.groupby('Match_Key')['pop_total'].transform('sum')

# Calculate Weight (Subzone Pop / Group Pop)
merged_df['weight'] = np.where(
    merged_df['group_pop_sum'] > 0,
    merged_df['pop_total'] / merged_df['group_pop_sum'],
    0
)

# Allocate Income Counts (Multiply all income columns by weight)
for col in income_cols:
    merged_df[col] = merged_df[col] * merged_df['weight']

# --------------------------
# Step 5: Format and Save Output
# --------------------------
# Select final columns: Planning Area, Subzone, and all Income Columns
final_cols = ['planning_area', 'subzone'] + income_cols
result_df = merged_df[final_cols].copy()

# Sort
result_df = result_df.sort_values(by=['planning_area', 'subzone'])

# Formatting: Round to 4 decimal places
result_df[income_cols] = result_df[income_cols].round(4)

print("\nFirst 5 rows of result:")
print(result_df.head())

# Save to CSV
output_path = "/Users/zhangwenyu/Desktop/NUSFYP/Stage 1/1.1 Define Origins and Destinations/subzone_income_allocated.csv"
result_df.to_csv(output_path, index=False)

print(f"\nFile saved to: {output_path}")
