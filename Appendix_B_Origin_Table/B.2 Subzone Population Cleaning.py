"""
B.2 – Subzone Population Cleaning

This script:
1. Loads Table 88 CSV from Census of Population 2020.
2. Removes non-table rows (titles, notes, blanks).
3. Cleans and standardizes column names.
4. Handles missing values in population columns.
5. Outputs a cleaned CSV ready for subzone population integration.

Author: Zhang Wenyu
Date: 2025-12-10
"""

import pandas as pd

# -------------------------------
# Step 1: Load the cleaned CSV
# -------------------------------
csv_path = '/Users/zhangwenyu/Desktop/NUSFYP/Stage 1/1.1 Define Origins and Destinations/Table 88.csv'
cols_to_keep = ['planning_area', 'subzone', 'pop_total']
df = pd.read_csv(csv_path, usecols=cols_to_keep, dtype=str)

# Strip whitespace from column names and lowercase
df.columns = df.columns.str.strip().str.lower()

# -------------------------------
# Step 2: Remove rows where 'subzone' is 'Total'
# -------------------------------
df = df[df['subzone'].str.strip().str.lower() != 'total']

# -------------------------------
# Step 3: Remove completely empty rows
# -------------------------------
df = df.dropna(how='all')

# -------------------------------
# Step 4: Handle missing values in 'pop_total' column
# -------------------------------
if 'pop_total' in df.columns:
    # Replace '-' with 0, remove commas, convert to numeric
    df['pop_total'] = df['pop_total'].replace('-', '0')       # replace '-' with '0'
    df['pop_total'] = df['pop_total'].str.replace(',', '')    # remove commas
    df['pop_total'] = pd.to_numeric(df['pop_total'], errors='coerce').fillna(0).astype(int)

# -------------------------------
# Step 5: Save cleaned CSV
# -------------------------------
output_path = '/Users/zhangwenyu/Desktop/NUSFYP/Stage 1/1.1 Define Origins and Destinations/Population.csv'
df.to_csv(output_path, index=False)

print(f"Final cleaned Table 88 saved to: {output_path}")
print(df.head())
