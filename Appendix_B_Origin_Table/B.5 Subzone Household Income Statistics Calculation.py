"""
B.5 – Subzone Household Income Statistics Calculation

This script calculates Mean and Median monthly household income for each planning area
based on grouped income distribution data from the Singapore Census.

Key steps:
1. Loads household income distribution data from Excel file
2. Defines income group parameters (lower bounds, midpoints, width)
3. Calculates Mean income using group midpoints
4. Calculates Median income using linear interpolation method
5. Outputs results with Mean and Median columns added

Note: 
- 'No Employed Person' households are excluded from Mean calculation
- For open-ended '$20,000 & Over' group, an assumed midpoint is used
- Median calculation handles special cases for high-income groups

Author: Zhang Wenyu
Date: 2025-12-11
"""

import pandas as pd
import numpy as np
from pathlib import Path

# File path for input data (household income distribution by planning area)
FILE_PATH = Path("/Users/zhangwenyu/Desktop/NUSFYP/Stage 1/1.1 Define Origins and Destinations/Tables/subzone_income_allocated.xlsx")

# Output file path (can overwrite original or save as new file)
OUTPUT_FILE_PATH = FILE_PATH.parent / "subzone_income_with_stats.xlsx"

# -------------------------------------------------------------------
# 1. Define Income Group Structure and Parameters
# -------------------------------------------------------------------

"""
Income group definition:
Each tuple contains: (Label, Lower_Bound, Midpoint, Width)
- Label: Column name in the dataset
- Lower_Bound: Minimum value of the income range (for Median calculation)
- Midpoint: Central value of the range (for Mean calculation)
- Width: Range width (for Median interpolation)

Special notes:
- 'No Employed Person': Zero income, excluded from Mean calculation
- 'Below $1,000': Assumed range 0-1000 with midpoint 500
- '$20,000 & Over': Open-ended group with assumed midpoint and width
"""

ASSUMED_MIDPOINT_FOR_20K_PLUS = 25000

INCOME_GROUPS = [
    # (Label, Lower_Bound, Midpoint, Width)
    ('No Employed Person', 0, 0, 0),
    ('Below $1,000', 0, 500, 1000),       # Assumed upper bound: 1000
    ('$1,000 - $1,999', 1000, 1500, 1000),
    ('$2,000 - $2,999', 2000, 2500, 1000),
    ('$3,000 - $3,999', 3000, 3500, 1000),
    ('$4,000 - $4,999', 4000, 4500, 1000),
    ('$5,000 - $5,999', 5000, 5500, 1000),
    ('$6,000 - $6,999', 6000, 6500, 1000),
    ('$7,000 - $7,999', 7000, 7500, 1000),
    ('$8,000 - $8,999', 8000, 8500, 1000),
    ('$9,000 - $9,999', 9000, 9500, 1000),
    ('$10,000 - $10,999', 10000, 10500, 1000),
    ('$11,000 - $11,999', 11000, 11500, 1000),
    ('$12,000 - $12,999', 12000, 12500, 1000),
    ('$13,000 - $13,999', 13000, 13500, 1000),
    ('$14,000 - $14,999', 14000, 14500, 1000),
    ('$15,000 - $17,499', 15000, 16250, 2500),   # Note: width changes here
    ('$17,500 - $19,999', 17500, 18750, 2500),   # Note: width changes here
    ('$20,000 & Over', 20000, ASSUMED_MIDPOINT_FOR_20K_PLUS, 5000)  # Open-ended group, assume midpoint to be S$25000
]

# Extract column names list (excluding 'Total' column)
INCOME_COLUMNS = [group[0] for group in INCOME_GROUPS]

# We'll use both planning_area and subzone as index
INDEX_COLUMNS = ['planning_area', 'subzone']


# -------------------------------------------------------------------
# 2. Helper Functions
# -------------------------------------------------------------------

def calculate_median(income_distribution_series, groups_data):
    """
    Calculates median income using linear interpolation for grouped data.
    
    Formula: Median = L + [ (N/2 - CF) / f ] × w
    Where:
    - L: Lower boundary of median class
    - N: Total number of households
    - CF: Cumulative frequency before median class
    - f: Frequency of median class
    - w: Width of median class
    
    Parameters:
    -----------
    income_distribution_series : pandas.Series
        Household count distribution across income groups for one planning area
    groups_data : list of tuples
        Income group definitions (label, lower_bound, midpoint, width)
    
    Returns:
    --------
    float or str
        Calculated median value (rounded to 2 decimals) or '>$20,000' 
        if median falls in open-ended group
    """
    
    # Total households N
    total_households = income_distribution_series.sum()
    
    # Cumulative frequency tracker
    cumulative_frequency = 0
    
    # Find median class (group containing the N/2-th household)
    for label, lower_bound, midpoint, width in groups_data:
        # Skip 'Total' column if present
        if label == 'Total':
            continue

        # Frequency of current income group
        frequency = income_distribution_series.get(label, 0)
        
        # Check if current group contains the median position
        if cumulative_frequency + frequency >= total_households / 2:
            # Median class found
            lower_boundary = lower_bound
            cumulative_before = cumulative_frequency
            class_width = width
            
            # Special case: median in open-ended '$20,000 & Over' group
            if label == '$20,000 & Over':
                return '>$20,000'  # Indicate median exceeds $20,000
            
            # Apply linear interpolation formula
            median_value = lower_boundary + (
                ((total_households / 2) - cumulative_before) / frequency
            ) * class_width
            
            return round(median_value, 2)
        
        # Update cumulative frequency for next iteration
        cumulative_frequency += frequency
        
    # Should not reach here if data is valid
    return np.nan


# -------------------------------------------------------------------
# 3. Main Processing Pipeline
# -------------------------------------------------------------------

def main():
    try:
        # Step 1: Check if file exists
        if not FILE_PATH.exists():
            print(f"[ERROR] File not found: {FILE_PATH}")
            return
        
        # Step 2: Load data
        print(f"Loading data from: {FILE_PATH}")
        
        # Try to read the Excel file
        excel_file = pd.ExcelFile(FILE_PATH)
        sheet_names = excel_file.sheet_names
        print(f"Available sheets: {sheet_names}")
        
        # Use the first sheet (or specify by name if needed)
        sheet_to_use = sheet_names[0]
        print(f"Using sheet: '{sheet_to_use}'")
        
        data_frame = pd.read_excel(FILE_PATH, sheet_name=sheet_to_use)
        
        # Step 3: Display data info
        print(f"\nData loaded successfully.")
        print(f"Shape: {data_frame.shape}")
        print(f"Columns: {list(data_frame.columns)}")
        print(f"\nFirst few rows:")
        print(data_frame.head())
        
        # Step 4: Check if required columns exist
        missing_cols = []
        for col in INDEX_COLUMNS:
            if col not in data_frame.columns:
                missing_cols.append(col)
        
        if missing_cols:
            print(f"\n[ERROR] Missing required columns: {missing_cols}")
            print("Available columns:", list(data_frame.columns))
            return
        
        # Step 5: Set multi-index (planning_area, subzone)
        print(f"\nSetting multi-index: {INDEX_COLUMNS}")
        data_frame = data_frame.set_index(INDEX_COLUMNS)
        
        # Step 6: Ensure all income columns are numeric
        print("\nConverting income columns to numeric...")
        for column in INCOME_COLUMNS:
            if column in data_frame.columns:
                # Convert to numeric, coerce errors to NaN, then fill NaN with 0
                data_frame[column] = pd.to_numeric(data_frame[column], errors='coerce').fillna(0)
        
        # Also check for 'Total' column
        if 'Total' in data_frame.columns:
            print("Found 'Total' column")
            data_frame['Total'] = pd.to_numeric(data_frame['Total'], errors='coerce').fillna(0)
        
        # Step 7: Calculate Mean Monthly Income
        print("\nCalculating Mean income...")
        data_frame['Mean_Monthly_Income'] = 0.0
        
        # Define groups with actual income (exclude 'No Employed Person')
        income_only_groups = [group for group in INCOME_GROUPS if group[0] != 'No Employed Person']
        
        # Calculate total income (sum of midpoint × frequency)
        total_income = 0
        for label, lower_bound, midpoint, width in income_only_groups:
            if label in data_frame.columns:
                income_contribution = data_frame[label] * midpoint
                data_frame['Mean_Monthly_Income'] += income_contribution
                total_income += income_contribution.sum()
        
        # Calculate total employed households
        employed_columns = [group[0] for group in income_only_groups if group[0] in data_frame.columns]
        data_frame['Total_Employed_Households'] = data_frame[employed_columns].sum(axis=1)
        
        # Calculate Mean = Total Income / Total Employed Households
        # Handle division by zero
        print("Computing mean values:")
        with np.errstate(divide='ignore', invalid='ignore'):
            data_frame['Mean_Monthly_Income'] = (
                data_frame['Mean_Monthly_Income'] / data_frame['Total_Employed_Households']
            ).round(2)
        
        # Replace infinite values with NaN
        data_frame['Mean_Monthly_Income'].replace([np.inf, -np.inf], np.nan, inplace=True)
        
        print(f"Mean calculation completed.")
        
        # Step 8: Calculate Median Monthly Income
        print("\nCalculating Median income:")
        
        # Calculate total households (including 'No Employed Person')
        all_income_columns = [group[0] for group in INCOME_GROUPS if group[0] in data_frame.columns]
        data_frame['Total_Households'] = data_frame[all_income_columns].sum(axis=1)

        # Apply median calculation to each subzone
        print(f"Processing {len(data_frame)} subzones.")
        median_results = []
        
        for i, (index, row) in enumerate(data_frame.iterrows()):
            if (i + 1) % 100 == 0:  # Show progress every 100 subzones
                print(f"  Processed {i + 1}/{len(data_frame)} subzones.")
            
            median_value = calculate_median(
                row[all_income_columns],
                INCOME_GROUPS
            )
            median_results.append(median_value)
        
        data_frame['Median_Monthly_Income'] = median_results
        
        print(f"Median calculation completed.")
        
        # Step 9: Create output DataFrame
        print("\nPreparing output")
        
        # Reset index to have planning_area and subzone as regular columns
        output_data_frame = data_frame.reset_index()
        
        # Reorder columns for better readability
        # Put Mean and Median columns right after subzone
        columns_order = INDEX_COLUMNS + ['Mean_Monthly_Income', 'Median_Monthly_Income']
        
        # Add remaining columns (income distribution columns)
        remaining_cols = [col for col in output_data_frame.columns
                         if col not in columns_order + ['Total_Employed_Households', 'Total_Households']]
        columns_order.extend(remaining_cols)
        
        # Select columns in order
        output_data_frame = output_data_frame[columns_order]
        
        # Save results
        print(f"Saving results to: {OUTPUT_FILE_PATH}")
        with pd.ExcelWriter(OUTPUT_FILE_PATH, engine='openpyxl') as writer:
            output_data_frame.to_excel(writer, sheet_name='Income_Statistics', index=False)
            
            # Also create a summary sheet
            summary_stats = pd.DataFrame({
                'Statistic': ['Number of Subzones',
                             'Average Mean Income',
                             'Minimum Mean Income',
                             'Maximum Mean Income',
                             'Subzones with Median > $20,000'],
                'Value': [len(output_data_frame),
                         f"${output_data_frame['Mean_Monthly_Income'].mean():,.2f}",
                         f"${output_data_frame['Mean_Monthly_Income'].min():,.2f}",
                         f"${output_data_frame['Mean_Monthly_Income'].max():,.2f}",
                         (output_data_frame['Median_Monthly_Income'] == '>$20,000').sum()]
            })
            summary_stats.to_excel(writer, sheet_name='Summary', index=False)
        
        # Summary statistics
        print(f"\n--- Processing Complete ---")
        print(f"Results saved to: {OUTPUT_FILE_PATH}")
        print(f"\nSummary Statistics:")
        print(f"Number of subzones: {len(output_data_frame)}")
        print(f"Average mean income: ${output_data_frame['Mean_Monthly_Income'].mean():,.2f}")
        print(f"Minimum mean income: ${output_data_frame['Mean_Monthly_Income'].min():,.2f}")
        print(f"Maximum mean income: ${output_data_frame['Mean_Monthly_Income'].max():,.2f}")
        
        # Count subzones with median > $20,000
        high_income_count = (output_data_frame['Median_Monthly_Income'] == '>$20,000').sum()
        print(f"Subzones with median > $20,000: {high_income_count}")
        
        print(f"\nFirst 5 subzones:")
        print(output_data_frame[['planning_area', 'subzone', 'Mean_Monthly_Income', 'Median_Monthly_Income']].head())
        
        # Additional: Save a CSV version for easier viewing
        csv_path = OUTPUT_FILE_PATH.with_suffix('.csv')
        output_data_frame.to_csv(csv_path, index=False)
        print(f"\nCSV version also saved to: {csv_path}")
        
        # Show some examples of different median values
        print(f"\nExamples of median values:")
        medians_sample = output_data_frame['Median_Monthly_Income'].value_counts().head(10)
        for value, count in medians_sample.items():
            print(f"  {value}: {count} subzones")
        
    except Exception as error:
        print(f"\n[ERROR] Unexpected error occurred: {error}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
