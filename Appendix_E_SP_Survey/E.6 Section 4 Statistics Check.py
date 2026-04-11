"""
E.6 – Survey Reliability Analysis (ICC & Krippendorff’s Alpha)

This script:
Evaluate intra-rater consistency and inter-rater reliability of
perceptual ratings using descriptive statistics, ICC (two-way random),
and Krippendorff’s alpha for ordinal data.

Author: Zhang Wenyu
Date: 2026-03-27
"""

import pandas as pd
import numpy as np
import pingouin as pg
import krippendorff

# =========================
# 1. Load and filter data
# =========================
file_path = "/Users/zhangwenyu/Desktop/SurveyV4.csv"
df = pd.read_csv(file_path)

# Filter respondents
df = df[df["Q2_24"] == 4].copy()
print(f"Sample size after filtering: {len(df)}")


# =========================
# 2. Setup
# =========================
dimensions = [
    "Street Liveliness",
    "Environmental Comfort",
    "Pedestrian Accessibility",
    "Perceived Safety",
    "Experiential Offerings"
]

n_images = 12


# =========================
# 3. Intra-rater descriptives
# =========================
print("\n=== Intra-rater Descriptives ===")

intra_summary = []

for d_idx, dim_name in enumerate(dimensions):
    cols = [f"Q4-{i}_{d_idx+1}" for i in range(1, n_images+1)]
    
    values = df[cols].values.flatten()
    values = values[~np.isnan(values)]
    
    mean_val = np.mean(values)
    std_val = np.std(values)
    
    intra_summary.append([dim_name, mean_val, std_val])
    
    print(f"{dim_name}: Mean = {mean_val:.3f}, SD = {std_val:.3f}")


# =========================
# 4. Inter-rater reliability
# =========================
print("\n=== Inter-rater Reliability ===")

results = []

for d_idx, dim_name in enumerate(dimensions):
    
    # Construct long format for ICC
    long_df = []
    
    for img in range(1, n_images+1):
        col = f"Q4-{img}_{d_idx+1}"
        
        temp = df[[col]].copy()
        temp = temp.rename(columns={col: "rating"})
        temp["image"] = img
        temp["rater"] = temp.index
        
        long_df.append(temp)
    
    long_df = pd.concat(long_df, axis=0)
    
    # Drop missing
    long_df = long_df.dropna(subset=["rating"])
    
    # =========================
    # ICC calculation (two-way random, absolute agreement)
    # =========================
    icc_table = pg.intraclass_corr(
        data=long_df,
        targets="image",
        raters="rater",
        ratings="rating"
    )
    
    # Select ICC2 (commonly reported)
    print(icc_table[["Type", "ICC"]])

    # Prefer ICC2, fallback to ICC2k if needed
    if "ICC2" in icc_table["Type"].values:
        icc2 = icc_table.loc[icc_table["Type"] == "ICC2", "ICC"].values[0]
    elif "ICC2k" in icc_table["Type"].values:
        icc2 = icc_table.loc[icc_table["Type"] == "ICC2k", "ICC"].values[0]
    else:
        icc2 = np.nan
        print(f"Warning: ICC2 not found for {dim_name}")
    
    # =========================
    # Krippendorff’s alpha
    # =========================
    # Build matrix: rows = raters, cols = images
    matrix = []
    
    for idx, row in df.iterrows():
        ratings = []
        for img in range(1, n_images+1):
            ratings.append(row[f"Q4-{img}_{d_idx+1}"])
        matrix.append(ratings)
    
    matrix = np.array(matrix)
    
    # Krippendorff alpha (ordinal scale recommended for Likert)
    alpha = krippendorff.alpha(
        reliability_data=matrix,
        level_of_measurement='ordinal'
    )
    
    results.append([dim_name, icc2, alpha])
    
    print(f"{dim_name}: ICC(2,1) = {icc2:.3f}, Krippendorff α = {alpha:.3f}")


# =========================
# 5. Save results
# =========================
results_df = pd.DataFrame(results, columns=[
    "Dimension", "ICC(2,1)", "Krippendorff_alpha"
])

output_path = "/Users/zhangwenyu/Desktop/Q4_reliability_results.csv"
results_df.to_csv(output_path, index=False)

print(f"\nResults saved to: {output_path}")


# =========================
# 6. Interpretation guide
# =========================
print("\n=== Interpretation Guide ===")

print("\nICC (Koo & Li, 2016):")
print("  < 0.50  → poor")
print("  0.50–0.75 → moderate")
print("  0.75–0.90 → good")
print("  > 0.90 → excellent")

print("\nKrippendorff’s α:")
print("  < 0.667 → insufficient")
print("  0.667–0.80 → tentative")
print("  > 0.80 → reliable")
