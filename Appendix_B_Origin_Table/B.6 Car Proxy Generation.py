"""
B.6 – Car Proxy Generation

This script performs the following tasks:

1. Load planning area income data
    Reads subzone_income_with_stats.csv
    Extracts Median_Monthly_Income for each planning area

2. Compute Income Percentiles
    Uses piecewise linear interpolation between known percentile–income points
    Outputs: Median_Monthly_Income_Percentile

3. Compute Expenditure from Income
    Uses income → expenditure mapping via interpolation
    Outputs: Interpolated_Expenditure

4. Compute Car Ownership Probability (Car Proxy 1)
    Based on Expenditure → Ownership% mapping
    Uses piecewise linear interpolation between expenditure buckets
    Outputs:
       • Ownership (bucket value)
       • Interpolated_Ownership
       • Car Proxy 1 (Ownership%)

5. Compute VehicleExpenditure Parameter (Car Proxy 2)
    Based on income percentile → vehicle expenditure bucket
    Uses piecewise linear interpolation
    Outputs:
       • VehicleExp_p (bucket value)
       • Interpolated_VehicleExp_p
       • Car Proxy 2 (Economic Parameter)

6. Save final output CSV

Final columns:
    planning_area
    Median_Monthly_Income
    Median_Monthly_Income_Percentile
    Interpolated_Income
    Interpolated_Expenditure
    VehicleExp_p
    Ownership
    Car Proxy 1 (Ownership%)
    Car Proxy 2 (Economic Parameter)

Author: Zhang Wenyu
Date: 2025-12-11
"""

import pandas as pd
import numpy as np

# -------------------------------------------------------------
# Helper function: piecewise linear interpolation
# -------------------------------------------------------------
def piecewise_interpolate(x, xp, fp, enforce_monotonic=False, clip=True):
    """
    Piecewise linear interpolation wrapper.

    Parameters
    ----------
    x : scalar or array-like
        Query point(s) to interpolate.
    xp : 1-D array-like (sorted ascending)
        Breakpoints (x-coordinates).
    fp : 1-D array-like
        Function values at xp (same length as xp).
    enforce_monotonic : bool, optional (default False)
        If True, forces fp to be non-decreasing by applying a
        monotonic (isotonic-like) correction using cumulative maximum.
        Use this if you want to guarantee a monotone result even if fp has noise.
    clip : bool, optional (default True)
        If True, values of x outside [xp[0], xp[-1]] are clipped to that range
        so the returned value equals fp[0] or fp[-1]. If False, np.interp's default
        left/right behavior is used (which can be specified separately).

    Returns
    -------
    scalar or ndarray
        Interpolated value(s), matching input shape of x.
    """
    xp = np.asarray(xp, dtype=float)
    fp = np.asarray(fp, dtype=float)

    if xp.ndim != 1 or fp.ndim != 1:
        raise ValueError("xp and fp must be 1-D sequences.")
    if xp.size != fp.size:
        raise ValueError("xp and fp must have the same length.")
    if np.any(np.diff(xp) <= 0):
        raise ValueError("xp must be strictly increasing (sorted ascending).")

    if enforce_monotonic:
        # simple monotone correction: cumulative maximum (non-decreasing)
        fp = np.maximum.accumulate(fp)

    x_arr = np.asarray(x, dtype=float)
    if clip:
        # clip x to the xp range to avoid extrapolation artifacts
        x_clipped = np.clip(x_arr, xp[0], xp[-1])
        out = np.interp(x_clipped, xp, fp)
    else:
        out = np.interp(x_arr, xp, fp)

    # return scalar if input was scalar
    if np.isscalar(x):
        return float(out)
    return out


# -------------------------------------------------------------
# Load planning area income data
# -------------------------------------------------------------
income_path = "/Users/zhangwenyu/Desktop/NUSFYP/Stage 1/1.1 Define Origins and Destinations/Tables/subzone_income_with_stats.csv"
df = pd.read_csv(income_path)

# -------------------------------------------------------------
# 0. Handle Zero-value
# -------------------------------------------------------------
# Convert 0 to NaN in Median_Monthly_Income to exclude them from calculations that treat 0 as a valid low income.
df['Income_Cleaned'] = df["Median_Monthly_Income"].replace(0, np.nan)

# -------------------------------------------------------------
# 1. Income percentile → income mapping
# -------------------------------------------------------------
percentiles = np.array([0, 20, 40, 60, 80, 100])
income_points = np.array([0,
                          1921.009795,   # 20th Percentile Boundary (P20)
                          5744.422399,   # 40th Percentile Boundary (P40)
                          9972.452913,   # 60th Percentile Boundary (P60)
                          15623.84812,   # 80th Percentile Boundary (P80)
                          25000])        # Use 25000 as the cap for P100

# compute percentile from income
df["Median_Monthly_Income_Percentile"] = df["Income_Cleaned"].apply(
    lambda inc: piecewise_interpolate(inc, income_points, percentiles)
)

# -------------------------------------------------------------
# 2. Income → expenditure mapping
# -------------------------------------------------------------
expenditure_points = np.array([
    0,
    4149.4,   # 20th Percentile Boundary (P20)
    5381.3,   # 40th Percentile Boundary (P40)
    6985.6,   # 60th Percentile Boundary (P60)
    8257.2,   # 80th Percentile Boundary (P80)
    10820.1   # 100th Percentile Boundary (P100)
])

df["Interpolated_Expenditure"] = df["Income_Cleaned"].apply(
    lambda inc: piecewise_interpolate(inc, income_points, expenditure_points)
)

# -------------------------------------------------------------
# 3. Expenditure → Ownership mapping (Car Proxy 1)
# -------------------------------------------------------------
"""
This section uses the calculated Interpolated_Expenditure (a dollar amount) to estimate the corresponding Household Car Ownership Percentage (Car Proxy 1).
We use two methods: 1) Hard Bucketing and 2) Piecewise Linear Interpolation.

--- 
Data Setup (From Monthly Household Expenditure Among Resident Households 2023)
---

X-axis for smooth interpolation (xp): Midpoints of the 11 official expenditure groups (in S$).
This ensures that np.interp uses the center of the bucket for its calculation.
Assumptions for open-ended buckets:
  - Below 1k -> Midpoint 500
  - 6k-7.999k -> Midpoint 7000
  - 15k & Over -> Midpoint 17500 (Assuming a cap at 20k)
"""
ownership_midpoints_xp = np.array([
    500, 1500, 2500, 3500, 4500, 5500,
    7000, 9000, 11000, 13500, 17500
]) # 11 points for xp

# Y-axis for both methods (fp): The Car Ownership Percentage (%) for each expenditure bucket
ownership_values_fp = np.array([1.3, 7.6, 18.5, 31.9, 44.0, 50.6, 63.2, 73.1, 76.7, 88.5, 87.7]) # 11 points for fp

# X-axis for hard bucketing: 12 boundary points (exp_bins)
exp_bins_for_bucket = np.array([
    0, 1000, 2000, 3000, 4000, 5000, 6000, 8000, 10000, 12000, 15000, 20000
])


# --- Method 1: Hard Bucketing (Nearest Neighbor) ---

def get_bucket_value_revised(exp):
    """
    Finds the exact expenditure bucket for 'exp' and returns the official
    average ownership value (fp) for that bucket. This provides the 'hard' estimate.
    """
    # Use searchsorted to find the index of the upper boundary
    idx = np.searchsorted(exp_bins_for_bucket, exp, side="right") - 1
    
    # Clip index to ensure it stays within the valid range [0, 10]
    idx = np.clip(idx, 0, len(ownership_values_fp) - 1)
    
    return ownership_values_fp[idx]

# 'Ownership' column stores the official average P_Car of the bucket
df["Ownership"] = df["Interpolated_Expenditure"].apply(get_bucket_value_revised)


# --- Method 2: Piecewise Linear Interpolation (Smooth Estimate) ---

# Interpolated ownership = smooth interpolation using midpoints (xp) and ownership values (fp)
df["Interpolated_Ownership"] = df["Interpolated_Expenditure"].apply(
    # Note: Interpolation uses midpoints_xp (11 points) and values_fp (11 points)
    lambda exp: piecewise_interpolate(exp, ownership_midpoints_xp, ownership_values_fp)
)

# Car Proxy 1 (Core Proxy) is defined as the interpolated smooth value.
df["Car Proxy 1 (Ownership%)"] = df["Interpolated_Ownership"]

# -------------------------------------------------------------
# 4. Vehicle Expenditure based proxy (Car Proxy 2)
# -------------------------------------------------------------
"""
This section estimates the Vehicle Expenditure based on the Planning Area's
Median Income Percentile, which acts as the core ranking factor.

--- 
Data Setup (From Monthly Household Income from Work Per Household Member)
---
"""

# X-axis (xp): The Income Percentile Boundaries
vehicle_exp_percentile_boundaries = np.array([0., 20., 40., 60., 80., 100.])

# Y-axis (fp): The AVERAGE Vehicle Expenditure for each income quintile
vehicle_exp_values_quintile_avg = np.array([276.9, 381.2, 652.9, 812.4, 1243.2])

# Replicate the last average value to cap the interpolation curve (P80 to P100 plateau)
vehicle_exp_values_interp_fp = np.append(vehicle_exp_values_quintile_avg, 1243.2)


# --- Method 1: Hard Bucketing (Bucket Value) ---

def get_vehicle_exp_bucket(pc):
    """
    Finds which 20% income percentile bucket 'pc' falls into and returns the
    official average Vehicle Expenditure for that specific bucket.
    """
    # Use the 6 boundaries to find the 0-4 index of the 5 quintiles
    idx = np.searchsorted(vehicle_exp_percentile_boundaries, pc, side="right") - 1
    
    # Clip index to ensure it is within the 5 average values [0, 4]
    idx = np.clip(idx, 0, len(vehicle_exp_values_quintile_avg) - 1)
    
    return vehicle_exp_values_quintile_avg[idx]

# VehicleExp_p (This is the Hard Bucketing output)
df["VehicleExp_p (Bucket Value)"] = df["Median_Monthly_Income_Percentile"].apply(get_vehicle_exp_bucket)


# --- Method 2: Piecewise Linear Interpolation (Smooth Estimate) ---

# Interpolation uses the 6 boundaries (xp) and the 6 extended values (fp)
df["Interpolated_VehicleExp_p"] = df["Median_Monthly_Income_Percentile"].apply(
    lambda pc: piecewise_interpolate(pc,
                                     vehicle_exp_percentile_boundaries,
                                     vehicle_exp_values_interp_fp)
)

# Car Proxy 2 (Core Proxy) is defined as the interpolated smooth value.
df["Car Proxy 2 (Economic Parameter)"] = df["Interpolated_VehicleExp_p"]

# -------------------------------------------------------------
# 5. Output CSV
# -------------------------------------------------------------
output_path = "/Users/zhangwenyu/Desktop/NUSFYP/car_proxy_outputs.csv"
df.to_csv(output_path, index=False)

print("Output saved to:", output_path)
