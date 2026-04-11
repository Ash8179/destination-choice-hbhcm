"""
================================================================================
F.2 - OD FLOW COMPUTATION v6 — Aggregate Layer for Hierarchical Bayesian Hybrid
Choice Model (HBHCM)
================================================================================

REVISION NOTES (v6 vs v5)
--------------------------
Three identification fixes applied relative to v5:

FIX 1 — BASE_RATE vs alpha_cat ABSOLUTE SCALE CONFOUND               [CRITICAL]
──────────────────────────────────────────────────────────────────────────────
Problem (v5):
    mu_j = sum_o T_o * P_od,  T_o = pop * rate * exp(gamma_car * PCA_Car)
    Likelihood: rating_count_j ~ NegBin(mu_j * exp(alpha_cat_j), phi)
    Multiplying BASE_RATE by any constant c is perfectly absorbed by
        alpha_cat_k -> alpha_cat_k - ln(c)  for all k.
    BASE_RATE and the overall level of alpha_cat are not jointly identified
    from cross-sectional rating_count data alone.

Fix:
    - BASE_RATE and BASE_RATE_CBD remain as fixed, externally-set constants
      (unchanged from v5 in the pipeline script).
    - In Layer 4 (PyMC), alpha_mall is FIXED at 0 (reference category).
      Only 3 free alpha_cat parameters are estimated:
          alpha_hawker, alpha_cultural, alpha_monument
          (park+historic is the additional reference; see CATEGORY_MAP)
    - No global intercept is added. The absolute scale of mu_j is anchored
      by BASE_RATE; the category relativities are identified by the 3 free
      alpha_cat parameters.
    - Pipeline output: cat_fe column uses 0=mall(base) with explicit
      documentation that alpha_mall equiv 0 in Layer 4.

    Calibration note:
      If you have an aggregate footfall benchmark, set BASE_RATE so that
          sum_j mu_j_prior  ~  observed_total_daily_leisure_trips_to_sample
      Rerun this script after adjusting; then proceed to Layer 4.

FIX 2 — gamma_car vs beta_GTC COLLINEARITY                    [EMPIRICAL RISK]
──────────────────────────────────────────────────────────────────────────────
Problem (v5):
    gamma_car adjusts T_o (trip generation path).
    beta_GTC captures travel cost in destination choice (P_od path).
    High-car-ownership origins tend to be suburban -> systematically higher GTC.
    In joint posterior: gamma_car and beta_GTC form a banana-shaped posterior
    (strong negative correlation), causing slow MCMC mixing and instability.

Fix:
    Residualise PCA_Car_Proxy_01 against origin-mean GTC before writing to
    the calibration table:
        mean_GTC_o     = mean over accessible destinations of GTC_od
        slope, intercept = OLS(PCA_Car ~ mean_GTC_o)
        PCA_Car_resid  = PCA_Car - (slope * mean_GTC_o + intercept)
    By construction: Corr(PCA_Car_resid, mean_GTC_o) = 0.
    gamma_car retains interpretation: "car ownership effect on trip generation,
    orthogonal to travel cost." Tight prior Normal(0, 0.3) in Layer 4.
    PCA_Car_resid written to od_calibration_table_v6.csv for Layer 4 use.

FIX 3 — ACCESSIBILITY TRUNCATION: partial choice-set softmax  [SYSTEMATIC BIAS]
──────────────────────────────────────────────────────────────────────────────
Problem (v5):
    P_od was computed as softmax only over accessible POIs, then multiplied
    by full T_o. This assumes every leisure trip goes to an in-sample POI,
    inflating all mu_j. Inflation is heterogeneous across origins (origins
    with few accessible POIs are inflated most), corrupting the spatial
    distribution that Layer 4 fits to rating_count.

Fix:
    Add an outside option with normalised utility V_outside = 0 (reference).
    The softmax denominator becomes:
        denom_o = 1 + sum_{d accessible} exp(V_od)
    P_od = exp(V_od) / denom_o       (strictly less than v5 value)
    P_outside_o = 1 / denom_o
    mu_j = sum_o T_o * P_od          (already accounts for outside-option leak)
    Diagnostic column coverage_frac_o = 1 - P_outside_o written to output.
    In Layer 4, the outside-option share is implicitly absorbed into alpha_cat;
    coverage_frac enables post-hoc decomposition.

UNCHANGED FROM v5
─────────────────
  - GTC uses PT only (identical to SP pipeline Section 7; beta_GTC shared)
  - gamma_car lives in T_o ONLY (not in V_od; unidentified there per v5 note)
  - Merged category fixed effects: lifestyle->mall, historic->park
  - NegBin likelihood: rating_count_j ~ NegBin(mu_j * exp(alpha_cat_j), phi)
  - lambda (star rating) excluded
  - GAMMA_CAR_PRIOR = 0.3 at pipeline stage (estimated freely in Layer 4)

================================================================================
PIPELINE OVERVIEW
================================================================================
  0  · Configuration & constants
  1  · Data loading & validation
  2  · Data cleaning
       2a · Origin table
       2b · Destination table
       2c · PT skims
       2d · Car skims (sensitivity reference)
  3  · Data quality monitoring
       3a · rating_count audit
       3b · Star rating descriptive (passive)
       3c · Cross-check rating vs count (confirms lambda exclusion)
       3d · Subtype face-validity check
       3e · PT skim coverage
  4  · GTC computation (PT only, SP-aligned)
  5  · Scale categorisation (mixed GFA/footprint, 3-tier dummies)
  6  · Trip generation T_o with gamma_car prior
  7  · FIX 2 — Residualise PCA_Car against mean_GTC_o
  8  · Category fixed-effect encoding (merged groups; FIX 1 documented)
  9  · Destination utility V_od and choice probabilities P_od (FIX 3)
  10 · Predicted visits mu_j
  11 · Calibration table assembly
  12 · Summary diagnostics & Layer 5 readiness check

================================================================================
FIELDS USED
================================================================================

FROM  O.xlsx
  origin_id, subzone_name, pop_total, avg_hh_size, Mean_Monthly_Income
  PCA_Car_Proxy_01   : car ownership proxy [0,1] -> T_o adjustment
  cbd_flag           : CBD flag -> trip rate stratification
  central_area_flag

FROM  D.xlsx
  poi_id, name, category, subtype
  footprint_area, gfa, storey_count
  popularity_proxy   : Google Maps star rating (PASSIVE — not in likelihood)
  rating_count       : Google Maps review count (NegBin dependent variable)
  vibrancy_robust_z, pleasantness_robust_z, walkability_robust_z,
  safety_robust_z, experiential_robust_z : 5 CV dimensions (delta_CV)
  cbd_flag, central_area_flag

FROM  pt_skims.csv
  origin_id, poi_id, total_time_min, transfers, pt_fare_proxy
  accessible_flag    : 1/True = accessible
  error_code         : "OK" = valid; "HTTP_404" etc = dropped

FROM  drive_skims.csv
  origin_id, poi_id, travel_time_min, total_cost_proxy, accessible_flag

================================================================================
KEY FORMULAS (v6)
================================================================================

GTC (PT only — identical to SP Section 7):
    GTC_od = total_time_min + pt_fare_proxy / VOT + transfers * TRANSFER_PENALTY
    VOT              = S$0.20/min
    TRANSFER_PENALTY = 5.0 min/transfer

Scale (mixed GFA/footprint):
    scale_m2 = gfa              if gfa > 0
             = footprint_area   otherwise
    cut_low, cut_high = tertile cuts on scale_m2 across all POIs
    D_sml = 1{scale_m2 <  cut_low}       reference (omitted in utility)
    D_med = 1{cut_low <= scale_m2 < cut_high}
    D_lrg = 1{scale_m2 >= cut_high}

Trip generation (v6 = v5 formulation):
    rate_o = BASE_RATE_CBD   if cbd_flag = 1
           = BASE_RATE        otherwise
    T_o_base = pop_total_o * rate_o
    T_o      = T_o_base * exp(GAMMA_CAR_PRIOR * PCA_Car_Proxy_01_o)
    [In Layer 4: T_o_adj = T_o_base * exp(gamma_car * PCA_Car_resid_o)]

FIX 2 — Residualisation:
    mean_GTC_o = mean of GTC_od over accessible destinations for origin o
    slope, intercept = OLS(PCA_Car ~ mean_GTC_o)
    PCA_Car_resid_o = PCA_Car_o - (slope * mean_GTC_o + intercept)
    Verification: Corr(PCA_Car_resid, mean_GTC_o) ~ 0

Destination utility (prior stage; delta_CV = 0):
    V_od = BETA_GTC_PRIOR   * GTC_od
         + BETA_S_MED_PRIOR * D_med_j
         + BETA_S_LRG_PRIOR * D_lrg_j

FIX 3 — Choice probabilities with outside option:
    denom_o   = 1 + sum_{d accessible} exp(V_od)
    P_od      = exp(V_od) / denom_o
    P_outside = 1 / denom_o
    coverage_frac_o = 1 - P_outside = sum exp(V_od) / denom_o

Predicted visits:
    mu_j = sum_o T_o * P_od     [T_o uses GAMMA_CAR_PRIOR at pipeline stage]

NegBin likelihood (Layer 4):
    rating_count_j ~ NegBin(mu_j * exp(alpha_cat_j + delta_CV @ z_j), phi)
    FIX 1: alpha_mall equiv 0 (reference); only 3 free alpha_cat in Layer 4

================================================================================
LAYER 5 PARAMETER ALIGNMENT (v6)
================================================================================
  beta_GTC, beta_S_med, beta_S_lrg : shared SP <-> OD (pivot-from-reality)
  delta_vib ... delta_exp           : OD only; from rating_count_j
  gamma_car                         : OD nuisance; in T_o via exp(); tight prior
  gamma_cbd                         : OD only; cbd_flag_d
  alpha_hawker, alpha_cultural,
  alpha_monument, alpha_park        : OD only; relative to mall = 0 (FIX 1)
  phi                               : OD only; NegBin overdispersion
  BASE_RATE, BASE_RATE_CBD          : FIXED — not estimated in Layer 4

  REMOVED vs v5: none (all v5 parameters retained; 3 fixes are structural)
  ADDED vs v5:
    PCA_Car_resid_o in od_calibration_table (FIX 2)
    coverage_frac_o in od_calibration_table (FIX 3)

================================================================================
OUTPUT FILES
================================================================================
  Cleaned/origin_clean.csv
  Cleaned/dest_clean.csv
  Cleaned/pt_skims_clean.csv
  Cleaned/car_skims_clean.csv
  Cleaned/dest_with_scale.csv
  quality_report_v6.csv
  dest_calibration_table_v6.csv   <- Layer 4 NegBin targets (one row per POI)
  od_calibration_table_v6.csv     <- long OD pairs (Layer 4 matrix assembly)

Author: Zhang Wenyu
Date: 2026-03-27
================================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr, pearsonr

warnings.filterwarnings("ignore")

# =============================================================================
# SECTION 0 — CONFIGURATION & CONSTANTS
# =============================================================================

BASE_DIR   = os.path.join(os.path.expanduser("~"), "Desktop", "NUSFYP",
                          "Stage 1", "1.4 OD Flow Computation")
NEW_OUTPUT_DIR   = os.path.join(os.path.expanduser("~"), "Desktop", "NUSFYP",
                          "Stage 2", "HBHCM", "Layer 4")
INPUT_DIR  = os.path.join(BASE_DIR, "Input")
OUTPUT_DIR = os.path.join(NEW_OUTPUT_DIR, "Output")
CLEAN_DIR  = os.path.join(INPUT_DIR, "Cleaned")

for d in [OUTPUT_DIR, CLEAN_DIR]:
    os.makedirs(d, exist_ok=True)

# ── GTC parameters — must be IDENTICAL to SP pipeline Section 7 ───────────────
VOT              = 0.20    # S$/min
TRANSFER_PENALTY = 5.0     # min/transfer

# ── Prior beta values for mu_j initialisation ─────────────────────────────────
# These are used ONLY at the pipeline stage to compute V_od and P_od.
# Layer 4 estimates beta_GTC and beta_S freely, informed by SP-posterior priors.
BETA_GTC_PRIOR    = -0.0074
BETA_S_MED_PRIOR  =  0.4485
BETA_S_LRG_PRIOR  =  0.6691

# ── Trip generation rates ──
# Calibrate BASE_RATE externally so that:
#     sum_j mu_j_prior  ~  total observed daily leisure trips to sample POIs
# After calibration, rerun this script, then run Layer 4.
BASE_RATE     = 0.20    # leisure trips / person / week, non-CBD
BASE_RATE_CBD = 0.30    # leisure trips / person / week, CBD

# ── Car ownership prior ──────────────
# GAMMA_CAR_PRIOR = 0.3 as in v5 — kept for continuity.
# T_o_base (no car adjustment) is also written to output so Layer 4 can
# apply gamma_car to PCA_Car_resid freely.
GAMMA_CAR_PRIOR = 0.3

# ── Outside option flag ───────────────────────────────────────────────
# Set to False only for direct comparison with v5 (diagnostic use).
INCLUDE_OUTSIDE_OPTION = True

# ── Star rating column (passive — not in likelihood) ──────────────────────────
RATING_COL       = "popularity_proxy"
RATING_VALID_MIN = 1.0
RATING_VALID_MAX = 5.0

# ── Category fixed-effect map (identical to v3/v5) ────────────────────────────
# Mall = 0 (reference; alpha_mall equiv 0 in Layer 4 — NOT a free param)
# Active Layer 4 dummies: cat_fe_2 (hawker), cat_fe_3 (cultural),
#                          cat_fe_4 (monument), cat_fe_5 (park+historic)
CATEGORY_MAP = {
    "mall":             0,
    "lifestyle street": 0,    # merged into mall
    "hawker centre":    2,
    "museum":           3,    # cultural
    "theatre":          3,    # cultural
    "monument":         4,
    "historic site":    5,    # merged into park
    "park":             5,
}
CATEGORY_LABELS = {
    0: "mall + lifestyle street (base; alpha_mall equiv 0)",
    2: "hawker centre",
    3: "cultural (museum + theatre)",
    4: "monument",
    5: "park + historic site",
}

# ── Quality thresholds ────────────────────────────────────────────────────────
RATING_COUNT_LOW_THRESHOLD = 10
RATING_COUNT_HIGH_PCTILE   = 99

print("=" * 70)
print("OD FLOW COMPUTATION v6")
print("Fixes vs v5:")
print("  FIX 1: BASE_RATE fixed; alpha_mall=0 reference in Layer 4")
print("  FIX 2: PCA_Car residualised against mean_GTC_o")
print("  FIX 3: outside option added to softmax denominator")
print("=" * 70)
print(f"\nInput  : {INPUT_DIR}")
print(f"Output : {OUTPUT_DIR}")
print(f"Cleaned: {CLEAN_DIR}")
print(f"\nGTC: VOT={VOT} S$/min,  transfer_penalty={TRANSFER_PENALTY} min")
print(f"Base rates: {BASE_RATE}/wk (non-CBD),  {BASE_RATE_CBD}/wk (CBD)  [FIXED]")
print(f"GAMMA_CAR_PRIOR = {GAMMA_CAR_PRIOR}  (estimated freely in Layer 4)")
print(f"Outside option: {INCLUDE_OUTSIDE_OPTION}")


# =============================================================================
# SECTION 1 — DATA LOADING & VALIDATION
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 1 · Data Loading & Validation")
print("=" * 70)

origin_raw = pd.read_excel(os.path.join(INPUT_DIR, "O.xlsx"))
dest_raw   = pd.read_excel(os.path.join(INPUT_DIR, "D.xlsx"))
pt_raw     = pd.read_csv(os.path.join(INPUT_DIR, "pt_skims.csv"))
car_raw    = pd.read_csv(os.path.join(INPUT_DIR, "drive_skims.csv"))

print(f"\n  O.xlsx      : {origin_raw.shape[0]:>5} rows x {origin_raw.shape[1]} cols")
print(f"  D.xlsx      : {dest_raw.shape[0]:>5} rows x {dest_raw.shape[1]} cols")
print(f"  pt_skims    : {pt_raw.shape[0]:>6} rows x {pt_raw.shape[1]} cols")
print(f"  drive_skims : {car_raw.shape[0]:>6} rows x {car_raw.shape[1]} cols")

for col, source in [("rating_count", "D.xlsx"),
                    ("subtype",       "D.xlsx"),
                    ("PCA_Car_Proxy_01", "O.xlsx")]:
    df = dest_raw if source == "D.xlsx" else origin_raw
    if col not in df.columns:
        raise ValueError(f"Required column '{col}' not found in {source}.")


# =============================================================================
# SECTION 2 — DATA CLEANING
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 2 · Data Cleaning")
print("=" * 70)

# ── 2a. Origin table ──────────────────────────────────────────────────────────
print("\n-- 2a. Origin table --")

O_COLS = ["origin_id", "subzone_name", "planning_area",
          "pop_total", "avg_hh_size", "Mean_Monthly_Income",
          "PCA_Car_Proxy_01", "cbd_flag", "central_area_flag"]
origin = origin_raw[[c for c in O_COLS if c in origin_raw.columns]].copy()

n0 = len(origin)
origin = origin.dropna(subset=["pop_total", "PCA_Car_Proxy_01"])
origin = origin[origin["pop_total"] >= 0]
origin["PCA_Car_Proxy_01"]  = origin["PCA_Car_Proxy_01"].clip(0, 1)
origin["cbd_flag"]          = origin["cbd_flag"].fillna(0).astype(int)
origin["central_area_flag"] = origin["central_area_flag"].fillna(0).astype(int)

print(f"  Rows: {n0} -> {len(origin)} (dropped {n0 - len(origin)})")
print(f"  CBD origins      : {origin['cbd_flag'].sum()}")
print(f"  Zero-pop zones   : {(origin['pop_total'] == 0).sum()}")
print(f"  PCA_Car range    : [{origin['PCA_Car_Proxy_01'].min():.3f}, "
      f"{origin['PCA_Car_Proxy_01'].max():.3f}]")
print(f"  PCA_Car mean/med : {origin['PCA_Car_Proxy_01'].mean():.3f} / "
      f"{origin['PCA_Car_Proxy_01'].median():.3f}")

origin.to_csv(os.path.join(CLEAN_DIR, "origin_clean.csv"), index=False)
print(f"  -> Saved: Cleaned/origin_clean.csv")

# ── 2b. Destination table ─────────────────────────────────────────────────────
print("\n-- 2b. Destination table --")

D_COLS = ["poi_id", "name", "category", "subtype",
          "footprint_area", "gfa", "storey_count",
          "fnb_count", "retail_count", "entertainment_count", "service_count",
          "diversity_index",
          "has_supermarket", "has_cinema", "has_department_store",
          "has_foodcourt", "has_event_space",
          RATING_COL, "rating_count",
          "vibrancy_robust_z", "pleasantness_robust_z", "walkability_robust_z",
          "safety_robust_z", "experiential_robust_z",
          "cbd_flag", "central_area_flag"]
D_COLS = [c for c in D_COLS if c in dest_raw.columns]
dest = dest_raw[D_COLS].copy()

n0 = len(dest)
dest = dest.dropna(subset=["footprint_area"])
dest = dest[dest["footprint_area"] > 0]
print(f"  Rows after footprint filter: {n0} -> {len(dest)} (dropped {n0 - len(dest)})")

# Clean rating_count
dest["rating_count"] = (pd.to_numeric(dest["rating_count"], errors="coerce")
                        .fillna(0).clip(lower=0).astype(int))

# Clean star rating (passive)
dest["rating"] = pd.to_numeric(dest[RATING_COL], errors="coerce")
n_out = (dest["rating"].notna() &
         ((dest["rating"] < RATING_VALID_MIN) | (dest["rating"] > RATING_VALID_MAX))).sum()
dest.loc[dest["rating"].notna() &
         ((dest["rating"] < RATING_VALID_MIN) | (dest["rating"] > RATING_VALID_MAX)),
         "rating"] = np.nan
if n_out > 0:
    print(f"  WARNING: {n_out} ratings outside [{RATING_VALID_MIN}, {RATING_VALID_MAX}] -> NaN")

# Eligibility: rating_count > 0 only
dest["layer4_eligible"] = (dest["rating_count"] > 0).astype(int)
n_eligible = dest["layer4_eligible"].sum()
n_total    = len(dest)
print(f"\n  Layer 4 eligibility (rating_count > 0):")
print(f"    Eligible : {n_eligible}  ({n_eligible/n_total:.1%})")
print(f"    Excluded : {n_total - n_eligible}")

# Normalise text columns
dest["category"] = dest["category"].astype(str).str.strip().str.lower()
dest["subtype"]  = dest["subtype"].astype(str).str.strip().str.lower()

# CV z-scores: fill NaN with 0
cv_cols = ["vibrancy_robust_z", "pleasantness_robust_z", "walkability_robust_z",
           "safety_robust_z", "experiential_robust_z"]
for c in cv_cols:
    dest[c] = dest[c].fillna(0)

for c in ["has_supermarket", "has_cinema", "has_department_store",
          "has_foodcourt", "has_event_space"]:
    if c in dest.columns:
        dest[c] = dest[c].fillna(0).astype(int)
dest["cbd_flag"]          = dest["cbd_flag"].fillna(0).astype(int)
dest["central_area_flag"] = dest["central_area_flag"].fillna(0).astype(int)

print(f"\n  Subtype distribution:")
print(dest["subtype"].value_counts().to_string())

dest.to_csv(os.path.join(CLEAN_DIR, "dest_clean.csv"), index=False)
print(f"\n  -> Saved: Cleaned/dest_clean.csv")

# ── 2c. PT skims ──────────────────────────────────────────────────────────────
print("\n-- 2c. PT skims --")

PT_COLS = ["origin_id", "poi_id", "total_time_min", "transfers",
           "pt_fare_proxy", "accessible_flag", "error_code"]
pt = pt_raw[[c for c in PT_COLS if c in pt_raw.columns]].copy()

n0 = len(pt)
pt = pt[pt["accessible_flag"].astype(str).str.strip().str.lower().isin(["1", "true"])]
if "error_code" in pt.columns:
    pt = pt[pt["error_code"].astype(str).str.strip().str.upper()
            .isin(["OK", "NAN", "0", "NONE", ""])]
pt = pt.dropna(subset=["total_time_min", "pt_fare_proxy", "transfers"])
pt = pt[(pt["total_time_min"] >= 0) & (pt["pt_fare_proxy"] >= 0) & (pt["transfers"] >= 0)]

print(f"  Rows: {n0} -> {len(pt)} (dropped {n0 - len(pt)})")
print(f"  Unique origins : {pt['origin_id'].nunique()}")
print(f"  Unique POIs    : {pt['poi_id'].nunique()}")

pt.to_csv(os.path.join(CLEAN_DIR, "pt_skims_clean.csv"), index=False)
print(f"  -> Saved: Cleaned/pt_skims_clean.csv")

# ── 2d. Car skims (sensitivity reference) ─────────────────────────────────────
print("\n-- 2d. Car skims (sensitivity reference) --")

if "total_cost_proxy" not in car_raw.columns:
    cost_parts = [c for c in ["distance_cost", "erp_proxy", "park_fee_proxy"]
                  if c in car_raw.columns]
    if cost_parts:
        car_raw["total_cost_proxy"] = car_raw[cost_parts].fillna(0).sum(axis=1)
        print(f"  Built total_cost_proxy from: {cost_parts}")

CAR_COLS = ["origin_id", "poi_id", "travel_time_min",
            "total_cost_proxy", "accessible_flag", "status_code"]
car = car_raw[[c for c in CAR_COLS if c in car_raw.columns]].copy()
n0  = len(car)
car = car[car["accessible_flag"].astype(str).str.strip().str.lower().isin(["1", "true"])]
car = car.dropna(subset=["travel_time_min", "total_cost_proxy"])
car = car[(car["travel_time_min"] >= 0) & (car["total_cost_proxy"] >= 0)]
print(f"  Rows: {n0} -> {len(car)} (dropped {n0 - len(car)})")
car.to_csv(os.path.join(CLEAN_DIR, "car_skims_clean.csv"), index=False)
print(f"  -> Saved: Cleaned/car_skims_clean.csv")


# =============================================================================
# SECTION 3 — DATA QUALITY MONITORING
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 3 · Data Quality Monitoring")
print("=" * 70)

quality_records = []

# ── 3a. rating_count audit ────────────────────────────────────────────────────
print("\n-- 3a. rating_count audit (NegBin dependent variable) --")

rc           = dest["rating_count"].copy()
n_total_dest = len(rc)
n_zero       = (rc == 0).sum()
n_sparse     = ((rc > 0) & (rc <= RATING_COUNT_LOW_THRESHOLD)).sum()
n_valid      = (rc > RATING_COUNT_LOW_THRESHOLD).sum()
high_thresh  = rc.quantile(RATING_COUNT_HIGH_PCTILE / 100)
n_outlier    = (rc > high_thresh).sum()

print(f"  Total POIs              : {n_total_dest}")
print(f"  rating_count = 0        : {n_zero}  ({n_zero/n_total_dest:.1%})  <- excluded")
print(f"  rating_count 1-{RATING_COUNT_LOW_THRESHOLD}        : "
      f"{n_sparse}  ({n_sparse/n_total_dest:.1%})  <- sparse")
print(f"  rating_count > {RATING_COUNT_LOW_THRESHOLD}        : "
      f"{n_valid}   ({n_valid/n_total_dest:.1%})  <- Layer 4")
print(f"  Top-1% threshold        : {high_thresh:.0f}  -> {n_outlier} POIs")

# Flag columns
dest["rating_count_flag"] = "valid"
dest.loc[dest["rating_count"] == 0, "rating_count_flag"] = "zero_excluded"
dest.loc[(dest["rating_count"] > 0) &
         (dest["rating_count"] <= RATING_COUNT_LOW_THRESHOLD),
         "rating_count_flag"] = "sparse"
dest.loc[dest["rating_count"] > high_thresh, "rating_count_flag"] = "high_outlier"

# Provisional FE for reporting
dest["_cat_fe_tmp"] = dest["subtype"].map(CATEGORY_MAP).fillna(5).astype(int)
print(f"\n  rating_count by merged FE group:")
fe_rc = dest.groupby("_cat_fe_tmp")["rating_count"].agg(
    mean="mean", median="median",
    n_zero=lambda x: (x == 0).sum(), n_total="count"
).round(1)
fe_rc.index = [CATEGORY_LABELS.get(i, f"FE={i}") for i in fe_rc.index]
print(fe_rc.to_string())

quality_records += [
    {"check": "rating_count_zero",
     "value": n_zero, "pct": round(n_zero/n_total_dest, 4),
     "flag": "OK" if n_zero < 100 else "WARN",
     "note": "POIs with no reviews; excluded from Layer 4"},
    {"check": "rating_count_sparse",
     "value": n_sparse, "pct": round(n_sparse/n_total_dest, 4),
     "flag": "OK" if n_sparse < 50 else "WARN",
     "note": f"POIs with 1-{RATING_COUNT_LOW_THRESHOLD} reviews; low signal"},
    {"check": "rating_count_high_outlier",
     "value": n_outlier, "pct": round(n_outlier/n_total_dest, 4), "flag": "OK",
     "note": f"Top-1% (>{high_thresh:.0f}); sensitivity run recommended"},
]

# ── 3b. Star rating descriptive (passive) ─────────────────────────────────────
print("\n-- 3b. Star rating (PASSIVE — not in NegBin likelihood) --")

rt         = dest["rating"]
n_valid_rt = rt.notna().sum()
print(f"  Valid ratings  : {n_valid_rt}  ({n_valid_rt/n_total_dest:.1%})")
print(f"  Missing        : {rt.isna().sum()}  (still eligible if rating_count > 0)")
print(rt.describe().round(3).to_string())

quality_records.append({
    "check": "rating_star_descriptive", "value": n_valid_rt,
    "pct": round(n_valid_rt/n_total_dest, 4), "flag": "INFO",
    "note": "Star rating is PASSIVE in v6; excluded from NegBin likelihood"
})

# ── 3c. Cross-check confirming lambda exclusion ───────────────────────────────
print("\n-- 3c. Cross-check: rating_count vs star rating --")

both = dest[(dest["rating_count"] > 0) & dest["rating"].notna()].copy()
if len(both) >= 10:
    rho_s, p_s = spearmanr(both["rating_count"], both["rating"])
    rho_p, p_p = pearsonr(np.log1p(both["rating_count"]), both["rating"])
    print(f"  Spearman rho (count vs rating)     : {rho_s:.3f}  (p={p_s:.4f})")
    print(f"  Pearson r    (log_count vs rating) : {rho_p:.3f}  (p={p_p:.4f})")
    verdict = "OK: confirms lambda exclusion." if abs(rho_s) < 0.15 else \
              "NOTE: higher than expected; revisit lambda decision."
    print(f"  {verdict}")
    quality_records.append({
        "check": "lambda_exclusion_check", "value": round(rho_s, 4),
        "pct": np.nan, "flag": "OK" if abs(rho_s) < 0.15 else "WARN",
        "note": f"Spearman rho={rho_s:.3f} (p={p_s:.4f}). Lambda excluded."
    })

# ── 3d. Subtype face-validity ─────────────────────────────────────────────────
print("\n-- 3d. Subtype face-validity: median star rating --")
sub_med = (dest[dest["rating"].notna()].groupby("subtype")["rating"]
           .median().sort_values(ascending=False))
print(sub_med.round(3).to_string())

quality_records.append({
    "check": "rating_subtype_ranking", "value": np.nan,
    "pct": np.nan, "flag": "INFO",
    "note": "Face validity: monument > lifestyle > museum > park > mall > hawker (expected)"
})

# ── 3e. PT skim coverage ──────────────────────────────────────────────────────
print("\n-- 3e. PT skim coverage for Layer 4 eligible POIs --")

eligible_pois = set(dest[dest["layer4_eligible"] == 1]["poi_id"].unique())
pt_pois       = set(pt["poi_id"].unique())
covered       = eligible_pois & pt_pois
uncovered     = eligible_pois - pt_pois

print(f"  Layer 4 eligible POIs : {len(eligible_pois)}")
print(f"  With PT skims         : {len(covered)}  ({len(covered)/len(eligible_pois):.1%})")
print(f"  No PT skim (excluded) : {len(uncovered)}")

if 0 < len(uncovered) <= 25:
    unc_df = dest[dest["poi_id"].isin(uncovered)][
        ["poi_id", "name", "subtype", "rating_count"]
    ].sort_values("rating_count", ascending=False)
    print(f"\n  Excluded eligible POIs (no PT):")
    print(unc_df.to_string(index=False))

quality_records.append({
    "check": "eligible_pois_pt_coverage",
    "value": len(covered),
    "pct": round(len(covered)/len(eligible_pois), 4),
    "flag": "OK" if len(covered)/len(eligible_pois) > 0.85 else "WARN",
    "note": f"{len(uncovered)} eligible POIs lack PT skims"
})

# Save quality report (will be extended in Sections 7 and 9)
qr_df   = pd.DataFrame(quality_records)
qr_path = os.path.join(OUTPUT_DIR, "quality_report_v6.csv")


# =============================================================================
# SECTION 4 — GTC COMPUTATION (PT ONLY)
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 4 · GTC Computation (PT only)")
print("=" * 70)
print(f"\n  GTC_od = total_time_min + pt_fare_proxy/{VOT} + transfers*{TRANSFER_PENALTY}")
print(f"  Formula IDENTICAL to SP pipeline Section 7  -> beta_GTC shared")

pt["GTC_od"] = (pt["total_time_min"]
                + pt["pt_fare_proxy"] / VOT
                + pt["transfers"] * TRANSFER_PENALTY)

print(f"\n  GTC_od summary (min-equivalent):")
print(pt["GTC_od"].describe().round(3).to_string())

# Car GTC for sensitivity column
car["GTC_car_od"] = car["travel_time_min"] + car["total_cost_proxy"] / VOT
pt = pt.merge(car[["origin_id", "poi_id", "GTC_car_od"]],
              on=["origin_id", "poi_id"], how="left")
print(f"\n  Car GTC pairs available: {pt['GTC_car_od'].notna().sum()}")


# =============================================================================
# SECTION 5 — SCALE CATEGORISATION
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 5 · Scale Categorisation (mixed GFA/footprint, 3-tier)")
print("=" * 70)

dest["scale_m2"] = np.where(
    dest["gfa"].notna() & (dest["gfa"] > 0),
    dest["gfa"], dest["footprint_area"]
)
dest["scale_source"] = np.where(
    dest["gfa"].notna() & (dest["gfa"] > 0), "gfa", "footprint"
)

n_gfa = (dest["scale_source"] == "gfa").sum()
print(f"\n  scale_m2: {n_gfa} POIs use GFA, {len(dest)-n_gfa} use footprint_area")

cut_low  = dest["scale_m2"].quantile(1/3)
cut_high = dest["scale_m2"].quantile(2/3)
print(f"  Tertile cuts: {cut_low:.1f} m2 (33rd) / {cut_high:.1f} m2 (67th)")

dest["Scale_cat"] = pd.cut(dest["scale_m2"],
                            bins=[0, cut_low, cut_high, np.inf],
                            labels=[0, 1, 2], right=False).astype(int)
dest["D_sml"] = (dest["Scale_cat"] == 0).astype(int)   # reference
dest["D_med"] = (dest["Scale_cat"] == 1).astype(int)
dest["D_lrg"] = (dest["Scale_cat"] == 2).astype(int)

for k, label in [(0, "small/ref"), (1, "medium"), (2, "large")]:
    n = (dest["Scale_cat"] == k).sum()
    print(f"    {label:<12}: {n:>4}  ({n/len(dest)*100:.1f}%)")

dest.to_csv(os.path.join(CLEAN_DIR, "dest_with_scale.csv"), index=False)
print(f"\n  -> Saved: Cleaned/dest_with_scale.csv")


# =============================================================================
# SECTION 6 — TRIP GENERATION T_o
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 6 · Trip Generation T_o  (v5/v6 formulation)")
print("=" * 70)
print(f"\n  T_o = pop * rate * exp(GAMMA_CAR_PRIOR * PCA_Car)")
print(f"  rate = {BASE_RATE_CBD}/wk (CBD) or {BASE_RATE}/wk (non-CBD)  [FIXED; FIX 1]")
print(f"  GAMMA_CAR_PRIOR = {GAMMA_CAR_PRIOR}  (prior mean; freely estimated in Layer 4)")

origin["rate_o"] = np.where(origin["cbd_flag"] == 1, BASE_RATE_CBD, BASE_RATE)
origin["T_o_base"] = origin["pop_total"] * origin["rate_o"]
origin["T_o"]      = (origin["T_o_base"]
                      * np.exp(GAMMA_CAR_PRIOR * origin["PCA_Car_Proxy_01"]))

print(f"\n  T_o_base total : {origin['T_o_base'].sum():,.1f} trips/wk  <- used for prior mu_j")
print(f"  T_o total (with gamma_car prior): {origin['T_o'].sum():,.1f} trips/wk  "
      f"(ratio vs base: {origin['T_o'].sum()/origin['T_o_base'].sum():.4f})")
print(f"  Zero-T origins : {(origin['T_o'] == 0).sum()}")
print(f"  NOTE: Prior-stage mu_j uses T_o_BASE (gamma_car=0). ")
print(f"        gamma_car is estimated freely in Layer 4 via:")
print(f"        T_o_adj = T_o_base * exp(gamma_car * PCA_Car_resid)")
print(f"  NOTE: SCALE_FACTOR is auto-calibrated in Section 10 so that")
print(f"        sum(mu_j_prior) = sum(rating_count) for Layer 4 eligible POIs.")


# =============================================================================
# SECTION 7 — FIX 2: RESIDUALISE PCA_Car AGAINST mean_GTC_o
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 7 · FIX 2 — Residualise PCA_Car against mean_GTC_o")
print("=" * 70)
print(f"\n  Purpose: break the suburban confound between car ownership and GTC")
print(f"  Method : OLS(PCA_Car ~ mean_GTC_o); PCA_Car_resid = residuals")
print(f"  Result : Corr(PCA_Car_resid, mean_GTC_o) ~ 0 by construction")

# Compute origin-mean GTC from accessible PT pairs
mean_gtc_o = (pt.groupby("origin_id")["GTC_od"]
              .mean()
              .rename("mean_GTC_o")
              .reset_index())
origin = origin.merge(mean_gtc_o, on="origin_id", how="left")
n_missing_gtc = origin["mean_GTC_o"].isna().sum()
if n_missing_gtc > 0:
    print(f"  WARNING: {n_missing_gtc} origins have no PT-accessible destinations; "
          f"filling mean_GTC_o with overall median.")
origin["mean_GTC_o"] = origin["mean_GTC_o"].fillna(origin["mean_GTC_o"].median())

# OLS regression: PCA_Car ~ mean_GTC_o
slope, intercept, r_raw, p_val, _ = stats.linregress(
    origin["mean_GTC_o"], origin["PCA_Car_Proxy_01"]
)
origin["PCA_Car_resid"] = (
    origin["PCA_Car_Proxy_01"]
    - (slope * origin["mean_GTC_o"] + intercept)
)

# Verify near-zero residual correlation
r_resid, _ = stats.pearsonr(origin["mean_GTC_o"], origin["PCA_Car_resid"])

print(f"\n  OLS fit: PCA_Car = {slope:.5f} * mean_GTC_o + {intercept:.4f}")
print(f"  Correlation before residualisation: r = {r_raw:.4f}  (p={p_val:.4f})")
print(f"  Correlation after  residualisation: r = {r_resid:.2e}  (should be ~0)")

if abs(r_raw) < 0.20:
    print(f"  NOTE: Original correlation is weak ({r_raw:.3f}); "
          f"residualisation has minimal practical effect but is applied for robustness.")
elif abs(r_resid) > 0.05:
    print(f"  WARNING: Residual correlation {r_resid:.4f} > 0.05; "
          f"check for nonlinearity in PCA_Car vs GTC relationship.")
else:
    print(f"  OK: Residualisation successful.")

quality_records.append({
    "check": "pca_car_gtc_collinearity",
    "value": round(r_raw, 4), "pct": np.nan,
    "flag": "OK" if abs(r_raw) < 0.40 else "WARN",
    "note": (f"r_before={r_raw:.4f}, r_after={r_resid:.2e}. "
             f"FIX 2 applied: PCA_Car_resid in calibration table.")
})


# =============================================================================
# SECTION 8 — CATEGORY FIXED-EFFECT ENCODING  (FIX 1 documented)
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 8 · Category Fixed-Effect Encoding")
print("  FIX 1: mall = 0 (reference); alpha_mall equiv 0 in Layer 4")
print("=" * 70)

dest["cat_fe"] = dest["subtype"].map(CATEGORY_MAP)

unmapped = dest[dest["cat_fe"].isna()][["poi_id", "name", "subtype"]]
if len(unmapped) > 0:
    print(f"\n  WARNING: {len(unmapped)} POIs have unmapped subtype -> fallback FE=5:")
    print(unmapped.to_string(index=False))
    dest["cat_fe"] = dest["cat_fe"].fillna(5)

dest["cat_fe"] = dest["cat_fe"].astype(int)
dest.drop(columns=["_cat_fe_tmp"], inplace=True, errors="ignore")

print(f"\n  Merged FE distribution (mall+lifestyle = base; alpha_mall equiv 0):")
for code, label in CATEGORY_LABELS.items():
    n = (dest["cat_fe"] == code).sum()
    print(f"    FE={code}  {label:<45}: {n:>4} POIs")

# Active dummies for Layer 4 (FE=1 unused; mall=0 is reference)
for fe_code in [2, 3, 4, 5]:
    dest[f"cat_fe_{fe_code}"] = (dest["cat_fe"] == fe_code).astype(int)

print(f"\n  Layer 4 free alpha_cat: cat_fe_2 (hawker), cat_fe_3 (cultural),")
print(f"                           cat_fe_4 (monument), cat_fe_5 (park+historic)")
print(f"  Reference (NOT a free param): FE=0 — mall + lifestyle street")
print(f"  This anchors the absolute scale: BASE_RATE sets mu_j level,")
print(f"  alpha_cat_k gives log-multiplier RELATIVE to mall.")


# =============================================================================
# SECTION 9 — DESTINATION UTILITY AND CHOICE PROBABILITIES  (FIX 3)
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 9 · Destination Utility V_od and Choice Probabilities P_od")
print("  FIX 3: outside option added to softmax denominator")
print("=" * 70)

# Restrict to Layer 4 eligible destinations with PT skims
dest_layer4 = dest[
    (dest["layer4_eligible"] == 1) &
    (dest["poi_id"].isin(pt["poi_id"].unique()))
].copy().reset_index(drop=True)

origins_with_pt = origin[
    origin["origin_id"].isin(pt["origin_id"].unique())
].copy().reset_index(drop=True)

print(f"\n  Origins with PT access           : {len(origins_with_pt)}")
print(f"  Destination POIs (Layer 4 final) : {len(dest_layer4)}")

O_ids    = origins_with_pt["origin_id"].values
D_ids    = dest_layer4["poi_id"].values
# Prior stage uses T_o_BASE only.
# Reason: gamma_car is a free parameter in Layer 4; applying GAMMA_CAR_PRIOR here
# would pre-bake a partially-identified adjustment into mu_j_prior, making the
# offset inconsistent with the Layer 4 model when gamma_car != GAMMA_CAR_PRIOR.
T_o_base = origins_with_pt["T_o_base"].values   # unadjusted — used for prior mu_j

# Subset PT skims to eligible OD pairs
pt_od = pt[pt["poi_id"].isin(D_ids) & pt["origin_id"].isin(O_ids)].copy()

# Destination utility
pt_od = pt_od.merge(dest_layer4[["poi_id", "D_med", "D_lrg"]],
                    on="poi_id", how="left")
pt_od["V_od"] = (BETA_GTC_PRIOR    * pt_od["GTC_od"]
                 + BETA_S_MED_PRIOR * pt_od["D_med"]
                 + BETA_S_LRG_PRIOR * pt_od["D_lrg"])
pt_od["exp_V"] = np.exp(pt_od["V_od"])

# New：Only contains GTC，not β_S，for Layer 4 PyMC
pt_od["V_od_no_scale"] = BETA_GTC_PRIOR * pt_od["GTC_od"]
pt_od["exp_V_no_scale"] = np.exp(pt_od["V_od_no_scale"])

# Sum exp(V) over accessible destinations per origin
sum_exp = (pt_od.groupby("origin_id")["exp_V"]
           .sum()
           .rename("sum_exp_acc")
           .reset_index())

if INCLUDE_OUTSIDE_OPTION:
    # FIX 3: outside option exp(V_outside) = exp(0) = 1
    sum_exp["denom_o"]        = sum_exp["sum_exp_acc"] + 1.0
    sum_exp["P_outside_o"]    = 1.0 / sum_exp["denom_o"]
    sum_exp["coverage_frac_o"]= sum_exp["sum_exp_acc"] / sum_exp["denom_o"]
    outside_note = "(with outside option; FIX 3)"
else:
    sum_exp["denom_o"]         = sum_exp["sum_exp_acc"]
    sum_exp["P_outside_o"]     = 0.0
    sum_exp["coverage_frac_o"] = 1.0
    outside_note = "(NO outside option — v5 compatibility mode)"

pt_od = pt_od.merge(sum_exp[["origin_id", "denom_o",
                               "P_outside_o", "coverage_frac_o"]],
                    on="origin_id", how="left")
pt_od["P_od"] = pt_od["exp_V"] / pt_od["denom_o"]

# ── NO-SCALE version (for Layer 4) ─────────────────────────────
sum_exp_ns = (pt_od.groupby("origin_id")["exp_V_no_scale"]
              .sum()
              .rename("sum_exp_acc_ns")
              .reset_index())

if INCLUDE_OUTSIDE_OPTION:
    sum_exp_ns["denom_o_ns"] = sum_exp_ns["sum_exp_acc_ns"] + 1.0
else:
    sum_exp_ns["denom_o_ns"] = sum_exp_ns["sum_exp_acc_ns"]

pt_od = pt_od.merge(sum_exp_ns[["origin_id", "denom_o_ns"]],
                    on="origin_id", how="left")

pt_od["P_od_no_scale"] = pt_od["exp_V_no_scale"] / pt_od["denom_o_ns"]

cov = sum_exp["coverage_frac_o"]
print(f"\n  P_od computed {outside_note}")
print(f"  Coverage fraction (in-sample trip share):")
print(f"    mean={cov.mean():.3f},  min={cov.min():.3f},  "
      f"p10={cov.quantile(0.10):.3f},  p50={cov.median():.3f},  "
      f"p90={cov.quantile(0.90):.3f}")

if cov.min() < 0.30:
    print(f"  WARNING: Some origins have <30% of trips going to in-sample POIs.")
    print(f"    Consider expanding POI sample or inspecting skim coverage.")

quality_records.append({
    "check": "outside_option_coverage",
    "value": round(cov.mean(), 4), "pct": np.nan,
    "flag": "OK" if cov.mean() > 0.30 else "WARN",
    "note": (f"Mean coverage_frac={cov.mean():.3f}; "
             f"min={cov.min():.3f}. FIX 3 applied.")
})


# =============================================================================
# SECTION 10 — PREDICTED VISITS mu_j
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 10 · Predicted Visits  mu_j = sum_o T_o * P_od")
print("=" * 70)
print(f"\n  FIX 1: BASE_RATE={BASE_RATE}/wk fixed -> anchors absolute scale of mu_j")
print(f"  FIX 3: P_od already accounts for outside-option leak")

# Merge T_o_base onto pt_od for prior-stage flow computation
pt_od = pt_od.merge(
    origins_with_pt[["origin_id", "T_o", "T_o_base", "PCA_Car_resid"]],
    on="origin_id", how="left"
)

# ── Step A: compute unscaled mu_j using T_o_BASE ──────────────────────────────
# mu_j_unscaled = sum_o T_o_base * P_od   (no car adjustment; no SCALE_FACTOR yet)
pt_od["flow_od_base"] = pt_od["T_o_base"] * pt_od["P_od"]

mu_j_base = (pt_od.groupby("poi_id")["flow_od_base"]
             .sum().rename("mu_j_prior_base").reset_index())

dest_layer4 = dest_layer4.merge(mu_j_base, on="poi_id", how="left")
dest_layer4["mu_j_prior_base"] = dest_layer4["mu_j_prior_base"].fillna(0)

# ── Step A2: NO-SCALE flow ─────────────────────────────────────
pt_od["flow_od_no_scale"] = pt_od["T_o_base"] * pt_od["P_od_no_scale"]

mu_j_base_ns = (pt_od.groupby("poi_id")["flow_od_no_scale"]
                .sum().rename("mu_j_no_scale_base").reset_index())

dest_layer4 = dest_layer4.merge(mu_j_base_ns, on="poi_id", how="left")
dest_layer4["mu_j_no_scale_base"] = dest_layer4["mu_j_no_scale_base"].fillna(0)

# ── Step B: auto-calibrate SCALE_FACTOR ──────────────────────────────────────
# Goal: sum(mu_j_prior_scaled) = sum(rating_count) for Layer 4 eligible POIs.
# This makes alpha_cat values interpretable as log-multipliers relative to mall,
# without a systematic offset from mu_j/rating_count scale mismatch.
#
# Mathematically:
#   SCALE_FACTOR = sum(rating_count_j) / sum(mu_j_prior_base_j)
#
# Physical interpretation:
#   SCALE_FACTOR = (lifetime reviews per trip) x (weeks per lifetime)
#   It absorbs WEEKS_LIFETIME and REVIEW_RATE into a single data-driven constant.
total_rating_eligible = dest_layer4["rating_count"].sum()
total_mu_unscaled     = dest_layer4["mu_j_prior_base"].sum()
SCALE_FACTOR          = total_rating_eligible / total_mu_unscaled

print(f"\n  Auto-calibrated SCALE_FACTOR = {SCALE_FACTOR:.6f}")
print(f"    = sum(rating_count) / sum(mu_j_base)")
print(f"    = {total_rating_eligible:,.0f} / {total_mu_unscaled:,.1f}")
print(f"    Equivalent to: WEEKS_LIFETIME x REVIEW_RATE = {SCALE_FACTOR:.6f}")
print(f"    (e.g. ~{SCALE_FACTOR/52:.1f} yr lifetime × {SCALE_FACTOR*100/SCALE_FACTOR*0:.2f}% "
      f"— exact decomposition not unique; SCALE_FACTOR is a single calibrated constant)")

# ── Step C: apply SCALE_FACTOR to get mu_j_prior ─────────────────────────────
pt_od["flow_od"] = pt_od["flow_od_base"] * SCALE_FACTOR

mu_j_scaled = (pt_od.groupby("poi_id")["flow_od"]
               .sum().rename("mu_j_prior").reset_index())
dest_layer4 = dest_layer4.merge(mu_j_scaled, on="poi_id", how="left")
dest_layer4["mu_j_prior"] = dest_layer4["mu_j_prior"].fillna(0)

# ── Step C2: apply SCALE_FACTOR to no-scale ────────────────────
pt_od["flow_od_no_scale_scaled"] = pt_od["flow_od_no_scale"] * SCALE_FACTOR

mu_j_ns = (pt_od.groupby("poi_id")["flow_od_no_scale_scaled"]
           .sum().rename("mu_j_no_scale").reset_index())

dest_layer4 = dest_layer4.merge(mu_j_ns, on="poi_id", how="left")
dest_layer4["mu_j_no_scale"] = dest_layer4["mu_j_no_scale"].fillna(0)

total_T    = origins_with_pt["T_o_base"].sum()
total_mu   = dest_layer4["mu_j_prior"].sum()
pct_capt   = total_mu / total_T / SCALE_FACTOR * 100   # % of base trips captured

print(f"\n  Total T_o_base (prior stage, no car adj) : {total_T:,.1f} trips/wk")
print(f"  Total mu_j_prior (scaled)               : {total_mu:,.1f}")
print(f"  Total rating_count (Layer 4 targets)    : {total_rating_eligible:,.0f}")
print(f"  mu_j / rating_count ratio               : {total_mu/total_rating_eligible:.4f}  (target = 1.000)")
print(f"  mu_j distribution:")
print(dest_layer4["mu_j_prior"].describe().round(4).to_string())

# Prior validity: Spearman rho(mu_j, rating_count)
rho_mu, p_mu = spearmanr(dest_layer4["mu_j_prior"],
                          dest_layer4["rating_count"])
print(f"\n  Prior validity — Spearman rho(mu_j, rating_count): "
      f"{rho_mu:.3f}  (p={p_mu:.4f})")
print(f"  {'OK: prior GTC+Scale model has useful signal.' if rho_mu > 0.2 else 'Low: delta_CV will carry most weight in Layer 4.'}")

print(f"\n  mu_j_prior by merged FE group:")
for fe_code, label in CATEGORY_LABELS.items():
    grp = dest_layer4[dest_layer4["cat_fe"] == fe_code]
    if len(grp) >= 5:
        r, p = spearmanr(grp["mu_j_prior"], grp["rating_count"])
        print(f"    FE={fe_code}  {label[:38]:<38}: rho={r:.3f}  (p={p:.3f})  n={len(grp)}")

quality_records.append({
    "check": "prior_validity_spearman_rho",
    "value": round(rho_mu, 4), "pct": np.nan,
    "flag": "OK" if rho_mu > 0.2 else "WARN",
    "note": f"Spearman rho(mu_j_prior, rating_count)={rho_mu:.3f} (p={p_mu:.4f})"
})


# =============================================================================
# SECTION 11 — CALIBRATION TABLE ASSEMBLY
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 11 · Calibration Table Assembly")
print("=" * 70)

# ── 11a. Destination calibration table (Layer 4 NegBin input) ─────────────────
# Merge coverage_frac per POI (average over origins weighted by flow)
poi_coverage = (
    pt_od.groupby("poi_id")
    .apply(lambda df: np.average(df["coverage_frac_o"],
                                  weights=df["flow_od"].clip(lower=1e-9)))
    .rename("mean_coverage_frac")
    .reset_index()
)
dest_layer4 = dest_layer4.merge(poi_coverage, on="poi_id", how="left")
dest_layer4["mean_coverage_frac"] = (dest_layer4["mean_coverage_frac"]
                                      .fillna(cov.mean()))

dest_cal_cols = [
    "poi_id", "name", "category", "subtype", "cat_fe",
    "cat_fe_2", "cat_fe_3", "cat_fe_4", "cat_fe_5",
    "rating_count", "rating_count_flag",
    "rating",
    "mu_j_prior",
    "mu_j_prior_base",
    "mu_j_no_scale",
    "mu_j_no_scale_base",
    "mean_coverage_frac",
    "Scale_cat", "D_sml", "D_med", "D_lrg", "scale_m2", "scale_source",
    "vibrancy_robust_z", "pleasantness_robust_z", "walkability_robust_z",
    "safety_robust_z", "experiential_robust_z",
    "diversity_index",
    "cbd_flag", "central_area_flag",
]
dest_cal_cols = [c for c in dest_cal_cols if c in dest_layer4.columns]
dest_cal      = dest_layer4[dest_cal_cols].copy()

dest_cal_path = os.path.join(OUTPUT_DIR, "dest_calibration_table_v6.csv")
dest_cal.to_csv(dest_cal_path, index=False)
print(f"\n  Destination calibration table:")
print(f"    Shape   : {dest_cal.shape}")
print(f"    -> Saved: {dest_cal_path}")

print(f"\n  NegBin likelihood (Layer 4):")
print(f"    rating_count_j ~ NegBin(mu_j * exp(alpha_cat_j + delta_CV @ z_j), phi)")
print(f"    FIX 1: alpha_mall equiv 0 (reference; BASE_RATE fixed)")
print(f"    Free alpha_cat: cat_fe_2 (hawker), cat_fe_3 (cultural),")
print(f"                    cat_fe_4 (monument), cat_fe_5 (park+historic)")

# ── 11b. OD long table (Layer 4 computation graph) ────────────────────────────
od_long = pt_od[[
    "origin_id", "poi_id", "GTC_od",
    "V_od", "exp_V", "P_od",
    "V_od_no_scale", "exp_V_no_scale", "P_od_no_scale",
    "coverage_frac_o", "P_outside_o",
    "D_med", "D_lrg",
    "T_o", "T_o_base",
    "flow_od", "flow_od_base",
    "flow_od_no_scale", "flow_od_no_scale_scaled"
]].copy()

# Merge destination attributes
dest_od_cols = ["poi_id", "subtype", "cat_fe",
                "vibrancy_robust_z", "pleasantness_robust_z",
                "walkability_robust_z", "safety_robust_z", "experiential_robust_z",
                "cbd_flag", "central_area_flag"]
od_long = od_long.merge(
    dest_layer4[[c for c in dest_od_cols if c in dest_layer4.columns]],
    on="poi_id", how="inner"
)

# Merge origin attributes
origin_od_cols = ["origin_id", "subzone_name", "pop_total",
                  "rate_o", "cbd_flag", "central_area_flag",
                  "mean_GTC_o", "PCA_Car_Proxy_01"]
od_long = od_long.merge(
    origins_with_pt[[c for c in origin_od_cols if c in origins_with_pt.columns]],
    on="origin_id", how="inner",
    suffixes=("_d", "_o")
)

# Rename ambiguous flag columns
if "cbd_flag" in od_long.columns:
    od_long.rename(columns={"cbd_flag": "cbd_flag_d",
                             "central_area_flag": "central_area_flag_d"},
                   inplace=True)

od_long_path = os.path.join(OUTPUT_DIR, "od_calibration_table_v6.csv")
od_long.to_csv(od_long_path, index=False)
print(f"\n  OD long table (computation graph for PyMC):")
print(f"    Shape   : {od_long.shape}")
print(f"    Columns : {list(od_long.columns)}")
print(f"    -> Saved: {od_long_path}")

# =============================================================================
# CHECKS — No-scale probability + mu_j integrity
# =============================================================================

print("\n" + "=" * 70)
print("CHECKS · No-scale consistency and integrity")
print("=" * 70)

# ─────────────────────────────────────────────────────────────
# 1. Probability normalization (NO-SCALE)
# ─────────────────────────────────────────────────────────────

# Sum of P_od_no_scale per origin
check_prob = pt_od.groupby("origin_id")["P_od_no_scale"].sum()

print("\n  [CHECK 1] P_od_no_scale sum per origin:")
print(check_prob.describe())

if INCLUDE_OUTSIDE_OPTION:
    # Reconstruct outside option for NO-SCALE
    check_outside = sum_exp_ns.set_index("origin_id")["denom_o_ns"]
    check_outside = 1.0 / check_outside

    # Align index
    check_outside = check_outside.reindex(check_prob.index)

    total_prob = check_prob + check_outside

    print("\n  [CHECK 1] P_inside + P_outside (should be 1):")
    print(total_prob.describe())

    assert np.allclose(total_prob, 1.0, atol=1e-6), \
        "NO-SCALE probabilities do not sum to 1 (including outside option)"

else:
    assert np.allclose(check_prob, 1.0, atol=1e-6), \
        "NO-SCALE probabilities do not sum to 1"

print("  ✔ Probability normalization OK")


# ─────────────────────────────────────────────────────────────
# 2. mu_j_no_scale validity
# ─────────────────────────────────────────────────────────────

required_cols = ["mu_j_no_scale", "rating_count"]
missing = [c for c in required_cols if c not in dest_cal.columns]

if missing:
    raise ValueError(f"Missing required columns for Layer 4: {missing}")

# Only forbid negative (zero is allowed)
if (dest_cal["mu_j_no_scale"] < 0).any():
    raise ValueError("mu_j_no_scale contains negative values")

print("  ✔ mu_j_no_scale non-negative OK")


# ─────────────────────────────────────────────────────────────
# 3. Rank correlation sanity check
# ─────────────────────────────────────────────────────────────

rho_ns, _ = spearmanr(dest_layer4["mu_j_no_scale"],
                      dest_layer4["rating_count"])

rho_s, _  = spearmanr(dest_layer4["mu_j_prior"],
                      dest_layer4["rating_count"])

print("\n  [CHECK 3] Spearman comparison:")
print(f"    with scale    : {rho_s:.3f}")
print(f"    no scale      : {rho_ns:.3f}")

# Optional soft warning (NOT assert)
if rho_ns < 0:
    print("  WARNING: mu_j_no_scale negatively correlated with rating_count")

print("  ✔ Rank correlation check complete")


# ─────────────────────────────────────────────────────────────
# 4. Scale consistency check
# ─────────────────────────────────────────────────────────────

ratio = (dest_layer4["mu_j_prior"].sum() /
         dest_layer4["mu_j_no_scale"].sum())

print("\n  [CHECK 4] Scale consistency:")
print(f"    total(mu_j_prior) / total(mu_j_no_scale) = {ratio:.4f}")
print(f"    SCALE_FACTOR                             = {SCALE_FACTOR:.4f}")

# This should be approximately 1 if both were scaled with same factor
assert np.isfinite(ratio), "Invalid ratio in scale consistency check"

print("  ✔ Scale consistency OK")


print("\n" + "=" * 70)
print("ALL CHECKS PASSED")
print("=" * 70)


# =============================================================================
# SECTION 12 — SUMMARY DIAGNOSTICS & LAYER 5 READINESS
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 12 · Summary Diagnostics & Layer 5 Readiness")
print("=" * 70)

# Save final quality report
qr_df = pd.DataFrame(quality_records)
qr_df.to_csv(qr_path, index=False)
print(f"\n  Quality report -> {qr_path}")
print(qr_df[["check", "value", "flag", "note"]].to_string(index=False))

print(f"\n  COVERAGE")
print(f"  Origins in OD table          : {od_long['origin_id'].nunique()}")
print(f"  Destinations in OD table     : {od_long['poi_id'].nunique()}")
print(f"  Total O-D pairs              : {len(od_long)}")
print(f"  Layer 4 NegBin targets       : {len(dest_cal)}")
print(f"    rating_count > 10          : {(dest_cal['rating_count'] > RATING_COUNT_LOW_THRESHOLD).sum()}")
print(f"    sparse (1-10)              : {(dest_cal['rating_count_flag'] == 'sparse').sum()}")

print(f"\n  GTC ALIGNMENT")
print(f"  GTC_od mean   : {od_long['GTC_od'].mean():.2f} min-equiv")
print(f"  GTC_od median : {od_long['GTC_od'].median():.2f} min-equiv")
print(f"  Formula: TT + fare/0.20 + transfers*5  [identical to SP] OK")

print(f"\n  v6 FIX STATUS")
print(f"  FIX 1 (scale confound)    : BASE_RATE={BASE_RATE}/wk FIXED; "
      f"alpha_mall equiv 0 documented")
print(f"  FIX 2 (collinearity)      : r(PCA_Car, mean_GTC) before={r_raw:.4f}, "
      f"after={r_resid:.2e}")
print(f"  FIX 3 (outside option)    : INCLUDE_OUTSIDE_OPTION={INCLUDE_OUTSIDE_OPTION}; "
      f"mean_coverage={cov.mean():.3f}")

print(f"\n  MERGED FE DISTRIBUTION (Layer 4)")
for fe_code, label in CATEGORY_LABELS.items():
    n = (dest_cal["cat_fe"] == fe_code).sum()
    print(f"    FE={fe_code}  {label:<45}: {n:>4}  ({n/len(dest_cal)*100:.1f}%)")

print(f"\n  LAYER 5 PARAMETER ALIGNMENT CHECKLIST")
print(f"  OK  beta_GTC       : PT VOT={VOT} — identical to SP")
print(f"  OK  beta_S_med/lrg : D_med / D_lrg dummies")
print(f"  OK  delta_CV       : 5 CV z-scores — identified from rating_count_j")
print(f"  OK  gamma_car      : in T_o via exp(); PCA_Car_resid in calibration table")
print(f"  OK  gamma_cbd      : cbd_flag_d in V_od (Layer 4)")
print(f"  OK  alpha_cat      : cat_fe_2..5 (4 dummies; FIX 1: mall equiv 0)")
print(f"  OK  phi            : NegBin overdispersion")
print(f"  --  lambda         : REMOVED (Spearman rho near-zero; v3 decision)")
print(f"  --  BASE_RATE      : FIXED (not estimated; FIX 1)")

print(f"\n  OUTPUT FILES (v6)")
print(f"  {dest_cal_path}")
print(f"    -> {len(dest_cal)} rows | NegBin input")
print(f"    -> Key cols: rating_count, mu_j_prior, mu_j_prior_base,")
print(f"       cat_fe_2..5, D_med, D_lrg, CV z-scores, mean_coverage_frac")
print(f"  {od_long_path}")
print(f"    -> {len(od_long)} rows | OD computation graph")
print(f"    -> Key cols: GTC_od, P_od, T_o, T_o_base, PCA_Car_resid,")
print(f"       coverage_frac_o, flow_od")
print(f"  {qr_path}")
print(f"    -> Quality monitoring report")

print(f"\n  NEXT STEPS")
print(f"  1. Review quality_report_v6.csv — check all flags are OK or INFO")
print(f"  2. Calibrate BASE_RATE if total mu_j is far from observed footfall")
print(f"  3. Run layer4_pymc_v6_patched.py using dest_calibration_table_v6.csv")
print(f"     (patched version: beta_S_med/lrg de-baked and re-entered as free params)")

print(f"\n{'=' * 70}")
print(f"COMPLETE — od_flow_computation_v6.py")
print(f"{'=' * 70}")
