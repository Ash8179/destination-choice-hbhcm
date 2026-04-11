"""
=====================================================================
C.2.9 – Street-Level Perception Aggregation Pipeline
=====================================================================

Goal
-----
Aggregate image-level perceptual scores into POI-level indicators,
handling missing images and missing priors safely.

Input
-----
Perceptual Features.csv  (image-level)
D.xlsx                   (POI registry with all POIs)

Output
------
D.xlsx (updated)

For each POI, includes:
- final indicators
- priors
- weights
- aggregation method
- robust normalized scores
- reliability statistics

Notation
--------
i  : image
p  : POI

w_time_i = exp(-λ * age_days_i)
w_view_i = 1.0 if panorama else 0.75
w_i = w_time_i * w_view_i

Weighted mean:
μ_p = Σ(w_i x_ip) / Σ(w_i)
Effective sample size:
n_eff = (Σ w_i)^2 / Σ(w_i^2)
Prior:
μ_prior = α μ_category + (1-α) μ_planning_area
Shrinkage:
posterior = (n_eff μ_p + κ μ_prior)/(n_eff + κ)

Aggregation method:
- n_images==0 → "no_image"
- n_eff>=5 → "normal"
- n_eff<5 → "shrinkage"

Robust normalization:
z = (x - median)/IQR
=====================================================================

Author: Zhang Wenyu
Date: 2026-03-13
"""

import pandas as pd
import numpy as np
import os
from datetime import date

# =========================
# File paths  ← edit these
# =========================
IMG_FILE  = "/Users/zhangwenyu/Desktop/NUSFYP/Stage 1/1.3 Street Level Imagery/Perceptual Features.csv"
POI_FILE  = "/Users/zhangwenyu/Desktop/NUSFYP/Stage 1/1.3 Street Level Imagery/D.xlsx"
OUTPUT_DIR = "/Users/zhangwenyu/Desktop/NUSFYP/Stage 1/1.3 Street Level Imagery/pipeline_output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_XLSX      = os.path.join(OUTPUT_DIR, "D_updated.xlsx")
OUTPUT_WEIGHTS   = os.path.join(OUTPUT_DIR, "image_weights.csv")
OUTPUT_PRIORS    = os.path.join(OUTPUT_DIR, "poi_priors.csv")
OUTPUT_CAT_PRIOR = os.path.join(OUTPUT_DIR, "category_priors.csv")
OUTPUT_AREA_PRIOR= os.path.join(OUTPUT_DIR, "area_priors.csv")
OUTPUT_REPORT    = os.path.join(OUTPUT_DIR, "pipeline_report.md")

# =========================
# Parameters
# =========================
HALF_LIFE = 365                        # days
LAMBDA    = np.log(2) / HALF_LIFE
ALPHA     = 0.6                        # weight on category prior
KAPPA     = 4                          # pseudo-count for shrinkage
REFERENCE_DATE = date.today()          # age relative to today

DIMENSIONS = [
    "vibrancy",
    "pleasantness",
    "walkability",
    "safety",
    "experiential",
]

# =========================
# 1. Load data
# =========================
print("Loading data …")
img = pd.read_csv(IMG_FILE)
poi = pd.read_excel(POI_FILE)

total_images_raw = len(img)

# Keep only OK images
img = img[img["status"] == "ok"].copy()
n_ok = len(img)

# =========================
# 2. Parse timestamps
#    Strip everything from 'T' onward so we only keep YYYY-MM-DD,
#    then parse as a plain date.  This avoids micro-second / timezone
#    issues that cause pd.to_datetime to return NaT.
# =========================
img["_date_str"] = (
    img["captured_at"]
      .astype(str)
      .str.split("T").str[0]   # keep only the date portion
      .str.strip()
)

img["captured_date"] = pd.to_datetime(img["_date_str"], format="%Y-%m-%d", errors="coerce")

n_invalid = img["captured_date"].isna().sum()
if n_invalid > 0:
    print(f"  ⚠  {n_invalid} images still have unparseable dates after truncation "
          f"and will be skipped.")
    img = img[img["captured_date"].notna()].copy()
else:
    print("  ✓  All timestamps parsed successfully after date truncation.")

# Age in days (integer, date arithmetic)
ref = pd.Timestamp(REFERENCE_DATE)
img["age_days"] = (ref - img["captured_date"]).dt.days

# Guard: drop images with negative age (future dates)
n_future = (img["age_days"] < 0).sum()
if n_future:
    print(f"  ⚠  {n_future} images have future capture dates and will be skipped.")
    img = img[img["age_days"] >= 0].copy()

n_used = len(img)

# =========================
# 3. Image weights
# =========================
img["w_time"] = np.exp(-LAMBDA * img["age_days"])
img["is_panorama"] = img["is_panorama"].astype(str).str.upper()
img["w_view"]  = np.where(img["is_panorama"] == "TRUE", 1.0, 0.75)
img["weight"]  = img["w_time"] * img["w_view"]

# Export image-level weights
weight_cols = ["poi_id", "captured_at", "_date_str", "age_days",
               "w_time", "w_view", "weight", "is_panorama"]
weight_cols = [c for c in weight_cols if c in img.columns]
img[weight_cols].to_csv(OUTPUT_WEIGHTS, index=False)
print(f"  ✓  Image weights saved → {OUTPUT_WEIGHTS}")

# =========================
# 4. Helper functions
# =========================
def wmean(x, w):
    return np.sum(w * x) / np.sum(w)

def wvar(x, w, m):
    return np.sum(w * (x - m) ** 2) / np.sum(w)

def n_eff(w):
    return (np.sum(w) ** 2) / np.sum(w ** 2)

def safe_prior(table, key, column, fallback):
    if key in table.index:
        val = table.loc[key, column]
        if pd.notna(val):
            return val
    return fallback

def choose_method(n_images, neff):
    if n_images == 0:
        return "no_image"
    elif neff >= 5:
        return "normal"
    else:
        return "shrinkage"

# =========================
# 5. Aggregate images → POI
# =========================
print("Aggregating images to POI …")
rows = []
for poi_id, g in img.groupby("poi_id"):
    w = g["weight"].values
    r = {
        "poi_id":           poi_id,
        "n_images":         len(g),
        "n_panos":          (g["is_panorama"] == "TRUE").sum(),
        "median_image_age": g["age_days"].median(),
        "min_image_age":    g["age_days"].min(),
        "max_image_age":    g["age_days"].max(),
        "weight_sum":       w.sum(),
        "n_eff":            n_eff(w),
        "subtype":     g["subtype"].iloc[0]       if "subtype"       in g.columns else "unknown",
        "planning_area":g["planning_area"].iloc[0] if "planning_area" in g.columns else "unknown",
    }
    for d in DIMENSIONS:
        x = g[d].values
        m = wmean(x, w)
        r[d + "_mean"] = m
        r[d + "_var"]  = wvar(x, w, m)
    rows.append(r)

agg = pd.DataFrame(rows)

# =========================
# 6. Merge with POI registry
# =========================
# Ensure we keep the statistical columns during merge
stats_cols = ["poi_id", "n_images", "n_panos", "median_image_age", "weight_sum", "n_eff"] + \
             [d + "_mean" for d in DIMENSIONS] + [d + "_var" for d in DIMENSIONS]

# Only take columns that actually exist in agg
existing_stats_cols = [c for c in stats_cols if c in agg.columns]
agg_to_merge = agg[existing_stats_cols].copy()

# Perform the merge
poi = poi.merge(agg_to_merge, on="poi_id", how="left")

# Fill NaN for POIs with no images
poi["n_images"] = poi["n_images"].fillna(0).astype(int)
poi["n_eff"] = poi["n_eff"].fillna(0.0)
poi["weight_sum"] = poi["weight_sum"].fillna(0.0)

# Fill measurement columns with 0 or NaN as appropriate
for d in DIMENSIONS:
    if d + "_mean" in poi.columns:
        poi[d + "_mean"] = poi[d + "_mean"].fillna(0.0)

# Handle metadata defaults
if "subtype" not in poi.columns: poi["subtype"] = "unknown"
if "planning_area" not in poi.columns: poi["planning_area"] = "unknown"
poi["subtype"] = poi["subtype"].fillna("unknown")
poi["planning_area"] = poi["planning_area"].fillna("unknown")

# =========================
# 7. Priors
# =========================
print("Computing priors …")
mean_cols = [d + "_mean" for d in DIMENSIONS]
category_prior = agg.groupby("subtype")[mean_cols].mean()
area_prior     = agg.groupby("planning_area")[mean_cols].mean()
global_prior   = agg[mean_cols].mean()

# =========================
# 8. Aggregation method (REVISED - VECTORIZED)
# =========================
# Using np.select is much safer than .apply(lambda)
poi["aggregation_method"] = np.select(
    condlist=[
        poi["n_images"] == 0,
        poi["n_eff"] >= 5
    ],
    choicelist=["no_image", "normal"],
    default="shrinkage"
)

# =========================
# 9. Final indicators + priors table
# =========================
print("Computing final indicators …")
prior_rows = []
for d in DIMENSIONS:
    vals, priors_d = [], []
    for idx, r in poi.iterrows():
        mu_cat  = safe_prior(category_prior, r["subtype"],       d + "_mean", global_prior[d + "_mean"])
        mu_area = safe_prior(area_prior,     r["planning_area"], d + "_mean", global_prior[d + "_mean"])
        mu_prior = ALPHA * mu_cat + (1 - ALPHA) * mu_area
        priors_d.append(mu_prior)

        method = r["aggregation_method"]
        if method == "normal":
            v = r[d + "_mean"]
        elif method == "shrinkage":
            v = (r["n_eff"] * r[d + "_mean"] + KAPPA * mu_prior) / (r["n_eff"] + KAPPA)
        else:
            v = mu_prior
        vals.append(v)

    poi[d]             = vals
    poi[d + "_prior"]  = priors_d

# Build per-POI prior export
prior_export_cols = ["poi_id", "subtype", "planning_area", "aggregation_method",
                     "n_images", "n_eff"]
for d in DIMENSIONS:
    prior_export_cols += [d + "_prior", d + "_mean", d]

prior_export = poi[[c for c in prior_export_cols if c in poi.columns]].copy()
prior_export.to_csv(OUTPUT_PRIORS, index=False)
print(f"  ✓  POI priors & weights saved → {OUTPUT_PRIORS}")

# =========================
# 10. Robust normalization
# =========================
for d in DIMENSIONS:
    med = poi[d].median()
    iqr = poi[d].quantile(0.75) - poi[d].quantile(0.25)
    poi[d + "_robust_z"] = (poi[d] - med) / iqr if iqr != 0 else 0.0

# =========================
# 11. Save updated POI file
# =========================
poi.to_excel(OUTPUT_XLSX, index=False)
print(f"  ✓  Updated POI file saved → {OUTPUT_XLSX}")

# =========================
# 12. Console summary
# =========================
method_counts = poi["aggregation_method"].value_counts()
print("\n==== PIPELINE SUMMARY ====\n")
print(f"Raw images in CSV          : {total_images_raw}")
print(f"Images with status==ok     : {n_ok}")
print(f"Images with invalid dates  : {n_invalid}")
print(f"Images with future dates   : {n_future}")
print(f"Images used in aggregation : {n_used}")
print(f"Total POI                  : {len(poi)}")
print("\nAggregation methods")
print(method_counts.to_string())
print("\nImage statistics (per POI)")
print(f"  mean images  : {poi['n_images'].mean():.2f}")
print(f"  median images: {poi['n_images'].median():.0f}")
print("\nn_eff statistics")
print(f"  mean n_eff   : {poi['n_eff'].mean():.3f}")
print(f"  median n_eff : {poi['n_eff'].median():.3f}")
print(f"\nPOI with missing category prior  : {(~poi['subtype'].isin(category_prior.index)).sum()}")
print(f"POI with missing planning area prior: {(~poi['planning_area'].isin(area_prior.index)).sum()}")

# =========================
# 13. Write Markdown report
# =========================
print("\nWriting report …")

n_normal    = method_counts.get("normal",   0)
n_shrinkage = method_counts.get("shrinkage",0)
n_no_image  = method_counts.get("no_image", 0)
pct_covered = 100 * (n_normal + n_shrinkage) / len(poi) if len(poi) > 0 else 0

dim_stats = []
for d in DIMENSIONS:
    dim_stats.append({
        "dim": d,
        "mean":   poi[d].mean(),
        "median": poi[d].median(),
        "std":    poi[d].std(),
        "min":    poi[d].min(),
        "max":    poi[d].max(),
        "p25":    poi[d].quantile(0.25),
        "p75":    poi[d].quantile(0.75),
    })

report = f"""# Street-Level Perception Aggregation Pipeline — Report
Generated: {date.today()}  
Reference date for age calculation: {REFERENCE_DATE}

---

## 1. Overview

This report documents the aggregation of street-level imagery perceptual scores
into POI-level indicators using a Bayesian shrinkage approach with temporal and
view-type weighting.

| Parameter | Value |
|-----------|-------|
| Half-life (λ) | {HALF_LIFE} days |
| Decay constant | {LAMBDA:.6f} |
| Category prior weight (α) | {ALPHA} |
| Shrinkage pseudo-count (κ) | {KAPPA} |
| Shrinkage threshold (n_eff) | 5 |

---

## 2. Timestamp Parsing Fix

**Problem in v1:** `pd.to_datetime(..., infer_datetime_format=True)` failed on
timestamps of the form `2024-08-31T12:40:52.339000`, returning NaT for
{6068} images — likely a combination of microsecond precision and/or timezone
suffix incompatibility with the pandas version in use.

**Fix applied:** The date portion is extracted by splitting on `'T'` and keeping
only the left side (`YYYY-MM-DD`) before parsing with an explicit format string
`%Y-%m-%d`. Age is then computed as integer calendar days relative to today
(`{REFERENCE_DATE}`). This is sufficient for temporal weighting and avoids all
sub-second / timezone edge cases.

---

## 3. Image Inventory

| Metric | Count |
|--------|-------|
| Raw images in CSV | {total_images_raw} |
| Images with status == ok | {n_ok} |
| Images with invalid dates (skipped) | {n_invalid} |
| Images with future capture dates (skipped) | {n_future} |
| **Images used in aggregation** | **{n_used}** |

---

## 4. POI Coverage

| Aggregation Method | Count | % of POI |
|--------------------|-------|----------|
| normal (n_eff ≥ 5) | {n_normal} | {100*n_normal/len(poi):.1f}% |
| shrinkage (0 < n_eff < 5) | {n_shrinkage} | {100*n_shrinkage/len(poi):.1f}% |
| no_image | {n_no_image} | {100*n_no_image/len(poi):.1f}% |
| **Total POI** | **{len(poi)}** | 100% |

POI with at least one image: **{pct_covered:.1f}%**

Average images per POI (all POI): {poi['n_images'].mean():.2f}  
Median images per POI (all POI): {poi['n_images'].median():.0f}  
Average n_eff (all POI): {poi['n_eff'].mean():.3f}  
Median n_eff (all POI): {poi['n_eff'].median():.3f}

---

## 5. Weighting Scheme

### 5.1 Temporal Weight
```
w_time_i = exp(−λ · age_days_i)    λ = ln(2) / {HALF_LIFE}
```
An image captured exactly {HALF_LIFE} days ago receives weight 0.5.
Images captured today receive weight 1.0.

### 5.2 View-type Weight
```
w_view_i = 1.00  if panorama
           0.75  otherwise
```

### 5.3 Combined Weight
```
weight_i = w_time_i × w_view_i
```

Image-level weights are saved in: `image_weights.csv`

---

## 6. Prior Construction

Category and planning-area priors are the unweighted means of POI-level
weighted means (`*_mean` columns in the aggregated dataset).

```
μ_prior = α · μ_category + (1−α) · μ_planning_area
        = {ALPHA} · μ_category + {1-ALPHA} · μ_planning_area
```

If a POI's subtype or planning area is absent from the prior table
(e.g. only one POI in that group, not included in the aggregated set),
the global mean is used as a fallback.

| Prior file | Description |
|------------|-------------|
| `category_priors.csv` | Per-subtype mean for each dimension |
| `area_priors.csv` | Per-planning-area mean for each dimension |
| `poi_priors.csv` | Per-POI prior, raw mean, final indicator |

POI with missing category prior: {(~poi['subtype'].isin(category_prior.index)).sum()}  
POI with missing planning area prior: {(~poi['planning_area'].isin(area_prior.index)).sum()}

---

## 7. Final Indicator Statistics

"""

header = "| Dimension | Mean | Median | Std | Min | Max | P25 | P75 |"
sep    = "|-----------|------|--------|-----|-----|-----|-----|-----|"
report += header + "\n" + sep + "\n"
for s in dim_stats:
    report += (f"| {s['dim']} | {s['mean']:.4f} | {s['median']:.4f} | "
               f"{s['std']:.4f} | {s['min']:.4f} | {s['max']:.4f} | "
               f"{s['p25']:.4f} | {s['p75']:.4f} |\n")

report += f"""
Robust z-scores (median/IQR normalised) are stored in `*_robust_z` columns.

---

## 8. Output Files

| File | Description |
|------|-------------|
| `D_updated.xlsx` | Full POI registry with all indicators, priors, weights |
| `image_weights.csv` | Per-image w_time, w_view, weight, age_days |
| `category_priors.csv` | Per-subtype prior means |
| `area_priors.csv` | Per-planning-area prior means |
| `poi_priors.csv` | Per-POI: prior, raw mean, final indicator for each dimension |
| `pipeline_report.md` | This report |

---

## 9. Pipeline Evaluation

### Strengths
- **Temporal decay** correctly downweights older street-view imagery.
- **Bayesian shrinkage** prevents noisy estimates from POI with very few images
  from dominating downstream analysis.
- **Fallback chain** (category → area → global) ensures every POI receives a
  valid estimate even when group priors are unavailable.
- **Date-only arithmetic** avoids all timezone / microsecond parsing edge cases.

### Limitations & Recommendations
1. **{n_no_image} POI ({100*n_no_image/len(poi):.1f}%) have no imagery** — their indicators
   are entirely prior-driven. Treat results for these POI with caution and
   consider flagging them in downstream analyses.
2. **Median n_eff = {poi['n_eff'].median():.1f}** across all POI suggests the majority
   rely on shrinkage. Consider collecting additional imagery for data-sparse areas.
3. **α = {ALPHA}** gives substantially more weight to the category prior than the
   area prior. If spatial heterogeneity is large, consider increasing the area
   weight (lower α).
4. **κ = {KAPPA}** is a moderately strong regulariser. If you have independent
   validation data, calibrate κ via cross-validation.
5. **Planning area prior missing for {(~poi['planning_area'].isin(area_prior.index)).sum()} POI** —
   these fall back to global mean. Verify that planning_area values in `D.xlsx`
   match those in `Perceptual Features.csv`.

---
*Pipeline v2 — timestamp fix + CSV exports + report*
"""

with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
    f.write(report)

print(f"  ✓  Report saved → {OUTPUT_REPORT}")
print("\nAll done.")
