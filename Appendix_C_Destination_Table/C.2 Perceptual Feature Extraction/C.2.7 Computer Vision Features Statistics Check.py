"""
C.2.7 – Perceptual Features Sanity Check

This script:
Perform comprehensive statistical validation of perceptual features and
composite dimensions, including distribution analysis, correlation,
PCA, multicollinearity (VIF), and scale consistency checks.

Author: Zhang Wenyu
Date: 2026-03-11
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

try:
    from rich.console import Console
    from rich.table import Table
    from rich import print as rprint
    console = Console()
    USE_RICH = True
except ImportError:
    USE_RICH = False
    console = None

SEP  = "=" * 72
SEP2 = "-" * 72

def header(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)

def subheader(title):
    print(f"\n{SEP2}")
    print(f"  {title}")
    print(SEP2)

# ══════════════════════════════════════════════════════════════════════════════
# 0. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
header("0. LOADING DATA")

DATA_PATH = "/Users/zhangwenyu/Desktop/NUSFYP/Stage 1/1.3 Street Level Imagery/perceptual_features_with_dimensions.csv"
df = pd.read_csv(DATA_PATH)

print(f"  Shape         : {df.shape[0]} rows × {df.shape[1]} cols")
print(f"  Columns       : {list(df.columns)}")

# ── Define column groups ──────────────────────────────────────────────────────
FEAT_COLS = [
    "percept_greenery",
    "percept_sky_visibility",
    "percept_building_frontage",
    "percept_ground_surface",
    "percept_lighting_presence",
    "percept_pedestrian_presence",
    "percept_vehicle_presence",
    "percept_signage_density",
    "percept_street_furniture",
    "percept_architectural_variety",
    "percept_activity_diversity",
    "percept_shading_coverage",
]

# Short display aliases
ALIASES = {
    "percept_greenery":              "greenery",
    "percept_sky_visibility":        "sky_vis",
    "percept_building_frontage":     "bldg_front",
    "percept_ground_surface":        "ground_surf",
    "percept_lighting_presence":     "lighting",
    "percept_pedestrian_presence":   "pedestrian",
    "percept_vehicle_presence":      "vehicle",
    "percept_signage_density":       "signage",
    "percept_street_furniture":      "street_furn",
    "percept_architectural_variety": "arch_variety",
    "percept_activity_diversity":    "act_diversity",
    "percept_shading_coverage":      "shading",
}

DIM_COLS = ["vibrancy", "pleasantness", "walkability", "safety", "experiential"]

# Verify columns exist
missing = [c for c in FEAT_COLS if c not in df.columns]
if missing:
    print(f"\n  ⚠ WARNING — Missing feature columns: {missing}")
    print("  → Attempting fuzzy column match …")
    FEAT_COLS = [c for c in FEAT_COLS if c in df.columns]

feat = df[FEAT_COLS].copy()

# ══════════════════════════════════════════════════════════════════════════════
# 1. DISTRIBUTION STATISTICS — 12 PERCEPTUAL FEATURES
# ══════════════════════════════════════════════════════════════════════════════
header("1. DISTRIBUTION STATISTICS — 12 PERCEPTUAL FEATURES")

desc = feat.describe(percentiles=[0.05, 0.25, 0.50, 0.75, 0.95]).T
desc["skewness"] = feat.skew()
desc["kurtosis"] = feat.kurtosis()
desc["null_count"] = feat.isnull().sum()
desc["null_pct"]   = (feat.isnull().sum() / len(feat) * 100).round(2)
desc["alias"]      = [ALIASES[c] for c in desc.index]

print(desc[["alias", "count", "mean", "std", "min", "5%", "25%", "50%",
            "75%", "95%", "max", "skewness", "kurtosis",
            "null_count", "null_pct"]].to_string())

# ══════════════════════════════════════════════════════════════════════════════
# 2. INFORMATION CONTENT CHECK
# ══════════════════════════════════════════════════════════════════════════════
header("2. INFORMATION CONTENT CHECK — Is Each Variable Informative?")

ZERO_THRESH    = 0.80   # flag if >80 % values are exactly zero
CONST_THRESH   = 0.95   # flag if >95 % values share the same value
LOW_STD_THRESH = 0.01   # flag if std < 0.01 (near-constant)

print(f"\n  {'Feature':<32} {'Zero%':>7} {'TopVal%':>8} {'Std':>8} {'Unique':>7}  Status")
print(f"  {'-'*32} {'-'*7} {'-'*8} {'-'*8} {'-'*7}  {'------'}")

for col in FEAT_COLS:
    s          = feat[col].dropna()
    zero_pct   = (s == 0).mean() * 100
    top_pct    = s.value_counts(normalize=True).iloc[0] * 100
    std_val    = s.std()
    n_unique   = s.nunique()

    flags = []
    if zero_pct > ZERO_THRESH * 100:
        flags.append("HIGH-ZERO")
    if top_pct > CONST_THRESH * 100:
        flags.append("NEAR-CONST")
    if std_val < LOW_STD_THRESH:
        flags.append("LOW-STD")
    if n_unique <= 2:
        flags.append("BINARY?")

    status = "✗  " + " | ".join(flags) if flags else "✔  OK"
    print(f"  {ALIASES[col]:<32} {zero_pct:>6.1f}% {top_pct:>7.1f}% {std_val:>8.4f} {n_unique:>7}  {status}")

# Shapiro-Wilk normality (sample ≤5000 for speed)
print("\n  Normality (Shapiro-Wilk, H0: normal, α=0.05):")
print(f"  {'Feature':<32} {'W':>8} {'p-value':>10}  Decision")
print(f"  {'-'*32} {'-'*8} {'-'*10}  {'--------'}")

SAMPLE = min(5000, len(feat))
sample_feat = feat.sample(SAMPLE, random_state=42) if len(feat) > SAMPLE else feat

for col in FEAT_COLS:
    s = sample_feat[col].dropna()
    if len(s) < 3:
        print(f"  {ALIASES[col]:<32}   {'N/A':>8}   {'N/A':>10}  insufficient data")
        continue
    w, p = stats.shapiro(s)
    decision = "NON-NORMAL" if p < 0.05 else "normal"
    print(f"  {ALIASES[col]:<32} {w:>8.4f} {p:>10.4e}  {decision}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. CORRELATION MATRIX — 12 FEATURES
# ══════════════════════════════════════════════════════════════════════════════
header("3. CORRELATION MATRIX — 12 PERCEPTUAL FEATURES (Pearson)")

corr = feat.rename(columns=ALIASES).corr()
print("\n  Full Correlation Matrix:")
print(corr.round(3).to_string())

# Flag high correlations
subheader("3a. High Correlations (|r| ≥ 0.70)")
high_pairs = []
cols = corr.columns.tolist()
for i in range(len(cols)):
    for j in range(i+1, len(cols)):
        r = corr.iloc[i, j]
        if abs(r) >= 0.70:
            high_pairs.append((cols[i], cols[j], r))

if high_pairs:
    print(f"\n  {'Variable A':<20} {'Variable B':<20} {'r':>8}  ⚠ Concern")
    for a, b, r in sorted(high_pairs, key=lambda x: abs(x[2]), reverse=True):
        flag = "VERY HIGH" if abs(r) >= 0.90 else "HIGH"
        print(f"  {a:<20} {b:<20} {r:>8.3f}  {flag}")
else:
    print("No pairs with |r| ≥ 0.70 found.")

subheader("3b. Correlation Summary Statistics")
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
flat  = upper.stack()
print(f"  Pairs evaluated  : {len(flat)}")
print(f"  Mean |r|         : {flat.abs().mean():.3f}")
print(f"  Max  |r|         : {flat.abs().max():.3f}  ({flat.abs().idxmax()})")
print(f"  |r| ≥ 0.50       : {(flat.abs() >= 0.50).sum()} pairs")
print(f"  |r| ≥ 0.70       : {(flat.abs() >= 0.70).sum()} pairs")
print(f"  |r| ≥ 0.90       : {(flat.abs() >= 0.90).sum()} pairs")

# Spearman as robustness check
corr_sp = feat.rename(columns=ALIASES).corr(method="spearman")
diff = (corr - corr_sp).abs()
upper_diff = diff.where(np.triu(np.ones(diff.shape), k=1).astype(bool)).stack()
print(f"\n  Pearson vs Spearman max |Δr|: {upper_diff.max():.3f} — "
      + ("non-linearity detected, prefer Spearman" if upper_diff.max() > 0.15 else "linear structure holds"))

# ══════════════════════════════════════════════════════════════════════════════
# 4. COMPOSITE DIMENSION DISTRIBUTIONS
# ══════════════════════════════════════════════════════════════════════════════
header("4. COMPOSITE DIMENSION DISTRIBUTIONS — 5 Dimensions")

# Re-compute dimensions from raw features (in case CSV values differ)
def compute_dimensions(d):
    g  = d["percept_greenery"]
    sk = d["percept_sky_visibility"]
    bf = d["percept_building_frontage"]
    gs = d["percept_ground_surface"]
    lp = d["percept_lighting_presence"]
    pp = d["percept_pedestrian_presence"]
    vp = d["percept_vehicle_presence"]
    sd = d["percept_signage_density"]
    sf = d["percept_street_furniture"]
    av = d["percept_architectural_variety"]
    ad = d["percept_activity_diversity"]
    sh = d["percept_shading_coverage"]

    vibrancy     = 0.45*pp + 0.30*bf + 0.20*sd + 0.05*ad
    pleasantness = 0.40*g  + 0.30*sk + 0.20*av - 0.30*vp
    walkability  = 0.35*gs + 0.40*sh + 0.15*g  - 0.10*vp
    safety       = 0.30*lp + 0.30*pp + 0.20*sk + 0.20*sf
    experiential = 0.45*av + 0.10*ad + 0.25*sd + 0.20*sf
    return vibrancy, pleasantness, walkability, safety, experiential

v, p, w, s, e = compute_dimensions(feat)
dims = pd.DataFrame({
    "vibrancy": v, "pleasantness": p, "walkability": w,
    "safety": s, "experiential": e
})

# Check against CSV columns if present
if all(c in df.columns for c in DIM_COLS):
    csv_dims = df[DIM_COLS]
    subheader("4a. Recomputed vs CSV Dimensions — Max Absolute Difference")
    for col in DIM_COLS:
        diff_max = (dims[col] - csv_dims[col]).abs().max()
        match    = "MATCH" if diff_max < 1e-6 else f"⚠ DIFF (max={diff_max:.6f})"
        print(f"  {col:<20}: {match}")
    dims_use = dims  # use recomputed as source of truth
else:
    print("  (Dimension columns not found in CSV — using recomputed values)")
    dims_use = dims

subheader("4b. Distribution Statistics")
desc_dim = dims_use.describe(percentiles=[0.05, 0.25, 0.50, 0.75, 0.95]).T
desc_dim["skewness"] = dims_use.skew()
desc_dim["kurtosis"] = dims_use.kurtosis()
print(desc_dim[["count","mean","std","min","5%","25%","50%","75%","95%","max",
                "skewness","kurtosis"]].round(4).to_string())

subheader("4c. Range Check — Theoretical vs Observed")
# Weights definition for range estimation
WEIGHT_RANGES = {
    "vibrancy":     {"min": 0.45*0+0.30*0+0.20*0+0.05*0,
                     "max": 0.45*1+0.30*1+0.20*1+0.05*1},
    "pleasantness": {"min": 0.40*0+0.30*0+0.20*0-0.30*1,
                     "max": 0.40*1+0.30*1+0.20*1-0.30*0},
    "walkability":  {"min": 0.35*0+0.40*0+0.15*0-0.10*1,
                     "max": 0.35*1+0.40*1+0.15*1-0.10*0},
    "safety":       {"min": 0.30*0+0.30*0+0.20*0+0.20*0,
                     "max": 0.30*1+0.30*1+0.20*1+0.20*1},
    "experiential": {"min": 0.45*0+0.25*0+0.25*0+0.20*0,
                     "max": 0.45*1+0.25*1+0.25*1+0.20*1},
}

print(f"\n  {'Dimension':<16} {'Theor.Min':>10} {'Theor.Max':>10} "
      f"{'Obs.Min':>10} {'Obs.Max':>10}  Status")
print(f"  {'-'*16} {'-'*10} {'-'*10} {'-'*10} {'-'*10}  {'------'}")
for dim in DIM_COLS:
    tmin = WEIGHT_RANGES[dim]["min"]
    tmax = WEIGHT_RANGES[dim]["max"]
    omin = dims_use[dim].min()
    omax = dims_use[dim].max()
    out  = omin < tmin - 1e-6 or omax > tmax + 1e-6
    flag = "⚠ OUT OF RANGE" if out else "✔ OK"
    print(f"  {dim:<16} {tmin:>10.3f} {tmax:>10.3f} {omin:>10.4f} {omax:>10.4f}  {flag}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. SCALE CONSISTENCY CHECK
# ══════════════════════════════════════════════════════════════════════════════
header("5. SCALE CONSISTENCY CHECK")

print(f"\n  Expected scale: [0, 1] for all 12 features\n")
print(f"  {'Feature':<32} {'Min':>8} {'Max':>8} {'Mean':>8} {'% > 1':>8}  Status")
print(f"  {'-'*32} {'-'*8} {'-'*8} {'-'*8} {'-'*8}  {'------'}")

all_ok = True
for col in FEAT_COLS:
    s       = feat[col].dropna()
    mn, mx  = s.min(), s.max()
    pct_gt1 = (s > 1.0).mean() * 100
    pct_lt0 = (s < 0.0).mean() * 100
    flags   = []
    if mx > 1.0:  flags.append(f"{pct_gt1:.1f}% > 1")
    if mn < 0.0:  flags.append(f"{pct_lt0:.1f}% < 0")
    status  = (" " + " | ".join(flags)) if flags else " [0,1]"
    if flags: all_ok = False
    print(f"  {ALIASES[col]:<32} {mn:>8.4f} {mx:>8.4f} {s.mean():>8.4f} {pct_gt1:>7.1f}%  {status}")

if all_ok:
    print("\n All features are within [0, 1] — scale consistent.")
else:
    print("\n Scale issues detected. Consider clipping or re-normalising.")

# ══════════════════════════════════════════════════════════════════════════════
# 6. DIMENSION CORRELATION MATRIX
# ══════════════════════════════════════════════════════════════════════════════
header("6. COMPOSITE DIMENSION CORRELATION MATRIX (Pearson)")

dim_corr = dims_use.corr()
print("\n  Pearson Correlation:\n")
print(dim_corr.round(3).to_string())

dim_corr_sp = dims_use.corr(method="spearman")
print("\n  Spearman Correlation:\n")
print(dim_corr_sp.round(3).to_string())

subheader("6a. High Dimension Correlations (|r| ≥ 0.70)")
dim_cols = dims_use.columns.tolist()
high_dim = []
for i in range(len(dim_cols)):
    for j in range(i+1, len(dim_cols)):
        r = dim_corr.iloc[i, j]
        if abs(r) >= 0.70:
            high_dim.append((dim_cols[i], dim_cols[j], r))

if high_dim:
    print(f"\n  {'Dimension A':<18} {'Dimension B':<18} {'r':>8}  Concern")
    for a, b, r in sorted(high_dim, key=lambda x: abs(x[2]), reverse=True):
        flag = "VERY HIGH — consider merging" if abs(r) >= 0.90 else "HIGH — review overlap"
        print(f"  {a:<18} {b:<18} {r:>8.3f}  {flag}")
else:
    print("  No dimension pairs with |r| ≥ 0.70 — dimensions are reasonably orthogonal.")

# ══════════════════════════════════════════════════════════════════════════════
# 7. PCA — 12 FEATURES
# ══════════════════════════════════════════════════════════════════════════════
header("7. PCA — 12 PERCEPTUAL FEATURES")

feat_clean = feat.dropna()
scaler     = MinMaxScaler()
feat_sc    = scaler.fit_transform(feat_clean)

pca12  = PCA()
pca12.fit(feat_sc)
ev     = pca12.explained_variance_ratio_
cum_ev = np.cumsum(ev)

print(f"\n  {'PC':<5} {'Eigenvalue':>12} {'Var%':>8} {'CumVar%':>10}  {'Bar':<30}")
print(f"  {'-'*5} {'-'*12} {'-'*8} {'-'*10}  {'-'*30}")
for i, (e_val, var, cum) in enumerate(zip(pca12.explained_variance_, ev, cum_ev)):
    bar   = "█" * int(var * 200)
    cross = "◀ 80%" if (cum >= 0.80 and (i == 0 or cum_ev[i-1] < 0.80)) else \
            "◀ 90%" if (cum >= 0.90 and (i == 0 or cum_ev[i-1] < 0.90)) else \
            "◀ 95%" if (cum >= 0.95 and (i == 0 or cum_ev[i-1] < 0.95)) else ""
    print(f"  PC{i+1:<3} {e_val:>12.4f} {var*100:>7.2f}% {cum*100:>9.2f}%  {bar}  {cross}")

n80  = np.searchsorted(cum_ev, 0.80) + 1
n90  = np.searchsorted(cum_ev, 0.90) + 1
n95  = np.searchsorted(cum_ev, 0.95) + 1
n_kaiser = (pca12.explained_variance_ > 1).sum()

print(f"\n  PCs to explain 80%  : {n80}")
print(f"  PCs to explain 90%  : {n90}")
print(f"  PCs to explain 95%  : {n95}")
print(f"  Kaiser rule (λ > 1) : {n_kaiser} components")
print(f"  → Intrinsic dimensionality ≈ {n90} (90% threshold)")

subheader("7a. PC Loadings (top 12 features × first 5 PCs)")
loadings = pd.DataFrame(
    pca12.components_[:5].T,
    index   = [ALIASES[c] for c in FEAT_COLS],
    columns = [f"PC{i+1}" for i in range(5)]
)
print("\n  Loadings (|loading| ≥ 0.30 are meaningful):\n")
print(loadings.round(3).to_string())

subheader("7b. Key Feature–PC Associations (|loading| ≥ 0.35)")
for pc in loadings.columns:
    top = loadings[pc][loadings[pc].abs() >= 0.35].sort_values(key=abs, ascending=False)
    if not top.empty:
        parts = [f"{f}({v:+.2f})" for f, v in top.items()]
        print(f"  {pc}: {', '.join(parts)}")

# ══════════════════════════════════════════════════════════════════════════════
# 8. PCA — 5 COMPOSITE DIMENSIONS
# ══════════════════════════════════════════════════════════════════════════════
header("8. PCA — 5 COMPOSITE DIMENSIONS")

dim_clean = dims_use.dropna()
dim_sc    = MinMaxScaler().fit_transform(dim_clean)

pca5  = PCA()
pca5.fit(dim_sc)
ev5   = pca5.explained_variance_ratio_
cum5  = np.cumsum(ev5)

print(f"\n  {'PC':<5} {'Eigenvalue':>12} {'Var%':>8} {'CumVar%':>10}  {'Bar'}")
print(f"  {'-'*5} {'-'*12} {'-'*8} {'-'*10}  {'---'}")
for i, (e_val, var, cum) in enumerate(zip(pca5.explained_variance_, ev5, cum5)):
    bar = "█" * int(var * 200)
    print(f"  PC{i+1:<3} {e_val:>12.4f} {var*100:>7.2f}% {cum*100:>9.2f}%  {bar}")

n_kaiser5 = (pca5.explained_variance_ > 1).sum()
print(f"\n  Kaiser rule (λ > 1) : {n_kaiser5} components")
if cum5[0] >= 0.70:
    print(f"   PC1 explains {cum5[0]*100:.1f}% — dimensions share a strong common factor (urban quality?)")
if n_kaiser5 == 1:
    print("   Only 1 eigenvalue > 1 — consider a single composite index.")
elif n_kaiser5 <= 2:
    print("   Dimensions are highly collinear — validate construct distinctiveness.")

subheader("8a. Dimension Loadings on PC1–3")
loadings5 = pd.DataFrame(
    pca5.components_[:3].T,
    index   = DIM_COLS,
    columns = [f"PC{i+1}" for i in range(3)]
)
print(loadings5.round(3).to_string())

# ══════════════════════════════════════════════════════════════════════════════
# 9. COLLINEARITY CHECK — VIF
# ══════════════════════════════════════════════════════════════════════════════
header("9. COLLINEARITY CHECK — Variance Inflation Factor (VIF)")

try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    subheader("9a. VIF — 12 Perceptual Features")
    feat_nona = feat_clean.copy()
    feat_nona.columns = [ALIASES[c] for c in FEAT_COLS]
    feat_nona = feat_nona.astype(float)

    vif_data = pd.DataFrame()
    vif_data["Feature"] = feat_nona.columns
    vif_data["VIF"]     = [variance_inflation_factor(feat_nona.values, i)
                           for i in range(feat_nona.shape[1])]
    vif_data["Status"]  = vif_data["VIF"].apply(
        lambda v: " OK" if v < 5 else ("⚠ MODERATE" if v < 10 else "✗ HIGH (>10)"))
    print(f"\n  VIF < 5: acceptable │ 5–10: moderate │ >10: problematic\n")
    print(vif_data.sort_values("VIF", ascending=False).to_string(index=False))

    subheader("9b. VIF — 5 Composite Dimensions")
    dim_nona = dim_clean.copy().astype(float)
    vif_dim  = pd.DataFrame()
    vif_dim["Dimension"] = dim_nona.columns
    vif_dim["VIF"]       = [variance_inflation_factor(dim_nona.values, i)
                            for i in range(dim_nona.shape[1])]
    vif_dim["Status"]    = vif_dim["VIF"].apply(
        lambda v: " OK" if v < 5 else ("⚠ MODERATE" if v < 10 else "✗ HIGH (>10)"))
    print(f"\n  VIF < 5: acceptable │ 5–10: moderate │ >10: problematic\n")
    print(vif_dim.sort_values("VIF", ascending=False).to_string(index=False))

except ImportError:
    print("  ⚠ statsmodels not installed. Run: pip install statsmodels")
    print("  Falling back to Condition Number …\n")

    # Condition number as fallback
    for label, mat in [("12 Features", feat_sc), ("5 Dimensions", dim_sc)]:
        eigenvalues = np.linalg.eigvalsh(np.cov(mat.T))
        eigenvalues = eigenvalues[eigenvalues > 0]
        cond = np.sqrt(eigenvalues.max() / eigenvalues.min())
        status = " OK" if cond < 30 else (" MODERATE" if cond < 100 else " HIGH")
        print(f"  Condition Number ({label}): {cond:.1f}  {status}")

# ══════════════════════════════════════════════════════════════════════════════
# 10. SUMMARY SCORECARD
# ══════════════════════════════════════════════════════════════════════════════
header("10. SANITY CHECK SUMMARY SCORECARD")

print("""
  ┌─────────────────────────────────────────────────────────────────┐
  │  #   Check                         Key Findings                  │
  ├─────────────────────────────────────────────────────────────────┤
  │  1   Distribution Stats            Review skewness / kurtosis    │
  │  2   Information Content           Check HIGH-ZERO / NEAR-CONST  │
  │  3   Feature Correlation           Flag |r| ≥ 0.70 pairs         │
  │  4   Dimension Distributions       Range & normality check       │
  │  5   Scale Consistency             All features in [0,1]?        │
  │  6   Dimension Correlation         Orthogonality of constructs   │
  │  7   PCA (12 features)             Intrinsic dimensionality      │
  │  8   PCA (5 dimensions)            Shared factor / redundancy    │
  │  9   VIF Collinearity              Multicollinearity risk        │
  └─────────────────────────────────────────────────────────────────┘

  Interpretation guide
  ──────────────────────────────────────────────────────────────────
  • Features with HIGH-ZERO (>80%): if due to true absence (e.g., no
    traffic lights in residential areas) this is valid; if due to
    detection failure, consider reprocessing.

  • High feature correlation (|r|≥0.70): check whether two features
    measure the same construct — possible redundancy.

  • PCA: if 90% variance is explained by ≤4 PCs out of 12 features,
    the feature space has moderate redundancy but is not degenerate.

  • High dimension VIF (>10): dimensions share too much variance;
    consider orthogonalisation or dropping a dimension.

  • pleasantness can be negative (due to −0.30×vehicle_presence);
    document this and clip if downstream models require [0,1].
""")

print(f"\n  Script complete. All checks printed above.\n")
