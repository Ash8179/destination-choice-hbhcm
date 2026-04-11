"""
E.5 – Exploratory Factor Analysis (EFA)

This script:
Perform 4-factor EFA with pre-screening (item-total correlation),
KMO/Bartlett tests, varimax rotation, and loading visualization.

Author: Zhang Wenyu
Date: 2026-03-25
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
import warnings
import os
warnings.filterwarnings("ignore")

# ── Output directories ─────────────────────────────────────────────
os.makedirs("/Users/zhangwenyu/Desktop/NUSFYP/Stage 2/mnt/user-data/outputs", exist_ok=True)

# ── 1. Load data ─────────────────────────────────────────────────────
paths_to_try = [
    "/Users/zhangwenyu/Desktop/NUSFYP/Stage 2/EFA Analysis 4.csv",
    "/Users/zhangwenyu/Desktop/NUSFYP/Stage 2/mnt/user-data/uploads/EFA Analysis 3.csv",
    "EFA Analysis 3.csv",
]
df_raw = None
for p in paths_to_try:
    if os.path.exists(p):
        df_raw = pd.read_csv(p, sep=None, engine="python")
        print(f"Loaded from: {p}")
        break
if df_raw is None:
    raise FileNotFoundError("Cannot find EFA Analysis 2.csv. Place it in the proper folder.")

print(f"Raw rows: {len(df_raw)}")

# ── 2. Attention check filter ─────────────────────────────────────────
col_attn = "Q2_24"
df = df_raw[df_raw[col_attn] == 4].copy()
print(f"After attention check filter: {len(df)} respondents")

# ── 3. Keep only 16 items for 4-factor EFA ────────────────────────────
deleted_items = ["Q2_1","Q2_2","Q2_6","Q2_11","Q2_14","Q2_17","Q2_13","Q2_15","Q2_18", "Q2_21"]
item_cols = [f"Q2_{i}" for i in range(1, 24) if f"Q2_{i}" not in deleted_items]
df = df[item_cols].copy().dropna()
print(f"After dropping NaN rows: {len(df)} respondents\n")

# ── 4. Reverse coding ─────────────────────────────────────────────────
REVERSE_ITEMS = ["Q2_2", "Q2_6", "Q2_14", "Q2_17"]
scale_max = 5
for col in REVERSE_ITEMS:
    if col in df.columns:
        df[col] = scale_max + 1 - df[col]
        print(f"  Reverse-coded: {col}")

# ── 5. Pre-screening: Item-Total Correlations ─────────────────────────
print("\n" + "="*70)
print("STEP 5 — Pre-screening: Item-Total Correlations")
print("="*70)

item_total_corr = {}
for col in item_cols:
    rest = df[[c for c in item_cols if c != col]].sum(axis=1)
    r, _ = stats.pearsonr(df[col], rest)
    item_total_corr[col] = r

itc_df = pd.DataFrame.from_dict(item_total_corr, orient="index", columns=["Item-Total r"])
itc_df["Flag"] = itc_df["Item-Total r"].apply(lambda x: "⚠ LOW (<0.20)" if x < 0.20 else "OK")
print(itc_df.to_string())
low_items = itc_df[itc_df["Item-Total r"] < 0.20].index.tolist()
print(f"\nItems flagged as potentially problematic (r < 0.20): {low_items}")

# ── 6. KMO & Bartlett's Test ──────────────────────────────────────────
print("\n" + "="*70)
print("STEP 6 — KMO & Bartlett's Test of Sphericity")
print("="*70)

kmo_all, kmo_model = calculate_kmo(df[item_cols])
chi2, p_value = calculate_bartlett_sphericity(df[item_cols])
print(f"  KMO (overall)     : {kmo_model:.3f}  {'✓ Adequate (≥.60)' if kmo_model >= 0.60 else '✗ Poor (<.60)'}")
print(f"  Bartlett χ²       : {chi2:.2f}  p = {p_value:.4f}  "
      f"{'✓ Significant' if p_value < 0.05 else '✗ Not significant'}")

kmo_item_df = pd.DataFrame({"Item": item_cols,
                            "MSA": kmo_all,
                            "Flag": ["⚠ DROP (<.50)" if k < 0.50 else "OK" for k in kmo_all]})
print("\nPer-item KMO (MSA):")
print(kmo_item_df.to_string(index=False))
low_msa = kmo_item_df[kmo_item_df["MSA"] < 0.50]["Item"].tolist()
print(f"\nItems with MSA < 0.50 (consider dropping): {low_msa}")

# ── 7. Fixed 4-factor EFA ─────────────────────────────────────────────
n_factors = 4

print("\n" + "="*70)
print(f"EFA — Version 3 (Fixed {n_factors} factors, varimax)")
print("="*70)

fa = FactorAnalyzer(n_factors=n_factors, rotation="varimax",
                    method="ml", use_smc=True)
fa.fit(df[item_cols])

loadings = pd.DataFrame(fa.loadings_, index=item_cols, columns=[f"F{i+1}" for i in range(n_factors)])
loadings["MaxLoading"] = loadings.abs().max(axis=1)
loadings["DomFactor"] = loadings.abs().idxmax(axis=1)
loadings["Flag"] = loadings["MaxLoading"].apply(
    lambda x: "✗ WEAK (<.35)" if x < 0.35 else ("⚠ BORDER" if x < 0.40 else "✓"))

print("\nFactor Loadings (varimax):")
fmt_cols = [f"F{i+1}" for i in range(n_factors)] + ["MaxLoading", "DomFactor", "Flag"]
print(loadings[fmt_cols].round(3).to_string())

var_df = pd.DataFrame(fa.get_factor_variance(), index=["SS Loadings", "Prop Var", "Cumul Var"],
                      columns=[f"F{i+1}" for i in range(n_factors)])
print("\nVariance Explained:")
print(var_df.round(3).to_string())

# ── 8. Factor correlation matrix ─────────────────────────────────────
phi = np.corrcoef(fa.transform(df[item_cols]).T)
print("\nFactor Correlation Matrix (should be near 0 for varimax):")
print(pd.DataFrame(phi, index=[f"F{i+1}" for i in range(n_factors)],
                   columns=[f"F{i+1}" for i in range(n_factors)]).round(3).to_string())

# ── 9. Suggested item groupings ──────────────────────────────────────
print("\nSuggested item groupings by dominant factor:")
for f in [f"F{i+1}" for i in range(n_factors)]:
    items_in_f = loadings[loadings["DomFactor"] == f].index.tolist()
    print(f"  {f}: {items_in_f}")

# ── 10. Loading heatmap ──────────────────────────────────────────────
def plot_loading_heatmap(loadings_df, n_factors, title, out_path):
    factor_cols = [f"F{i+1}" for i in range(n_factors)]
    mat = loadings_df[factor_cols].values
    items = loadings_df.index.tolist()

    fig, ax = plt.subplots(figsize=(max(6, n_factors*1.4), max(6, len(items)*0.42)))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.03)

    ax.set_xticks(range(n_factors))
    ax.set_xticklabels(factor_cols, fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels(items, fontsize=9)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)

    for i in range(len(items)):
        for j in range(n_factors):
            val = mat[i, j]
            color = "white" if abs(val) > 0.55 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=color)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\n  Loading heatmap saved → {out_path}")

plot_loading_heatmap(loadings, n_factors,
                     f"EFA Loadings v3 (varimax, {n_factors} factors)",
                     "/Users/zhangwenyu/Desktop/NUSFYP/Stage 2/mnt/user-data/outputs/efa_loadings_v3.png")

print("\nDone. Output files in /mnt/user-data/outputs/")
