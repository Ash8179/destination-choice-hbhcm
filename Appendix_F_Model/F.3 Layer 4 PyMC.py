"""
================================================================================
F.3 - HBHCM — Layer 4: OD Aggregate Likelihood (NegBin) via PyMC
================================================================================

PURPOSE
-------
Estimate the CV perceptual effects (delta_CV) on real destination visit flow
(proxied by Google Maps rating_count), using OD flow calibration tables
produced by od_flow_computation_v6.py.

KEY DESIGN DECISIONS
--------------------
1. mu_j_no_scale is used as the base offset (NOT mu_j_prior_base).
   mu_j_no_scale was computed with only BETA_GTC_PRIOR * GTC, without any
   beta_S contribution. This prevents beta_S from being counted twice —
   once inside mu_j and again in scale_eff.

2. beta_gtc is NOT estimated in Layer 4.
   Its contribution is already baked into mu_base via BETA_GTC_PRIOR.
   beta_gtc carries no OD-identifiable signal and is not needed for
   Layer 5 handoff — Layer 5 inherits it directly from the SP posterior.

3. beta_s_med and beta_s_lrg carry SP-posterior informative priors and are
   freely estimated by OD flow. This is the cross-layer calibration point.

4. delta_cv[0..4] maps to:
   [vibrancy, pleasantness, walkability, safety, experiential]
   These are identified exclusively by OD flow, not SP.

5. cat_fe uses FE=0 (mall) as reference (alpha_mall equiv 0 per FIX 1).
   Free parameters: alpha_cat[0]=hawker, [1]=cultural, [2]=monument, [3]=park.

INPUTS
------
  dest_calibration_table_v6.csv  — one row per destination (from od_flow v6)
  Required columns:
    rating_count, mu_j_no_scale, D_med, D_lrg,
    cat_fe_2, cat_fe_3, cat_fe_4, cat_fe_5,
    vibrancy_robust_z, pleasantness_robust_z, walkability_robust_z,
    safety_robust_z, experiential_robust_z

OUTPUTS
-------
  layer4_trace.nc                — ArviZ InferenceData (full posterior)
  layer4_summary.csv             — posterior summary (mean, sd, HDI, r_hat)
  layer4_delta_cv_posteriors.csv — delta_CV posteriors formatted for Layer 5
  layer4_diagnostics.txt         — convergence and fit diagnostics

LAYER 5 HANDOFF
---------------
The following posteriors from this model feed Layer 5 as informative priors:
  beta_s_med  — updated by OD flow (shared with SP)
  beta_s_lrg  — updated by OD flow (shared with SP)
  delta_cv    — identified here only; weakly informed in Layer 5
  beta_gtc    — NOT estimated here; Layer 5 inherits from SP posterior only
  
Author: Zhang Wenyu
Date: 2026-03-27
"""

import os
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================================
# SECTION 0 — CONFIGURATION
# =============================================================================

BASE_DIR   = os.path.join(os.path.expanduser("~"), "Desktop", "NUSFYP",
                          "Stage 2", "HBHCM", "Layer 4")
INPUT_CSV  = os.path.join(BASE_DIR, "dest_calibration_table_v6.csv")
OUTPUT_DIR = BASE_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

# SP posterior estimates — plug in values from iclv_v1_no_wfh_utility.py output
# beta_gtc constants retained here for documentation purposes only.
# They are NOT used to define any PyMC node in Layer 4.
SP_BETA_GTC_MU    = -0.0074
SP_BETA_GTC_SD    =  0.0016

SP_BETA_S_MED_MU  =  0.4485
SP_BETA_S_MED_SD  =  0.1455

SP_BETA_S_LRG_MU  =  0.6691
SP_BETA_S_LRG_SD  =  0.1367

# MCMC settings
N_DRAWS      = 2000
N_TUNE       = 1000
TARGET_ACCEPT= 0.90
RANDOM_SEED  = 42
N_CHAINS     = 4

# CV dimension labels — must match column order in Z_cv below
CV_DIM_NAMES = ["vibrancy", "pleasantness", "walkability", "safety", "experiential"]

print("=" * 70)
print("HBHCM Layer 4 — OD Aggregate Likelihood (NegBin)")
print("=" * 70)
print(f"\nInput : {INPUT_CSV}")
print(f"Output: {OUTPUT_DIR}")

# =============================================================================
# SECTION 1 — DATA LOADING AND VALIDATION
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 1 — Data Loading")
print("=" * 70)

dest_raw = pd.read_csv(INPUT_CSV)
print(f"  Rows loaded              : {len(dest_raw)}")

# Keep only Layer 4 eligible destinations (rating_count > 0)
dest = dest_raw[dest_raw["rating_count"] > 0].copy().reset_index(drop=True)
print(f"  After rating_count > 0   : {len(dest)}")

# Verify mu_j_no_scale column exists
if "mu_j_no_scale" not in dest.columns:
    raise ValueError(
        "'mu_j_no_scale' column not found in calibration table.\n"
        "Run od_flow_computation_v6.py with the V_od_no_scale fix applied first.\n"
        "The column should be computed as:\n"
        "  V_od_no_scale = BETA_GTC_PRIOR * GTC_od  (no beta_S)\n"
        "  mu_j_no_scale = sum_o T_o * softmax(V_od_no_scale)"
    )

# Verify required cat_fe dummy columns
for fe_k in [2, 3, 4, 5]:
    col = f"cat_fe_{fe_k}"
    if col not in dest.columns:
        raise ValueError(f"Required column '{col}' not found. Check od_flow script output.")

print(f"\n  rating_count range       : [{dest['rating_count'].min()}, {dest['rating_count'].max()}]")
print(f"  mu_j_no_scale range      : [{dest['mu_j_no_scale'].min():.2f}, {dest['mu_j_no_scale'].max():.2f}]")
print(f"  mu_j_no_scale sum        : {dest['mu_j_no_scale'].sum():.1f}")
print(f"  rating_count sum         : {dest['rating_count'].sum()}")

# Category distribution check
print(f"\n  Destination category FE distribution (FE=0=mall is reference):")
for fe_k, label in [(2,"hawker"),(3,"cultural"),(4,"monument"),(5,"park+historic")]:
    n = dest[f"cat_fe_{fe_k}"].sum()
    print(f"    cat_fe_{fe_k} ({label:<14}): {n:>3} POIs")
n_ref = len(dest) - sum(dest[f"cat_fe_{k}"].sum() for k in [2,3,4,5])
print(f"    cat_fe_0 (mall ref      ): {n_ref:>3} POIs  [alpha_mall equiv 0]")

# Scale distribution
print(f"\n  Scale dummy distribution:")
print(f"    D_med=1 (medium): {dest['D_med'].sum()}")
print(f"    D_lrg=1 (large) : {dest['D_lrg'].sum()}")
n_small = ((dest["D_med"]==0) & (dest["D_lrg"]==0)).sum()
print(f"    small (ref)     : {n_small}")

# =============================================================================
# SECTION 2 — DATA MATRIX ASSEMBLY
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 2 — Data Matrix Assembly")
print("=" * 70)

# Observed visit proxy
y = dest["rating_count"].values.astype(float)

# Base offset — mu_j computed WITHOUT beta_S to avoid double-counting.
# beta_gtc's contribution is baked in at BETA_GTC_PRIOR level.
# Layer 4 only adds incremental effects: scale_eff + cat_effect + cv_effect.
mu_base = dest["mu_j_no_scale"].values.astype(float)

# CV perceptual features — 5 dimensions, standardised z-scores
Z_cv = dest[[
    "vibrancy_robust_z",
    "pleasantness_robust_z",
    "walkability_robust_z",
    "safety_robust_z",
    "experiential_robust_z",
]].values.astype(float)   # shape: (n_dest, 5)

# Scale dummies
D_med = dest["D_med"].values.astype(float)
D_lrg = dest["D_lrg"].values.astype(float)

# Category FE dummies (FE=0 is reference; no free parameter)
# Index mapping: alpha_cat[0]=hawker(2), [1]=cultural(3), [2]=monument(4), [3]=park(5)
fe_hawker    = dest["cat_fe_2"].values.astype(float)
fe_cultural  = dest["cat_fe_3"].values.astype(float)
fe_monument  = dest["cat_fe_4"].values.astype(float)
fe_park      = dest["cat_fe_5"].values.astype(float)

n_dest = len(dest)
print(f"  n_dest (Layer 4 sample)  : {n_dest}")
print(f"  Z_cv shape               : {Z_cv.shape}")
print(f"  mu_base (no scale) range : [{mu_base.min():.3f}, {mu_base.max():.3f}]")

# =============================================================================
# SECTION 3 — PyMC MODEL DEFINITION
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 3 — PyMC Model Definition")
print("=" * 70)

with pm.Model() as od_model:

    # ------------------------------------------------------------------
    # Shared parameters: informed by SP posterior
    #
    # NOTE: beta_gtc is intentionally absent from Layer 4.
    # Its contribution is already baked into mu_base via:
    #   V_od_no_scale = BETA_GTC_PRIOR * GTC_od  (od_flow_computation_v6.py)
    # beta_gtc carries no OD-identifiable signal here and is not needed
    # for Layer 5 handoff — Layer 5 inherits it directly from SP posterior.
    #
    # beta_s_med / beta_s_lrg: SP-informed AND updated by OD flow.
    # These DO enter log_mu via scale_eff — this is intentional.
    # ------------------------------------------------------------------
    beta_s_med = pm.Normal("beta_s_med",
                           mu=SP_BETA_S_MED_MU, sigma=SP_BETA_S_MED_SD)
    beta_s_lrg = pm.Normal("beta_s_lrg",
                           mu=SP_BETA_S_LRG_MU, sigma=SP_BETA_S_LRG_SD)

    # ------------------------------------------------------------------
    # OD-exclusive parameters
    # ------------------------------------------------------------------

    # CV perceptual effects — identified from cross-destination flow variation
    delta_cv = pm.Normal("delta_cv", mu=0, sigma=1, shape=5)

    # Category fixed effects — log-multipliers relative to mall (FE=0)
    # alpha_cat[0]=hawker, [1]=cultural, [2]=monument, [3]=park+historic
    alpha_cat = pm.Normal("alpha_cat", mu=0, sigma=1, shape=4)

    # NegBin overdispersion — smaller phi = more overdispersed
    phi = pm.HalfNormal("phi", sigma=2)

    # ------------------------------------------------------------------
    # Expected visit count mu_est
    #
    # log_mu = log(mu_base)    <- GTC effect fixed at BETA_GTC_PRIOR level
    #        + scale_eff       <- beta_S applied once here
    #        + cat_effect      <- category FE relative to mall
    #        + cv_effect       <- CV perceptual attributes
    #
    # ABSENT from log_mu (intentional):
    #   ✗ beta_gtc — baked into mu_base; not estimated in Layer 4
    # ------------------------------------------------------------------
    scale_eff  = beta_s_med * D_med + beta_s_lrg * D_lrg

    cat_effect = (alpha_cat[0] * fe_hawker
                + alpha_cat[1] * fe_cultural
                + alpha_cat[2] * fe_monument
                + alpha_cat[3] * fe_park)

    cv_effect  = pm.math.dot(Z_cv, delta_cv)

    log_mu  = pm.math.log(mu_base + 1e-9) + scale_eff + cat_effect + cv_effect
    mu_est  = pm.math.exp(log_mu)

    # ------------------------------------------------------------------
    # NegBin likelihood
    # rating_count_j ~ NegBin(mu_est_j, phi)
    # ------------------------------------------------------------------
    pm.NegativeBinomial("rating_obs", mu=mu_est, alpha=phi, observed=y)

    # ------------------------------------------------------------------
    # Prior predictive check (before sampling)
    # ------------------------------------------------------------------
    print("  Running prior predictive check...")
    prior_pred = pm.sample_prior_predictive(samples=200, random_seed=RANDOM_SEED)

print("  Model graph compiled successfully.")

# =============================================================================
# SECTION 4 — MCMC SAMPLING
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 4 — MCMC Sampling")
print(f"  Draws={N_DRAWS}, Tune={N_TUNE}, Chains={N_CHAINS}, "
      f"target_accept={TARGET_ACCEPT}")
print("=" * 70)

with od_model:
    trace = pm.sample(
        draws        = N_DRAWS,
        tune         = N_TUNE,
        chains       = N_CHAINS,
        target_accept= TARGET_ACCEPT,
        random_seed  = RANDOM_SEED,
        return_inferencedata = True,
        idata_kwargs={"log_likelihood": True},
    )
    # Posterior predictive
    print("\n  Running posterior predictive check...")
    post_pred = pm.sample_posterior_predictive(trace, random_seed=RANDOM_SEED)

# Attach prior predictive to trace
trace.extend(prior_pred)
trace.extend(post_pred)

# =============================================================================
# SECTION 5 — CONVERGENCE DIAGNOSTICS
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 5 — Convergence Diagnostics")
print("=" * 70)

summary = az.summary(trace, var_names=[
    "beta_s_med", "beta_s_lrg",   # beta_gtc removed — not estimated in Layer 4
    "delta_cv", "alpha_cat", "phi"
], round_to=4)

print(summary.to_string())

# Check R-hat
rhat_vals = summary["r_hat"].values
n_bad_rhat = (rhat_vals > 1.05).sum()
n_warn_rhat= (rhat_vals > 1.01).sum()
print(f"\n  R-hat > 1.05 (convergence failure) : {n_bad_rhat}")
print(f"  R-hat > 1.01 (marginal)            : {n_warn_rhat}")
if n_bad_rhat > 0:
    print("  WARNING: Some chains did not converge. Consider increasing tune steps.")
else:
    print("  OK: All R-hat values acceptable.")

# Check ESS
ess_bulk = summary["ess_bulk"].values
n_low_ess = (ess_bulk < 400).sum()
print(f"  ESS_bulk < 400 (low effective samples): {n_low_ess}")
if n_low_ess > 0:
    print("  WARNING: Low ESS. Consider increasing N_DRAWS.")
else:
    print("  OK: ESS sufficient.")

# =============================================================================
# SECTION 6 — POSTERIOR SUMMARY AND INTERPRETATION
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 6 — Posterior Summary")
print("=" * 70)

# delta_CV posteriors
delta_post = trace.posterior["delta_cv"].values   # shape: (chains, draws, 5)
delta_flat = delta_post.reshape(-1, 5)             # (chains*draws, 5)

delta_summary = pd.DataFrame({
    "CV_dimension": CV_DIM_NAMES,
    "mean"        : delta_flat.mean(axis=0).round(4),
    "sd"          : delta_flat.std(axis=0).round(4),
    "hdi_3%"      : np.percentile(delta_flat, 3,  axis=0).round(4),
    "hdi_97%"     : np.percentile(delta_flat, 97, axis=0).round(4),
    "sign_prob"   : (delta_flat > 0).mean(axis=0).round(3),
})
delta_summary["interpretation"] = delta_summary["sign_prob"].apply(
    lambda p: "positive ✓" if p > 0.90 else
              "negative ✓" if p < 0.10 else
              "uncertain"
)
print("\n  delta_CV posteriors (CV effect on OD flow):")
print(delta_summary.to_string(index=False))

# beta_S posteriors (cross-layer calibration)
bs_med_post = trace.posterior["beta_s_med"].values.flatten()
bs_lrg_post = trace.posterior["beta_s_lrg"].values.flatten()

print(f"\n  beta_s_med posterior: mean={bs_med_post.mean():.4f}  "
      f"sd={bs_med_post.std():.4f}  "
      f"[{np.percentile(bs_med_post,3):.4f}, {np.percentile(bs_med_post,97):.4f}]")
print(f"  beta_s_lrg posterior: mean={bs_lrg_post.mean():.4f}  "
      f"sd={bs_lrg_post.std():.4f}  "
      f"[{np.percentile(bs_lrg_post,3):.4f}, {np.percentile(bs_lrg_post,97):.4f}]")
print(f"  (SP prior:  med={SP_BETA_S_MED_MU:.3f}, lrg={SP_BETA_S_LRG_MU:.3f})")
print(f"  OD update shift: "
      f"med={bs_med_post.mean()-SP_BETA_S_MED_MU:+.4f}  "
      f"lrg={bs_lrg_post.mean()-SP_BETA_S_LRG_MU:+.4f}")

# alpha_cat posteriors
alpha_post = trace.posterior["alpha_cat"].values.reshape(-1, 4)
cat_labels = ["hawker", "cultural", "monument", "park+historic"]
print(f"\n  alpha_cat posteriors (log-multiplier vs mall):")
for i, lbl in enumerate(cat_labels):
    m, s = alpha_post[:,i].mean(), alpha_post[:,i].std()
    lo, hi = np.percentile(alpha_post[:,i],[3,97])
    print(f"    {lbl:<16}: mean={m:+.3f}  sd={s:.3f}  [{lo:+.3f}, {hi:+.3f}]")

# phi
phi_post = trace.posterior["phi"].values.flatten()
print(f"\n  phi (NegBin overdispersion): mean={phi_post.mean():.3f}  "
      f"sd={phi_post.std():.3f}")

# =============================================================================
# SECTION 7 — FIT DIAGNOSTICS (posterior predictive)
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 7 — Fit Diagnostics")
print("=" * 70)

y_hat = trace.posterior_predictive["rating_obs"].values   # (chains, draws, n_dest)
y_hat_mean = y_hat.reshape(-1, n_dest).mean(axis=0)
y_hat_sd   = y_hat.reshape(-1, n_dest).std(axis=0)

resid = y - y_hat_mean
mae   = np.abs(resid).mean()
rmse  = np.sqrt((resid**2).mean())
corr  = np.corrcoef(y, y_hat_mean)[0,1]

print(f"  Posterior predictive vs observed rating_count:")
print(f"    Pearson r(y, y_hat_mean)  : {corr:.4f}")
print(f"    MAE                       : {mae:.2f}")
print(f"    RMSE                      : {rmse:.2f}")
print(f"    Mean(y)                   : {y.mean():.1f}")
print(f"    Mean(y_hat)               : {y_hat_mean.mean():.1f}")

# Bayesian R² (Gelman et al.)
var_fit = y_hat_mean.var()
var_res = resid.var()
bayes_r2 = var_fit / (var_fit + var_res)
print(f"    Bayesian R²               : {bayes_r2:.4f}")

# LOO-CV if feasible
try:
    loo = az.loo(trace, var_name="rating_obs")
    print(f"\n  LOO-CV:")
    print(f"    ELPD_LOO : {loo.elpd_loo:.2f}")
    print(f"    p_LOO    : {loo.p_loo:.2f}")
    print(f"    Pareto-k > 0.7 fraction: "
          f"{(loo.pareto_k > 0.7).mean():.1%}")
except Exception as e:
    print(f"\n  LOO-CV skipped: {e}")

# =============================================================================
# SECTION 8 — OUTPUTS
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 8 — Saving Outputs")
print("=" * 70)

# 8a. Full InferenceData (NetCDF)
trace_path = os.path.join(OUTPUT_DIR, "layer4_trace.nc")
trace.to_netcdf(trace_path)
print(f"  Full trace saved  -> {trace_path}")

# 8b. Posterior summary CSV
summary_path = os.path.join(OUTPUT_DIR, "layer4_summary.csv")
summary.to_csv(summary_path)
print(f"  Summary CSV saved -> {summary_path}")

# 8c. delta_CV posteriors formatted for Layer 5 handoff
delta_l5 = pd.DataFrame({
    "parameter"  : [f"delta_cv_{d}" for d in CV_DIM_NAMES],
    "prior_mu"   : delta_flat.mean(axis=0).round(5),
    "prior_sigma": delta_flat.std(axis=0).round(5),
    "hdi_3pct"   : np.percentile(delta_flat, 3,  axis=0).round(5),
    "hdi_97pct"  : np.percentile(delta_flat, 97, axis=0).round(5),
    "source"     : "Layer4_OD_NegBin",
    "note"       : "Use prior_mu/sigma as Normal prior in Layer 5 joint model",
})
# Also include beta_S updated posteriors
beta_s_l5 = pd.DataFrame({
    "parameter"  : ["beta_s_med", "beta_s_lrg"],
    "prior_mu"   : [bs_med_post.mean().round(5), bs_lrg_post.mean().round(5)],
    "prior_sigma": [bs_med_post.std().round(5),  bs_lrg_post.std().round(5)],
    "hdi_3pct"   : [np.percentile(bs_med_post,3).round(5), np.percentile(bs_lrg_post,3).round(5)],
    "hdi_97pct"  : [np.percentile(bs_med_post,97).round(5),np.percentile(bs_lrg_post,97).round(5)],
    "source"     : "Layer4_OD_NegBin",
    "note"       : "SP-informed prior updated by OD; feed to Layer 5 shared node",
})
# Also include alpha_cat posteriors
alpha_cat_names = ["alpha_cat_0", "alpha_cat_1", "alpha_cat_2", "alpha_cat_3"]
alpha_cat_l5 = pd.DataFrame({
    "parameter"  : alpha_cat_names,
    "prior_mu"   : alpha_post.mean(axis=0).round(5),
    "prior_sigma": alpha_post.std(axis=0).round(5),
    "hdi_3pct"   : np.percentile(alpha_post, 3,  axis=0).round(5),
    "hdi_97pct"  : np.percentile(alpha_post, 97, axis=0).round(5),
    "source"     : "Layer4_OD_NegBin",
    "note"       : "Category fixed effects (log-multiplier vs mall)",
})
layer5_priors = pd.concat([delta_l5, beta_s_l5, alpha_cat_l5], ignore_index=True)
priors_path = os.path.join(OUTPUT_DIR, "layer4_delta_cv_posteriors.csv")
layer5_priors.to_csv(priors_path, index=False)
print(f"  Layer 5 priors    -> {priors_path}")

# 8d. Diagnostics text file
diag_path = os.path.join(OUTPUT_DIR, "layer4_diagnostics.txt")
with open(diag_path, "w") as f:
    f.write("HBHCM Layer 4 Diagnostics\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"n_dest          : {n_dest}\n")
    f.write(f"R-hat > 1.05    : {n_bad_rhat}\n")
    f.write(f"ESS_bulk < 400  : {n_low_ess}\n")
    f.write(f"Pearson r(y,yhat): {corr:.4f}\n")
    f.write(f"MAE             : {mae:.2f}\n")
    f.write(f"RMSE            : {rmse:.2f}\n")
    f.write(f"Bayesian R2     : {bayes_r2:.4f}\n\n")
    f.write("delta_CV posteriors:\n")
    f.write(delta_summary.to_string(index=False))
    f.write("\n\nbeta_S OD update:\n")
    f.write(f"  beta_s_med: SP prior={SP_BETA_S_MED_MU:.3f} -> OD posterior={bs_med_post.mean():.4f}\n")
    f.write(f"  beta_s_lrg: SP prior={SP_BETA_S_LRG_MU:.3f} -> OD posterior={bs_lrg_post.mean():.4f}\n")
print(f"  Diagnostics       -> {diag_path}")

# 8e. Posterior predictive plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].scatter(y, y_hat_mean, alpha=0.5, s=20)
axes[0].plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw=1)
axes[0].set_xlabel("Observed rating_count")
axes[0].set_ylabel("Posterior mean y_hat")
axes[0].set_title(f"Posterior predictive fit  (r={corr:.3f})")

# delta_CV forest plot
y_pos = np.arange(5)
axes[1].barh(y_pos, delta_summary["mean"], xerr=delta_summary["sd"],
             color=["#1D9E75" if v > 0 else "#E24B4A"
                    for v in delta_summary["mean"]], alpha=0.7)
axes[1].axvline(0, color="black", lw=0.8, ls="--")
axes[1].set_yticks(y_pos)
axes[1].set_yticklabels(CV_DIM_NAMES)
axes[1].set_xlabel("delta_CV posterior mean ± 1 SD")
axes[1].set_title("CV perceptual effects on OD flow")

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "layer4_plots.png")
fig.savefig(plot_path, dpi=150)
plt.close()
print(f"  Plots             -> {plot_path}")

# =============================================================================
# SECTION 9 — LAYER 5 HANDOFF SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 9 — Layer 5 Handoff Summary")
print("=" * 70)
print("""
  Parameters to carry into Layer 5 joint model:
  ┌─────────────────────────────────────────────────────────────────┐
  │ Shared (SP + OD both update):                                   │
  │   beta_s_med  ← OD-updated posterior from layer4_delta_cv.csv  │
  │   beta_s_lrg  ← OD-updated posterior from layer4_delta_cv.csv  │
  │                                                                 │
  │ OD-exclusive (carry as informative priors):                     │
  │   delta_cv[0..4] ← layer4_delta_cv_posteriors.csv              │
  │                                                                 │
  │ SP-exclusive (unchanged from iclv_v1_no_wfh_utility.py):       │
  │   beta_gtc    ← SP posterior only; NOT estimated in Layer 4    │
  │   gamma_k (LV -> choice)                                        │
  │   alpha_k (Z_n -> LV structural equations)                      │
  │   LV loadings (measurement layer)                               │
  └─────────────────────────────────────────────────────────────────┘
  Layer 5 joint posterior:
    p(Θ | SP, OD) ∝ L_SP(beta, gamma, LV) × L_OD(beta, delta_CV) × p(prior)
""")

print("=" * 70)
print("COMPLETE — layer4_pymc_v1.py")
print("=" * 70)
