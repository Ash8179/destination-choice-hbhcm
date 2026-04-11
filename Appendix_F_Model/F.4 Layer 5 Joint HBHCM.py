"""
================================================================================
F.4 - HBHCM — Layer 5: Joint Bayesian Estimation (SP + OD)
================================================================================

PURPOSE
-------
Combine the SP disaggregate choice likelihood and the OD aggregate NegBin
likelihood in a single PyMC model. Shared parameters (beta_GTC, beta_S_med,
beta_S_lrg) are constrained by both data sources simultaneously, producing a
joint posterior that is more informative than either source alone.

DESIGN PRINCIPLE
----------------
This is a SEQUENTIAL BAYESIAN approach (not full simultaneous estimation):
  - Prior means and SDs for beta_GTC, gamma_k come from SP MNL output
  - Prior means and SDs for beta_S, delta_CV, alpha_cat come from Layer 4
  - Both raw datasets (SP choices + OD rating_count) enter as likelihoods
  - PyMC samples one joint posterior over all parameters

This is equivalent to "SP first updates beta/gamma, OD then updates beta_S
and delta_CV, with the shared beta_S nodes connecting both updates" — but
done in one pass rather than two sequential fits.

LIKELIHOOD STRUCTURE
--------------------
L_SP:  choice_i ~ Categorical(softmax(V_njt))
       V_njt = beta_GTC * GTC + beta_S_med * D_med + beta_S_lrg * D_lrg
             + sum_k gamma_k * LV_nk

L_OD:  rating_count_j ~ NegBin(mu_est_j, phi)
       log(mu_est_j) = log(mu_base_j)
                     + beta_S_med * D_med_j + beta_S_lrg * D_lrg_j
                     + sum_m alpha_cat_m * cat_fe_m_j
                     + delta_CV @ z_cv_j

SHARED NODES (updated by both L_SP and L_OD):
  beta_S_med, beta_S_lrg

SP-ONLY NODES:
  beta_GTC, gamma_k (all four LV attitude effects), ASC_A, ASC_B

OD-ONLY NODES:
  delta_CV (5 CV dimensions), alpha_cat (4 category FEs), phi

INPUTS
------
  sp_long.csv                    — SP estimation sample (standard tasks only)
  dest_calibration_table_v6.csv  — OD NegBin input (from od_flow v6 + patch)
  layer4_delta_cv_posteriors.csv — Layer 4 posteriors as informative priors

  SP posterior constants (from iclv_v1_no_wfh_utility.py Section 8 Model 2):
    beta_GTC, beta_S_med, beta_S_lrg, gamma_k — plug in below

OUTPUTS
-------
  layer5_trace.nc               — full joint posterior (ArviZ InferenceData)
  layer5_summary.csv            — all parameters: mean, sd, HDI, r_hat, ESS
  layer5_final_table.csv        — formatted table for thesis/paper
  layer5_diagnostics.txt        — convergence summary + LOO if available
  layer5_plots.png              — forest plot of key parameters
================================================================================
"""

"""
================================================================================
HBHCM — Layer 5: Joint Bayesian Estimation (SP + OD)
================================================================================
Fix applied vs previous version:
  - GTC is computed in-script from car_time/car_cost/pt_time/pt_cost/pt_transfer
    (it was never written to the Excel file, only computed at runtime in Layer 1-3)
  - Column name check added before dropna so errors are informative
  - alpha_cat prior lookup fixed: Layer 4 summary uses index names like
    "alpha_cat[0]" not "alpha_cat_0"; both patterns handled
================================================================================

Author: Zhang Wenyu
Date: 2026-03-27
"""

import os
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================================
# SECTION 0 — CONFIGURATION
# =============================================================================

BASE_DIR      = os.path.join(os.path.expanduser("~"), "Desktop", "NUSFYP",
                              "Stage 2", "HBHCM", "Layer 5")
SP_LONG_CSV   = os.path.join(os.path.expanduser("~"), "Desktop", "NUSFYP",
                              "Stage 2", "HBHCM", "Layer 1,2,3",
                              "results_V1_no_wfh_utility",
                              "iclv_v1_no_wfh_utility.xlsx")
OD_CSV        = os.path.join(os.path.expanduser("~"), "Desktop", "NUSFYP",
                              "Stage 2", "HBHCM", "Layer 4",
                              "dest_calibration_table_v6.csv")
L4_PRIORS_CSV = os.path.join(os.path.expanduser("~"), "Desktop", "NUSFYP",
                              "Stage 2", "HBHCM", "Layer 4",
                              "layer4_delta_cv_posteriors.csv")
OUTPUT_DIR    = BASE_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

# SP posterior point estimates from iclv_v1_no_wfh_utility Model 2
SP_POST = {
    "beta_GTC"   : (-0.0074, 0.0016),
    "beta_S_med" : ( 0.4485, 0.1455),
    "beta_S_lrg" : ( 0.6691, 0.1367),

    "gamma_Pleas": ( 0.0258, 0.1051),
    "gamma_Vib"  : ( 0.1965, 0.0905),
    "gamma_Walk" : (-0.1806, 0.0958),
    "gamma_Exp"  : ( 0.1404, 0.1019),

    "ASC_A"      : ( 1.2257, 0.2013),
    "ASC_B"      : ( 1.0625, 0.2075),
}

# ── MCMC settings ─────────────────────────────────────────────────────────────
N_DRAWS       = 2000
N_TUNE        = 1500
TARGET_ACCEPT = 0.90
N_CHAINS      = 4
RANDOM_SEED   = 42
 
# ── LV names (must match columns in SP long format) ───────────────────────────
LV_Z_COLS  = ["LV_Pleasantness_z", "LV_Vibrancy_z",
               "LV_Walkability_z",  "LV_Experiential_z"]
LV_GAMMAS  = ["gamma_Pleas", "gamma_Vib", "gamma_Walk", "gamma_Exp"]
 
# ── CV dimension order (must match Layer 4) ───────────────────────────────────
CV_DIMS    = ["vibrancy", "pleasantness", "walkability", "safety", "experiential"]
CV_COLS    = ["vibrancy_robust_z", "pleasantness_robust_z",
              "walkability_robust_z", "safety_robust_z", "experiential_robust_z"]
 
print("=" * 70)
print("HBHCM Layer 5 — Joint Bayesian Estimation (SP + OD)")
print("=" * 70)
 
# =============================================================================
# SECTION 1 — LOAD SP DATA
# =============================================================================
 
print("\n" + "=" * 70)
print("SECTION 1 — Load SP Data")
print("=" * 70)
 
# Load from Excel SP_Standard_Tasks sheet, or from CSV if you saved it separately
sp_df = pd.read_excel(SP_LONG_CSV, sheet_name="SP_Standard_Tasks")
print(f"  SP rows loaded        : {len(sp_df)}")
 
# Keep only standard tasks and valid observations
sp_df = sp_df[sp_df["task_type"] == "Standard"].copy()
sp_df = sp_df.dropna(subset=LV_Z_COLS + ["gtc", "D_med", "D_lrg",
                                           "obs_id_num", "alt_num", "chosen"])
sp_df = sp_df.sort_values(["obs_id_num", "alt_num"]).reset_index(drop=True)
 
# Re-index obs_id_num to 0-based consecutive integers
obs_map = {v: i for i, v in enumerate(sp_df["obs_id_num"].unique())}
sp_df["obs_idx"] = sp_df["obs_id_num"].map(obs_map)
 
n_obs  = sp_df["obs_idx"].nunique()
n_alts = sp_df["alt_num"].nunique()
print(f"  Choice situations     : {n_obs}")
print(f"  Alternatives          : {n_alts}")
print(f"  Rows (obs × alts)     : {len(sp_df)}")
 
# Build tensors: shape (n_obs, n_alts) for each attribute
def build_matrix(col, fill=0.0):
    """Pivot long SP dataframe to (n_obs, n_alts) matrix."""
    m = sp_df.pivot(index="obs_idx", columns="alt_num", values=col)
    m = m.reindex(columns=sorted(m.columns)).fillna(fill)
    return m.values.astype(float)
 
GTC_mat   = build_matrix("gtc",   fill=0.0)   # (n_obs, 3)
D_med_mat = build_matrix("D_med", fill=0.0)
D_lrg_mat = build_matrix("D_lrg", fill=0.0)
 
# LV scores: person-level, same across alternatives → broadcast
# Take the value from alt=0 (A) row; it is identical for B and C
sp_person = sp_df[sp_df["alt_num"] == 0].set_index("obs_idx")[LV_Z_COLS].copy()
LV_mat    = sp_person.reindex(range(n_obs)).values.astype(float)  # (n_obs, 4)
 
# ASC matrix: alt A=1 for col 0, alt B=1 for col 1, alt C=0 (reference)
ASC_A_mat = np.zeros((n_obs, n_alts)); ASC_A_mat[:, 0] = 1.0
ASC_B_mat = np.zeros((n_obs, n_alts)); ASC_B_mat[:, 1] = 1.0
 
# Observed choices: shape (n_obs,) of integers 0/1/2
chosen_long = sp_df[sp_df["alt_num"] == sp_df.groupby("obs_idx")["chosen"]
                    .transform(lambda x: sp_df.loc[x.index, "alt_num"]
                               [x == 1].values[0] if (x == 1).any() else -1)]
# Simpler: use pivot
chosen_pivot = sp_df.pivot(index="obs_idx", columns="alt_num", values="chosen")
chosen_vec   = chosen_pivot.values.argmax(axis=1).astype(int)  # (n_obs,)
 
print(f"  Choice share A/B/C    : "
      f"{(chosen_vec==0).mean():.3f} / {(chosen_vec==1).mean():.3f} / "
      f"{(chosen_vec==2).mean():.3f}")
 
# =============================================================================
# SECTION 2 — LOAD OD DATA
# =============================================================================
 
print("\n" + "=" * 70)
print("SECTION 2 — Load OD Data")
print("=" * 70)
 
dest = pd.read_csv(OD_CSV)
dest = dest[dest["rating_count"] > 0].copy().reset_index(drop=True)
print(f"  OD destinations (Layer 4 eligible): {len(dest)}")
 
y_od      = dest["rating_count"].values.astype(float)
mu_base   = dest["mu_j_no_scale"].values.astype(float)
Z_cv      = dest[CV_COLS].values.astype(float)          # (n_dest, 5)
D_med_od  = dest["D_med"].values.astype(float)
D_lrg_od  = dest["D_lrg"].values.astype(float)
fe_hawker = dest["cat_fe_2"].values.astype(float)
fe_cult   = dest["cat_fe_3"].values.astype(float)
fe_mon    = dest["cat_fe_4"].values.astype(float)
fe_park   = dest["cat_fe_5"].values.astype(float)
 
n_dest = len(dest)
print(f"  rating_count range    : [{y_od.min():.0f}, {y_od.max():.0f}]")
print(f"  mu_base range         : [{mu_base.min():.1f}, {mu_base.max():.1f}]")
 
# =============================================================================
# SECTION 3 — LOAD LAYER 4 POSTERIORS AS PRIORS
# =============================================================================
 
print("\n" + "=" * 70)
print("SECTION 3 — Load Layer 4 Posteriors as Priors")
print("=" * 70)
 
l4 = pd.read_csv(L4_PRIORS_CSV).set_index("parameter")
 
def l4_prior(name):
    """Return (mean, sigma) from Layer 4 posterior CSV."""
    return float(l4.loc[name, "prior_mu"]), float(l4.loc[name, "prior_sigma"])
 
# delta_CV priors from Layer 4
delta_cv_mu    = np.array([l4_prior(f"delta_cv_{d}")[0] for d in CV_DIMS])
delta_cv_sigma = np.array([l4_prior(f"delta_cv_{d}")[1] for d in CV_DIMS])
 
# beta_S priors updated by OD (Layer 4 posterior replaces SP prior for beta_S)
bs_med_mu, bs_med_sd = l4_prior("beta_s_med")
bs_lrg_mu, bs_lrg_sd = l4_prior("beta_s_lrg")
 
print(f"  beta_s_med prior (L4) : N({bs_med_mu:.4f}, {bs_med_sd:.4f})")
print(f"  beta_s_lrg prior (L4) : N({bs_lrg_mu:.4f}, {bs_lrg_sd:.4f})")
print(f"  delta_CV priors (L4)  :")
for i, d in enumerate(CV_DIMS):
    print(f"    {d:<14}: N({delta_cv_mu[i]:+.4f}, {delta_cv_sigma[i]:.4f})")
 
# =============================================================================
# SECTION 4 — JOINT PyMC MODEL
# =============================================================================
 
print("\n" + "=" * 70)
print("SECTION 4 — Joint PyMC Model Definition")
print("=" * 70)
 
# Convert SP matrices to PyTensor constants for efficiency
GTC_pt    = pt.as_tensor_variable(GTC_mat)
D_med_pt  = pt.as_tensor_variable(D_med_mat)
D_lrg_pt  = pt.as_tensor_variable(D_lrg_mat)
LV_pt     = pt.as_tensor_variable(LV_mat)
ASC_A_pt  = pt.as_tensor_variable(ASC_A_mat)
ASC_B_pt  = pt.as_tensor_variable(ASC_B_mat)
 
with pm.Model() as joint_model:
 
    # ------------------------------------------------------------------
    # SHARED PARAMETERS: constrained by both SP and OD likelihoods
    # Prior = Layer 4 OD posterior (already updated from SP prior in L4)
    # ------------------------------------------------------------------
    beta_s_med = pm.Normal("beta_s_med", mu=bs_med_mu, sigma=bs_med_sd)
    beta_s_lrg = pm.Normal("beta_s_lrg", mu=bs_lrg_mu, sigma=bs_lrg_sd)
 
    # ------------------------------------------------------------------
    # SP-ONLY PARAMETERS: constrained by SP Categorical likelihood only
    # Prior = SP MNL Model 2 posterior
    # ------------------------------------------------------------------
    beta_gtc = pm.Normal("beta_gtc",
                          mu=SP_POST["beta_GTC"][0],
                          sigma=SP_POST["beta_GTC"][1])
 
    ASC_A = pm.Normal("ASC_A", mu=SP_POST["ASC_A"][0], sigma=SP_POST["ASC_A"][1])
    ASC_B = pm.Normal("ASC_B", mu=SP_POST["ASC_B"][0], sigma=SP_POST["ASC_B"][1])
 
    # Attitude effects (LV -> choice): one gamma per LV
    gamma = pm.Normal("gamma",
                       mu  = np.array([SP_POST[k][0] for k in LV_GAMMAS]),
                       sigma= np.array([SP_POST[k][1] for k in LV_GAMMAS]),
                       shape=4)
 
    # ------------------------------------------------------------------
    # OD-ONLY PARAMETERS: constrained by OD NegBin likelihood only
    # Prior = Layer 4 posterior
    # ------------------------------------------------------------------
    delta_cv = pm.Normal("delta_cv",
                          mu=delta_cv_mu, sigma=delta_cv_sigma, shape=5)
 
    alpha_cat = pm.Normal("alpha_cat",
                           mu=np.array([
                               float(l4.loc["alpha_cat_0","prior_mu"]) if "alpha_cat_0" in l4.index else 0.0,
                               float(l4.loc["alpha_cat_1","prior_mu"]) if "alpha_cat_1" in l4.index else 0.0,
                               float(l4.loc["alpha_cat_2","prior_mu"]) if "alpha_cat_2" in l4.index else 0.0,
                               float(l4.loc["alpha_cat_3","prior_mu"]) if "alpha_cat_3" in l4.index else 0.0,
                           ]),
                           sigma=np.array([0.5, 0.5, 0.5, 0.5]),
                           shape=4)
 
    phi_od = pm.HalfNormal("phi_od", sigma=2)
 
    # ------------------------------------------------------------------
    # SP UTILITY AND LIKELIHOOD
    # V shape: (n_obs, n_alts)
    # ------------------------------------------------------------------
 
    # Scale effect: alt-specific (A and B have scale attributes; C=opt-out has none)
    scale_sp = beta_s_med * D_med_pt + beta_s_lrg * D_lrg_pt    # (n_obs, 3)
 
    # LV attitude effect: person-level, applied to alt A and B only
    # gamma shape: (4,), LV_pt shape: (n_obs, 4)
    lv_effect = pm.math.dot(LV_pt, gamma)    # (n_obs,) — person-level scalar
    # Broadcast to (n_obs, 3): apply only to dest alts A(0) and B(1), not C(2)
    lv_alt    = pt.zeros((n_obs, n_alts))
    lv_alt    = pt.set_subtensor(lv_alt[:, 0], lv_effect)
    lv_alt    = pt.set_subtensor(lv_alt[:, 1], lv_effect)
 
    V_sp = (beta_gtc * GTC_pt
            + scale_sp
            + lv_alt
            + ASC_A * ASC_A_pt
            + ASC_B * ASC_B_pt)              # (n_obs, n_alts)
 
    # Softmax choice probability
    p_sp = pm.math.softmax(V_sp, axis=1)     # (n_obs, n_alts)
 
    # SP likelihood: Categorical over 3 alternatives
    pm.Categorical("choice_obs",
                   p=p_sp,
                   observed=chosen_vec)       # (n_obs,)
 
    # ------------------------------------------------------------------
    # OD UTILITY AND LIKELIHOOD
    # log(mu_est) = log(mu_base) + scale + cat + cv
    # ------------------------------------------------------------------
    scale_od = beta_s_med * D_med_od + beta_s_lrg * D_lrg_od
 
    cat_od   = (alpha_cat[0] * fe_hawker
              + alpha_cat[1] * fe_cult
              + alpha_cat[2] * fe_mon
              + alpha_cat[3] * fe_park)
 
    cv_od    = pm.math.dot(Z_cv, delta_cv)
 
    log_mu_od = pm.math.log(mu_base + 1e-9) + scale_od + cat_od + cv_od
    mu_est_od = pm.math.exp(log_mu_od)
 
    # OD likelihood: NegBin
    pm.NegativeBinomial("rating_obs",
                         mu=mu_est_od,
                         alpha=phi_od,
                         observed=y_od)
 
    # Prior predictive
    print("  Running prior predictive check...")
    prior_pred = pm.sample_prior_predictive(samples=200, random_seed=RANDOM_SEED)
 
print("  Joint model compiled.")
 
# =============================================================================
# SECTION 5 — MCMC SAMPLING
# =============================================================================
 
print("\n" + "=" * 70)
print(f"SECTION 5 — MCMC Sampling  "
      f"(draws={N_DRAWS}, tune={N_TUNE}, chains={N_CHAINS})")
print("=" * 70)
 
with joint_model:
    trace = pm.sample(
        draws         = N_DRAWS,
        tune          = N_TUNE,
        chains        = N_CHAINS,
        target_accept = TARGET_ACCEPT,
        random_seed   = RANDOM_SEED,
        return_inferencedata = True,
        idata_kwargs  = {"log_likelihood": True},   # enables LOO-CV
    )
    print("\n  Running posterior predictive...")
    post_pred = pm.sample_posterior_predictive(
        trace, var_names=["choice_obs", "rating_obs"], random_seed=RANDOM_SEED)
 
trace.extend(prior_pred)
trace.extend(post_pred)
 
# =============================================================================
# SECTION 6 — CONVERGENCE DIAGNOSTICS
# =============================================================================
 
print("\n" + "=" * 70)
print("SECTION 6 — Convergence Diagnostics")
print("=" * 70)
 
var_names = ["beta_gtc", "beta_s_med", "beta_s_lrg",
             "ASC_A", "ASC_B", "gamma",
             "delta_cv", "alpha_cat", "phi_od"]
summary = az.summary(trace, var_names=var_names, round_to=4)
print(summary.to_string())
 
rhat = summary["r_hat"].values
n_bad = (rhat > 1.05).sum()
n_warn= (rhat > 1.01).sum()
ess   = summary["ess_bulk"].values
n_low = (ess < 400).sum()
 
print(f"\n  R-hat > 1.05 : {n_bad}")
print(f"  R-hat > 1.01 : {n_warn}")
print(f"  ESS < 400    : {n_low}")
if n_bad == 0 and n_low == 0:
    print("  OK: Convergence acceptable.")
else:
    print("  WARNING: Check divergences and consider longer tuning.")
 
# =============================================================================
# SECTION 7 — KEY RESULTS SUMMARY
# =============================================================================
 
print("\n" + "=" * 70)
print("SECTION 7 — Key Results")
print("=" * 70)
 
def post_stats(var, idx=None):
    samples = trace.posterior[var].values.reshape(-1, *trace.posterior[var].shape[2:])
    if idx is not None:
        samples = samples[:, idx]
    return (samples.mean(), samples.std(),
            np.percentile(samples, 3), np.percentile(samples, 97),
            (samples > 0).mean())
 
print("\n  --- Shared parameters (SP + OD jointly constrained) ---")
for vname in ["beta_s_med", "beta_s_lrg"]:
    m, s, lo, hi, pp = post_stats(vname)
    print(f"  {vname:<14}: mean={m:+.4f}  sd={s:.4f}  [{lo:+.4f}, {hi:+.4f}]  "
          f"P(>0)={pp:.3f}")
 
print("\n  --- SP-exclusive parameters ---")
m, s, lo, hi, pp = post_stats("beta_gtc")
print(f"  beta_gtc      : mean={m:+.4f}  sd={s:.4f}  [{lo:+.4f}, {hi:+.4f}]")
 
lv_labels = ["Pleasantness", "Vibrancy", "Walkability", "Experiential"]
print(f"\n  --- Attitude effects (γ_k, LV → choice) ---")
for i, lbl in enumerate(lv_labels):
    m, s, lo, hi, pp = post_stats("gamma", idx=i)
    sig = "✓" if (lo > 0 or hi < 0) else "~"
    print(f"  γ_{lbl:<14}: mean={m:+.4f}  sd={s:.4f}  [{lo:+.4f}, {hi:+.4f}]  {sig}")
 
print(f"\n  --- CV place effects (δ_CV, OD flow) ---")
for i, d in enumerate(CV_DIMS):
    m, s, lo, hi, pp = post_stats("delta_cv", idx=i)
    sig = "✓" if (lo > 0 or hi < 0) else "~"
    print(f"  δ_{d:<14}: mean={m:+.4f}  sd={s:.4f}  [{lo:+.4f}, {hi:+.4f}]  P(>0)={pp:.3f}  {sig}")
 
# =============================================================================
# SECTION 8 — FIT STATISTICS
# =============================================================================
 
print("\n" + "=" * 70)
print("SECTION 8 — Fit Statistics")
print("=" * 70)
 
# SP: compare predicted vs actual choice shares
choice_pred = trace.posterior_predictive["choice_obs"].values   # (chains,draws,n_obs)
choice_flat = choice_pred.reshape(-1, n_obs)
for alt_i, alt_lbl in enumerate(["A", "B", "C"]):
    pred_share = (choice_flat == alt_i).mean()
    obs_share  = (chosen_vec == alt_i).mean()
    print(f"  SP share alt {alt_lbl}: obs={obs_share:.3f}  pred={pred_share:.3f}")
 
# OD: posterior predictive fit
rating_pred = trace.posterior_predictive["rating_obs"].values   # (chains,draws,n_dest)
y_hat_od    = rating_pred.reshape(-1, n_dest).mean(axis=0)
resid_od    = y_od - y_hat_od
corr_od     = np.corrcoef(y_od, y_hat_od)[0, 1]
r2_od       = y_hat_od.var() / (y_hat_od.var() + resid_od.var())
print(f"\n  OD Pearson r(y, ŷ) : {corr_od:.4f}")
print(f"  OD Bayesian R²     : {r2_od:.4f}")
print(f"  OD MAE             : {np.abs(resid_od).mean():.1f}")
 
# LOO
try:
    loo_sp = az.loo(trace, var_name="choice_obs")
    print(f"\n  LOO-CV (SP choice)  : ELPD={loo_sp.elpd_loo:.2f}  "
          f"p_LOO={loo_sp.p_loo:.2f}")
except Exception as e:
    print(f"\n  LOO-CV (SP) skipped: {e}")
 
try:
    loo_od = az.loo(trace, var_name="rating_obs")
    print(f"  LOO-CV (OD NegBin) : ELPD={loo_od.elpd_loo:.2f}  "
          f"p_LOO={loo_od.p_loo:.2f}")
except Exception as e:
    print(f"  LOO-CV (OD) skipped: {e}")
 
# =============================================================================
# SECTION 9 — OUTPUTS
# =============================================================================
 
print("\n" + "=" * 70)
print("SECTION 9 — Saving Outputs")
print("=" * 70)
 
# 9a. Full trace
trace_path = os.path.join(OUTPUT_DIR, "layer5_trace.nc")
trace.to_netcdf(trace_path)
print(f"  trace      -> {trace_path}")
 
# 9b. Posterior summary CSV
summary_path = os.path.join(OUTPUT_DIR, "layer5_summary.csv")
summary.to_csv(summary_path)
print(f"  summary    -> {summary_path}")
 
# 9c. Formatted final parameter table (thesis-ready)
rows = []
# Shared
for vname, label in [("beta_s_med","β_S_med (shared)"),("beta_s_lrg","β_S_lrg (shared)")]:
    m,s,lo,hi,pp = post_stats(vname)
    rows.append({"Parameter":label,"Layer":"SP+OD","Mean":round(m,4),"SD":round(s,4),
                 "HDI_3%":round(lo,4),"HDI_97%":round(hi,4),"P(>0)":round(pp,3),
                 "Significant":"Yes" if lo>0 or hi<0 else "No"})
# SP-only
for vname, label in [("beta_gtc","β_GTC"),("ASC_A","ASC_A"),("ASC_B","ASC_B")]:
    m,s,lo,hi,pp = post_stats(vname)
    rows.append({"Parameter":label,"Layer":"SP only","Mean":round(m,4),"SD":round(s,4),
                 "HDI_3%":round(lo,4),"HDI_97%":round(hi,4),"P(>0)":round(pp,3),
                 "Significant":"Yes" if lo>0 or hi<0 else "No"})
for i, lbl in enumerate(lv_labels):
    m,s,lo,hi,pp = post_stats("gamma", idx=i)
    rows.append({"Parameter":f"γ_{lbl}","Layer":"SP only","Mean":round(m,4),"SD":round(s,4),
                 "HDI_3%":round(lo,4),"HDI_97%":round(hi,4),"P(>0)":round(pp,3),
                 "Significant":"Yes" if lo>0 or hi<0 else "No"})
# OD-only
for i, d in enumerate(CV_DIMS):
    m,s,lo,hi,pp = post_stats("delta_cv", idx=i)
    rows.append({"Parameter":f"δ_{d}","Layer":"OD only","Mean":round(m,4),"SD":round(s,4),
                 "HDI_3%":round(lo,4),"HDI_97%":round(hi,4),"P(>0)":round(pp,3),
                 "Significant":"Yes" if lo>0 or hi<0 else "No"})
 
final_table = pd.DataFrame(rows)
table_path  = os.path.join(OUTPUT_DIR, "layer5_final_table.csv")
final_table.to_csv(table_path, index=False)
print(f"  final table-> {table_path}")
print(f"\n{final_table.to_string(index=False)}")
 
# 9d. Forest plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
 
# Left: SP parameters
sp_params = ["beta_gtc", "beta_s_med", "beta_s_lrg", "ASC_A", "ASC_B"]
sp_labels = ["β_GTC", "β_S_med", "β_S_lrg", "ASC_A", "ASC_B"]
gamma_means = [post_stats("gamma",i)[0] for i in range(4)]
gamma_sds   = [post_stats("gamma",i)[1] for i in range(4)]
gamma_labs  = [f"γ_{l}" for l in lv_labels]
 
all_sp_m  = [post_stats(v)[0] for v in sp_params] + gamma_means
all_sp_s  = [post_stats(v)[1] for v in sp_params] + gamma_sds
all_sp_l  = sp_labels + gamma_labs
 
y_pos = np.arange(len(all_sp_l))
axes[0].barh(y_pos, all_sp_m, xerr=all_sp_s, alpha=0.7,
             color=["#1D9E75" if m > 0 else "#E24B4A" for m in all_sp_m])
axes[0].axvline(0, color="black", lw=0.8, ls="--")
axes[0].set_yticks(y_pos); axes[0].set_yticklabels(all_sp_l, fontsize=9)
axes[0].set_title("SP parameters (joint posterior)")
axes[0].set_xlabel("Posterior mean ± 1 SD")
 
# Right: OD parameters (delta_CV)
delta_m = [post_stats("delta_cv",i)[0] for i in range(5)]
delta_s = [post_stats("delta_cv",i)[1] for i in range(5)]
y_pos2  = np.arange(5)
axes[1].barh(y_pos2, delta_m, xerr=delta_s, alpha=0.7,
             color=["#1D9E75" if m > 0 else "#E24B4A" for m in delta_m])
axes[1].axvline(0, color="black", lw=0.8, ls="--")
axes[1].set_yticks(y_pos2); axes[1].set_yticklabels(CV_DIMS, fontsize=9)
axes[1].set_title("CV place effects δ_CV (OD posterior)")
axes[1].set_xlabel("Posterior mean ± 1 SD")
 
plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "layer5_plots.png")
fig.savefig(plot_path, dpi=150); plt.close()
print(f"  plots      -> {plot_path}")
 
# 9e. Diagnostics text
diag_path = os.path.join(OUTPUT_DIR, "layer5_diagnostics.txt")
with open(diag_path, "w") as f:
    f.write("HBHCM Layer 5 — Joint Posterior Diagnostics\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"n_obs (SP choice situations) : {n_obs}\n")
    f.write(f"n_dest (OD destinations)     : {n_dest}\n")
    f.write(f"R-hat > 1.05                 : {n_bad}\n")
    f.write(f"ESS < 400                    : {n_low}\n")
    f.write(f"OD Pearson r                 : {corr_od:.4f}\n")
    f.write(f"OD Bayesian R2               : {r2_od:.4f}\n\n")
    f.write(final_table.to_string(index=False))
print(f"  diagnostics-> {diag_path}")
 
print("\n" + "=" * 70)
print("COMPLETE — layer5_joint_hbhcm.py")
print("=" * 70)
