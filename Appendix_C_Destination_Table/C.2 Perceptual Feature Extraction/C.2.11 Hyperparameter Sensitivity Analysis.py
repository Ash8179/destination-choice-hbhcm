"""
C.2.11/G.1 - Hyperparameter Sensitivity Analysis
=====================================
Tests the robustness of the HBHCM pipeline to three aggregation hyperparameters:

  λ  (temporal decay)  — controlled via HALF_LIFE ∈ {270, 365, 730} days
  κ  (shrinkage strength) ∈ {2, 5, 15}
  α  (prior mixing weight) ∈ {0.0, 0.5, 1.0}

Strategy: One-At-A-Time (OAT)
  Baseline: HALF_LIFE=365, KAPPA=4, ALPHA=0.6
  For each parameter, vary it across its grid while holding the other
  two at baseline. Total runs = 1 (baseline) + 2 + 2 + 2 = 7 runs.

Execution order per run:
  1. Layer 4 / 1.Street-Level Perception Aggregation Pipeline.py
  2. Layer 4 / 2.od_flow_computation_v6 no Beta_GTC.py
  3. Layer 4 / 3.Layer 4 PyMC.py
  4. Layer 5 / 4.Layer5 Joint HBHCM.py

Output:
  sensitivity_results/
    ├── run_<tag>/          raw outputs from each run
    └── sensitivity_table.csv   consolidated parameter table
    └── sensitivity_report.txt  human-readable summary
    └── sensitivity_plot.png    forest-plot comparison

Author: Zhang Wenyu
Date: 2026-04-01
"""

import os
import sys
import re
import shutil
import subprocess
import tempfile
import time
import textwrap
from itertools import product
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ══════════════════════════════════════════════════════════════════════════════
# 0.  CONFIGURATION  — edit these paths if needed
# ══════════════════════════════════════════════════════════════════════════════

BASE = Path("/Users/zhangwenyu/Desktop/NUSFYP/Stage 2/HBHCM")

SCRIPT_1 = BASE / "Layer 4" / "1.Street-Level Perception Aggregation Pipeline.py"
SCRIPT_2 = BASE / "Layer 4" / "2.od_flow_computation_v6 no Beta_GTC.py"
SCRIPT_3 = BASE / "Layer 4" / "3.Layer 4 PyMC.py"
SCRIPT_4 = BASE / "Layer 5" / "4.Layer5 Joint HBHCM.py"

OUTPUT_ROOT = BASE / "sensitivity_results"

# Python interpreter — adjust if your conda/venv differs
PYTHON = sys.executable   # or e.g. "/opt/homebrew/Caches/pypoetry/virtualenvs/.../bin/python"

# ── Parameter grids ──────────────────────────────────────────────────────────

BASELINE = dict(half_life=365, kappa=4, alpha=0.6)

GRID = {
    "half_life": [270, 365, 730],   # λ = ln(2)/half_life
    "kappa":     [2,   4,  15],
    "alpha":     [0.0, 0.6, 1.0],
}

# OAT: for each parameter, vary it; hold others at baseline
# This produces 7 runs (1 baseline + 2 per parameter, baseline deduped)
def build_oa_table(baseline, grid):
    runs = {}
    # Baseline first
    tag = f"HL{baseline['half_life']}_K{baseline['kappa']}_A{str(baseline['alpha']).replace('.','')}"
    runs[tag] = dict(baseline)
    runs[tag]["label"] = "Baseline"
    runs[tag]["varied_param"] = "baseline"

    for param, values in grid.items():
        for val in values:
            if val == baseline[param]:
                continue   # skip — already in baseline
            cfg = dict(baseline)
            cfg[param] = val
            tag = f"HL{cfg['half_life']}_K{cfg['kappa']}_A{str(cfg['alpha']).replace('.','')}"
            if tag not in runs:
                runs[tag] = cfg
                runs[tag]["label"] = f"{param}={val}"
                runs[tag]["varied_param"] = param
    return runs

RUN_TABLE = build_oa_table(BASELINE, GRID)

# ── Focal output parameters (must match CSV column names in layer5_final_table.csv) ──

FOCAL_PARAMS = [
    "δ_vibrancy",
    "δ_pleasantness",
    "δ_walkability",
    "δ_safety",
    "δ_experiential",
    "β_S_med (shared)",
    "β_S_lrg (shared)",
    "β_GTC",
]

# ══════════════════════════════════════════════════════════════════════════════
# 1.  UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def patch_script_1(half_life: int, kappa: float, alpha: float,
                   src: Path, dst: Path) -> None:
    """
    Read Script 1, replace the four hyperparameter lines, write to dst.

    Target block (lines ~80-84):
        HALF_LIFE = 365
        LAMBDA = np.log(2) / HALF_LIFE
        ALPHA = 0.6
        KAPPA = 4
        REFERENCE_DATE = date.today()
    """
    text = src.read_text(encoding="utf-8")

    text = re.sub(r"^HALF_LIFE\s*=\s*\d+",
                  f"HALF_LIFE = {half_life}", text, flags=re.MULTILINE)
    text = re.sub(r"^LAMBDA\s*=\s*np\.log\(2\)\s*/\s*HALF_LIFE",
                  "LAMBDA = np.log(2) / HALF_LIFE", text, flags=re.MULTILINE)
    text = re.sub(r"^ALPHA\s*=\s*[\d.]+",
                  f"ALPHA = {alpha}", text, flags=re.MULTILINE)
    text = re.sub(r"^KAPPA\s*=\s*[\d.]+",
                  f"KAPPA = {kappa}", text, flags=re.MULTILINE)

    dst.write_text(text, encoding="utf-8")

def run_script(script_path: Path, env: dict = None,
               timeout: int = 600) -> tuple[int, str, str]:
    """Run a Python script, capture stdout/stderr, return (returncode, out, err)."""
    result = subprocess.run(
        [PYTHON, str(script_path)],
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, **(env or {})}
    )
    return result.returncode, result.stdout, result.stderr


def load_final_table(run_dir: Path) -> pd.DataFrame | None:
    """Load layer5_final_table.csv from a run output directory."""
    csv = run_dir / "layer5_final_table.csv"
    if not csv.exists():
        # also try one level up
        for f in run_dir.rglob("layer5_final_table.csv"):
            csv = f
            break
    if csv.exists():
        return pd.read_csv(csv)
    return None


def load_diagnostics(run_dir: Path) -> dict:
    """Load layer5_diagnostics.txt, parse key metrics."""
    txt = run_dir / "layer5_diagnostics.txt"
    if not txt.exists():
        for f in run_dir.rglob("layer5_diagnostics.txt"):
            txt = f
            break
    out = {"r2_od": None, "pearson_r": None, "rhat_bad": None, "ess_low": None}
    if txt.exists():
        content = txt.read_text(encoding="utf-8")
        for line in content.splitlines():
            if "OD Pearson r" in line:
                try: out["pearson_r"] = float(line.split(":")[-1].strip())
                except: pass
            if "OD Bayesian R2" in line or "OD Bayesian R²" in line:
                try: out["r2_od"] = float(line.split(":")[-1].strip())
                except: pass
            if "R-hat > 1.05" in line:
                try: out["rhat_bad"] = int(line.split(":")[-1].strip())
                except: pass
            if "ESS < 400" in line:
                try: out["ess_low"] = int(line.split(":")[-1].strip())
                except: pass
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 2.  MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUT_ROOT / "sensitivity_run_log.txt"
    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(msg)

    log("=" * 72)
    log(f"  SENSITIVITY ANALYSIS — {len(RUN_TABLE)} runs")
    log(f"  Strategy : One-At-A-Time (OAT)")
    log(f"  Baseline : HALF_LIFE={BASELINE['half_life']}  "
        f"KAPPA={BASELINE['kappa']}  ALPHA={BASELINE['alpha']}")
    log("=" * 72)

    all_results = []  # list of dicts, one per run

    for run_idx, (tag, cfg) in enumerate(RUN_TABLE.items()):
        half_life = cfg["half_life"]
        kappa     = cfg["kappa"]
        alpha     = cfg["alpha"]
        label     = cfg["label"]
        varied    = cfg["varied_param"]

        log(f"\n{'─'*72}")
        log(f"  RUN {run_idx+1}/{len(RUN_TABLE)}  [{tag}]  ({label})")
        log(f"  HALF_LIFE={half_life}  KAPPA={kappa}  ALPHA={alpha}")
        log(f"{'─'*72}")

        run_dir = OUTPUT_ROOT / f"run_{tag}"
        run_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir = run_dir / "_patched_scripts"
        tmp_dir.mkdir(exist_ok=True)

        t_start = time.time()
        run_ok = True

        # ── Step 1: patch and run Script 1 ────────────────────────────────
        log("  [1/4] Aggregation pipeline ...")
        s1_patched = tmp_dir / "script1_patched.py"
        patch_script_1(half_life, kappa, alpha, SCRIPT_1, s1_patched)
        env = {
            "OUTPUT_DIR": str(run_dir),

            "IMG_FILE": str(BASE / "Layer 4" / "Perceptual Features.csv"),
            "POI_FILE": str(BASE / "Layer 4" / "D.xlsx"),
        }

        rc, out, err = run_script(s1_patched, env=env)
        (run_dir / "script1_stdout.txt").write_text(out)
        (run_dir / "script1_stderr.txt").write_text(err)
        if rc != 0:
            log(f"  ✗ Script 1 FAILED (rc={rc}). See script1_stderr.txt")
            run_ok = False

        # ── Step 2: OD flow ───────────────────────────────────────────────
        if run_ok:
            log("  [2/4] OD flow computation ...")
            s2_patched = tmp_dir / "script2_patched.py"
            shutil.copy(SCRIPT_2, s2_patched)

            env = {
                "RUN_INPUT_DIR": str(run_dir),
                "OUTPUT_DIR": str(run_dir),
                "STATIC_INPUT_DIR": str(BASE / "Layer 4" / "Input"),
            }

            rc, out, err = run_script(s2_patched, env=env)
            (run_dir / "script2_stdout.txt").write_text(out)
            (run_dir / "script2_stderr.txt").write_text(err)
            if rc != 0:
                log(f"  ✗ Script 2 FAILED (rc={rc}). See script2_stderr.txt")
                run_ok = False

        # ── Step 3: Layer 4 PyMC ─────────────────────────────────────────
        if run_ok:
            log("  [3/4] Layer 4 PyMC ...")
            s3_patched = tmp_dir / "script3_patched.py"

            shutil.copy(SCRIPT_3, s3_patched)

            env = {
                "RUN_DIR": str(run_dir),
                "OUTPUT_DIR": str(run_dir),
                "INPUT_CSV": str(run_dir / "dest_calibration_table_v6.csv"),
            }

            rc, out, err = run_script(s3_patched, env=env)
            (run_dir / "script3_stdout.txt").write_text(out)
            (run_dir / "script3_stderr.txt").write_text(err)
            if rc != 0:
                log(f"  ✗ Script 3 FAILED (rc={rc}). See script3_stderr.txt")
                run_ok = False

        # ── Step 4: Layer 5 Joint HBHCM ──────────────────────────────────
        if run_ok:
            log("  [4/4] Layer 5 Joint HBHCM ...")
            s4_patched = tmp_dir / "script4_patched.py"
            
            shutil.copy(SCRIPT_4, s4_patched)
            
            env = {
                "BASE_DIR": str(run_dir),
                "OUTPUT_DIR": str(run_dir),

                "OD_CSV": str(run_dir / "dest_calibration_table_v6.csv"),
                "L4_PRIORS_CSV": str(run_dir / "layer4_delta_cv_posteriors.csv"),

                "SP_LONG_CSV": str(BASE / "Layer 1,2,3" / "results_V1_no_wfh_utility" / "iclv_v1_no_wfh_utility.xlsx"),
            }

            rc, out, err = run_script(s4_patched, env=env)
            (run_dir / "script4_stdout.txt").write_text(out)
            (run_dir / "script4_stderr.txt").write_text(err)
            if rc != 0:
                log(f"  ✗ Script 4 FAILED (rc={rc}). See script4_stderr.txt")
                run_ok = False

        elapsed = time.time() - t_start
        log(f"  Wall time : {elapsed/60:.1f} min  |  Status: {'✓ OK' if run_ok else '✗ FAILED'}")

        # ── Collect results ───────────────────────────────────────────────
        row = {
            "tag":          tag,
            "label":        label,
            "varied_param": varied,
            "half_life":    half_life,
            "lambda":       round(np.log(2) / half_life, 6),
            "kappa":        kappa,
            "alpha":        alpha,
            "status":       "ok" if run_ok else "failed",
            "elapsed_min":  round(elapsed / 60, 2),
        }

        if run_ok:
            df_params = load_final_table(run_dir)
            diag = load_diagnostics(run_dir)

            row.update({
                "r2_od":      diag["r2_od"],
                "pearson_r":  diag["pearson_r"],
                "rhat_bad":   diag["rhat_bad"],
                "ess_low":    diag["ess_low"],
            })

            if df_params is not None:
                for _, prow in df_params.iterrows():
                    pname = str(prow["Parameter"])
                    if pname in FOCAL_PARAMS:
                        safe = pname.replace(" ", "_").replace("(", "").replace(")", "")
                        row[f"{safe}_mean"]    = prow.get("Mean",    np.nan)
                        row[f"{safe}_sd"]      = prow.get("SD",      np.nan)
                        row[f"{safe}_hdi3"]    = prow.get("HDI_3%",  np.nan)
                        row[f"{safe}_hdi97"]   = prow.get("HDI_97%", np.nan)
                        row[f"{safe}_sig"]     = prow.get("Significant", "?")
            else:
                log(f"  ⚠ layer5_final_table.csv not found in {run_dir}")

        all_results.append(row)

    # ══════════════════════════════════════════════════════════════════════
    # 3.  CONSOLIDATE AND WRITE OUTPUTS
    # ══════════════════════════════════════════════════════════════════════

    df_all = pd.DataFrame(all_results)
    csv_path = OUTPUT_ROOT / "sensitivity_table.csv"
    df_all.to_csv(csv_path, index=False)
    log(f"\n  Consolidated table saved to: {csv_path}")

    # ── Human-readable report ─────────────────────────────────────────────
    report_lines = []
    report_lines.append("=" * 72)
    report_lines.append("  HYPERPARAMETER SENSITIVITY ANALYSIS — RESULTS REPORT")
    report_lines.append(f"  Generated: {date.today()}")
    report_lines.append("=" * 72)

    report_lines.append("""
STRATEGY: One-At-A-Time (OAT)
  Baseline:  HALF_LIFE=365  KAPPA=4  ALPHA=0.6
  λ grid:    HALF_LIFE ∈ {270, 365, 730}  →  λ ∈ {0.00257, 0.00190, 0.00095}
  κ grid:    {2, 4, 15}
  α grid:    {0.0, 0.5, 1.0}

INTERPRETATION GUIDE:
  A parameter is "robust" if:
    (a) The significance pattern (Significant=Yes/No) for the three focal
        CV parameters (δ_vibrancy, δ_pleasantness, δ_walkability) does
        not change across the grid, AND
    (b) The posterior mean shifts by less than ~1 SD of the baseline estimate.
  A change in δ_safety or δ_experiential (both non-significant at baseline)
  is expected and does not threaten the core conclusions.
""")

    report_lines.append("─" * 72)
    report_lines.append("  TABLE 1 — PARAMETER MEANS ACROSS ALL RUNS")
    report_lines.append("─" * 72)

    # Build a clean pivot: rows = runs, cols = focal param means
    focal_mean_cols = [
        f"{p.replace(' ','_').replace('(','').replace(')','')}_mean"
        for p in FOCAL_PARAMS
    ]
    display_cols = ["label", "half_life", "kappa", "alpha",
                    "r2_od", "pearson_r"] + focal_mean_cols

    available = [c for c in display_cols if c in df_all.columns]
    sub = df_all[["tag"] + available].copy()

    # Rename focal mean cols for display
    rename = {}
    for p in FOCAL_PARAMS:
        safe = p.replace(" ", "_").replace("(", "").replace(")", "")
        rename[f"{safe}_mean"] = p
    sub = sub.rename(columns=rename)

    report_lines.append(sub.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    report_lines.append("\n" + "─" * 72)
    report_lines.append("  TABLE 2 — SIGNIFICANCE FLAGS (Yes/No) FOR FOCAL CV PARAMETERS")
    report_lines.append("─" * 72)

    sig_params = ["δ_vibrancy", "δ_pleasantness", "δ_walkability",
                  "δ_safety", "δ_experiential"]
    sig_cols = [f"{p.replace(' ','_').replace('(','').replace(')','')}_sig"
                for p in sig_params]
    sig_avail = [c for c in sig_cols if c in df_all.columns]
    sub_sig = df_all[["label", "half_life", "kappa", "alpha"] + sig_avail].copy()
    rename_sig = {f"{p.replace(' ','_').replace('(','').replace(')','')}_sig": p
                  for p in sig_params}
    sub_sig = sub_sig.rename(columns=rename_sig)
    report_lines.append(sub_sig.to_string(index=False))

    report_lines.append("\n" + "─" * 72)
    report_lines.append("  TABLE 3 — FIT STATISTICS")
    report_lines.append("─" * 72)
    fit_cols = ["label", "half_life", "kappa", "alpha",
                "r2_od", "pearson_r", "rhat_bad", "ess_low", "elapsed_min"]
    fit_avail = [c for c in fit_cols if c in df_all.columns]
    report_lines.append(df_all[fit_avail].to_string(index=False,
                         float_format=lambda x: f"{x:.4f}"))

    report_lines.append("\n" + "─" * 72)
    report_lines.append("  ROBUSTNESS SUMMARY")
    report_lines.append("─" * 72)

    # Auto-compute robustness for focal CV params
    baseline_row = df_all[df_all["varied_param"] == "baseline"]
    if not baseline_row.empty:
        for p in ["δ_vibrancy", "δ_pleasantness", "δ_walkability"]:
            safe = p.replace(" ", "_").replace("(", "").replace(")", "")
            mean_col = f"{safe}_mean"
            sd_col   = f"{safe}_sd"
            sig_col  = f"{safe}_sig"
            if mean_col not in df_all.columns:
                continue
            b_mean = baseline_row[mean_col].values[0]
            b_sd   = baseline_row[sd_col].values[0] if sd_col in df_all.columns else np.nan
            b_sig  = baseline_row[sig_col].values[0] if sig_col in df_all.columns else "?"

            all_means = df_all[mean_col].dropna()
            all_sigs  = df_all[sig_col].dropna() if sig_col in df_all.columns else pd.Series([])

            max_shift = (all_means - b_mean).abs().max()
            sig_stable = (all_sigs == b_sig).all() if len(all_sigs) > 0 else False

            robust = "✓ ROBUST" if (sig_stable and (max_shift < b_sd)) else "⚠ CHECK"
            report_lines.append(
                f"  {p:<20} baseline={b_mean:+.4f}  max_shift={max_shift:.4f}  "
                f"(1SD={b_sd:.4f})  sig_stable={sig_stable}  → {robust}"
            )

    report_text = "\n".join(report_lines)
    report_path = OUTPUT_ROOT / "sensitivity_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    log(f"  Report saved to: {report_path}")
    print(report_text)

    # ── Save run log ──────────────────────────────────────────────────────
    log_path.write_text("\n".join(log_lines), encoding="utf-8")

    # ══════════════════════════════════════════════════════════════════════
    # 4.  FOREST PLOT
    # ══════════════════════════════════════════════════════════════════════
    _build_forest_plot(df_all, OUTPUT_ROOT)


def _build_forest_plot(df_all: pd.DataFrame, out_dir: Path):
    """
    Three-panel forest plot: one panel per varied parameter (λ, κ, α).
    Within each panel: horizontal bars for each focal CV δ parameter,
    coloured by the parameter value being tested.
    Error bars = posterior SD.
    """

    CV_FOCAL = ["δ_vibrancy", "δ_pleasantness", "δ_walkability",
                "δ_safety", "δ_experiential"]

    varied_params = ["half_life", "kappa", "alpha"]
    param_labels  = {
        "half_life": "λ  (via HALF_LIFE)",
        "kappa":     "κ  (shrinkage)",
        "alpha":     "α  (prior mixing)",
    }

    fig = plt.figure(figsize=(18, 9))
    fig.suptitle("Hyperparameter Sensitivity: CV Place-Effect Posteriors (δ)",
                 fontsize=14, fontweight="bold", y=0.99)

    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.45)

    palette = ["#1D9E75", "#2196F3", "#E24B4A", "#FF9800", "#9C27B0"]

    for col_idx, vparam in enumerate(varied_params):
        ax = fig.add_subplot(gs[0, col_idx])

        # rows for this parameter: baseline + its variants
        sub = df_all[
            (df_all["varied_param"] == vparam) |
            (df_all["varied_param"] == "baseline")
        ].copy().drop_duplicates(subset=["tag"])

        sub = sub.sort_values(vparam)
        n_runs = len(sub)

        y_base  = np.arange(len(CV_FOCAL))
        spacing = 0.18
        offsets = np.linspace(-(n_runs-1)/2 * spacing,
                               (n_runs-1)/2 * spacing, n_runs)

        for ri, (_, run_row) in enumerate(sub.iterrows()):
            run_label = (
                f"HL={run_row['half_life']}"  if vparam == "half_life" else
                f"κ={run_row['kappa']}"       if vparam == "kappa" else
                f"α={run_row['alpha']}"
            )
            is_baseline = run_row["varied_param"] == "baseline"
            lw  = 2.5 if is_baseline else 1.0
            ls  = "-" if is_baseline else "--"
            col = palette[ri % len(palette)]
            marker = "D" if is_baseline else "o"
            ms = 6 if is_baseline else 5

            for yi, p in enumerate(CV_FOCAL):
                safe     = p.replace(" ","_").replace("(","").replace(")","")
                mean_col = f"{safe}_mean"
                sd_col   = f"{safe}_sd"

                if mean_col not in run_row.index or pd.isna(run_row[mean_col]):
                    continue

                m  = run_row[mean_col]
                sd = run_row[sd_col] if sd_col in run_row.index else 0.0
                y  = y_base[yi] + offsets[ri]

                ax.errorbar(m, y,
                            xerr=sd,
                            fmt=marker,
                            markersize=ms,
                            color=col,
                            ecolor=col,
                            elinewidth=lw,
                            capsize=3,
                            linewidth=lw,
                            linestyle=ls,
                            label=run_label if yi == 0 else "_nolegend_",
                            alpha=0.85)

        ax.axvline(0, color="black", lw=0.8, ls=":")
        ax.set_yticks(y_base)
        ax.set_yticklabels(CV_FOCAL, fontsize=8)
        ax.set_xlabel("Posterior mean ± 1 SD", fontsize=9)
        ax.set_title(f"Varying  {param_labels[vparam]}", fontsize=10, fontweight="bold")
        ax.grid(axis="x", alpha=0.25)
        ax.legend(fontsize=8, loc="lower right")

        # Shade significant-positive region
        ax.axvspan(0, ax.get_xlim()[1] if ax.get_xlim()[1] > 0 else 0.5,
                   alpha=0.03, color="green", label="_nolegend_")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plot_path = OUTPUT_ROOT / "sensitivity_plot.png"
    fig.savefig(str(plot_path), dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Forest plot saved to: {plot_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 5.  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 72)
    print("  PRE-FLIGHT CHECK")
    print("=" * 72)
    for name, path in [("Script 1", SCRIPT_1), ("Script 2", SCRIPT_2),
                       ("Script 3", SCRIPT_3), ("Script 4", SCRIPT_4)]:
        exists = path.exists()
        print(f"  {name}: {'✓' if exists else '✗  NOT FOUND'} {path}")

    print(f"\n  Runs planned: {len(RUN_TABLE)}")
    for tag, cfg in RUN_TABLE.items():
        print(f"    [{tag}]  varied={cfg['varied_param']}  "
              f"HL={cfg['half_life']}  K={cfg['kappa']}  A={cfg['alpha']}")

    print(f"\n  Output root: {OUTPUT_ROOT}")
    print("=" * 72)

    ans = input("\n  Proceed? [y/N] : ").strip().lower()
    if ans == "y":
        main()
    else:
        print("  Aborted.")
