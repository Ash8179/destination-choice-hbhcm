# =============================================================================
# F.1 - Layer 1,2,3 Survey Processing (WFH Removed from Utility)
# =============================================================================
# Utility Function:
#   U_njt = β_GTC · GTC_njt
#           + (β_S_med · D_med_j + β_S_lrg · D_lrg_j)
#           + Σ_k γ_k · LV_nk
#           + ε_njt
#
# Change from V3:
#   - WFH × LV interaction terms (κ_k) are REMOVED from the utility function.
#     Rationale: LR test (Model 3 vs Model 2) was not significant (p=0.62),
#     and WFH effects are now fully modelled in the structural equations.
#
# Structural equations remain the FULL specification:
#   LV_nk = α_k0 + α_k^T·Z_n + β_k^T·PerSen_n + η_k^T·(PerSen_n×WFH_n) + ν_nk
#
#   Author: Zhang Wenyu
#   Date: 2026-03-27
# =============================================================================

import pandas as pd
import numpy as np
import warnings
from factor_analyzer import FactorAnalyzer
import pingouin as pg
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import os

warnings.filterwarnings('ignore')


# =============================================================================
# SECTION 0: CONFIGURATION
# =============================================================================

BASE_DIR   = "/Users/zhangwenyu/Desktop/NUSFYP/Stage 2/HBHCM/Layer 1,2,3/"
OUTPUT_DIR = os.path.join(BASE_DIR, "results_V1_no_wfh_utility")
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLUMN_MAP = {
    'Q1-1':  'age_band',       'Q1-2':  'gender',
    'Q1-3':  'residency',      'Q1-4':  'activity_status',
    'Q1-5':  'work_arrangement','Q1-5a': 'wfh_days',
    'Q1-6':  'work_postal_code','Q1-7':  'income_band',
    'Q1-8':  'education',      'Q1-9':  'license',
    'Q1-11': 'home_postal_code','Q1-12': 'dwelling_type',
    'Q1-13': 'household_size', 'Q1-14': 'vehicle_owned',
    'Q1-14a':'car_count',      'Q1-14b':'car_access',
    'Q1-15': 'main_mode',
    'car_time':    'car_time',  'car_cost':    'car_cost',
    'pt_time':     'pt_time',   'pt_cost':     'pt_cost',
    'pt_transfer': 'pt_transfer',
}

CHOICE_COLS = {
    (1,1):'Q3-1-1',(1,2):'Q3-1-2',(1,3):'Q3-1-3',(1,4):'Q3-1-4',
    (1,5):'Q3-1-5',(1,6):'Q3-1-6',(1,7):'Q3-1-7',(1,8):'Q3-1-8',
    (2,1):'Q3-2-1',(2,2):'Q3-2-2',(2,3):'Q3-2-3',(2,4):'Q3-2-4',
    (2,5):'Q3-2-5',(2,6):'Q3-2-6',(2,7):'Q3-2-7',(2,8):'Q3-2-8',
    (3,1):'Q3-3-1',(3,2):'Q3-3-2',(3,3):'Q3-3-3',(3,4):'Q3-3-4',
    (3,5):'Q3-3-5',(3,6):'Q3-3-6',(3,7):'Q3-3-7',(3,8):'Q3-3-8',
    (4,1):'Q3-4-1',(4,2):'Q3-4-2',(4,3):'Q3-4-3',(4,4):'Q3-4-4',
    (4,5):'Q3-4-5',(4,6):'Q3-4-6',(4,7):'Q3-4-7',(4,8):'Q3-4-8',
}

BLOCK_COL          = None
RETEST_SOURCE      = {1:3, 2:3, 3:3, 4:3}
BLOCK_RECODE_OFFSET= {1:3, 2:0, 3:0, 4:0}

TRAVEL_DELTA = {
    'car_time':    {0:-4,  1:0,  2:5,  3:9,  4:14},
    'car_cost':    {0:-1,  1:0,  2:1,  3:2,  4:4},
    'pt_time':     {0:-9,  1:0,  2:9,  3:17, 4:32},
    'pt_cost':     {0:-1,  1:0,  2:0,  3:0,  4:1},
    'pt_transfer': {0:-1,  1:0,  2:0,  3:0,  4:1},
}
PT_TRANSFER_BASELINE = 1
VOT_PER_MIN          = 0.20   # S$ per minute
TRANSFER_PENALTY     = 5.0    # minutes per transfer

SP_DESIGN = {
    (1,1):{'A':{'scale_j':0,'liveliness':1,'env_comfort':0,'ped_access':0,'safety':0,'experiential':0,'tt_level':4,'cost_level':4,'transfer_level':4},'B':{'scale_j':1,'liveliness':2,'env_comfort':1,'ped_access':1,'safety':1,'experiential':2,'tt_level':2,'cost_level':2,'transfer_level':2}},
    (1,2):{'A':{'scale_j':1,'liveliness':2,'env_comfort':0,'ped_access':0,'safety':1,'experiential':0,'tt_level':1,'cost_level':1,'transfer_level':1},'B':{'scale_j':2,'liveliness':0,'env_comfort':2,'ped_access':1,'safety':0,'experiential':1,'tt_level':2,'cost_level':2,'transfer_level':2}},
    (1,3):{'A':{'scale_j':0,'liveliness':2,'env_comfort':1,'ped_access':1,'safety':1,'experiential':1,'tt_level':4,'cost_level':4,'transfer_level':4},'B':{'scale_j':2,'liveliness':1,'env_comfort':0,'ped_access':2,'safety':2,'experiential':0,'tt_level':0,'cost_level':0,'transfer_level':0}},
    (1,4):{'A':{'scale_j':2,'liveliness':2,'env_comfort':0,'ped_access':2,'safety':2,'experiential':2,'tt_level':4,'cost_level':4,'transfer_level':4},'B':{'scale_j':0,'liveliness':0,'env_comfort':1,'ped_access':0,'safety':1,'experiential':0,'tt_level':3,'cost_level':3,'transfer_level':3}},
    (1,5):{'A':{'scale_j':0,'liveliness':1,'env_comfort':2,'ped_access':2,'safety':1,'experiential':2,'tt_level':3,'cost_level':3,'transfer_level':3},'B':{'scale_j':2,'liveliness':2,'env_comfort':0,'ped_access':0,'safety':0,'experiential':0,'tt_level':1,'cost_level':1,'transfer_level':1}},
    (1,6):{'A':{'scale_j':0,'liveliness':0,'env_comfort':1,'ped_access':0,'safety':2,'experiential':2,'tt_level':3,'cost_level':3,'transfer_level':3},'B':{'scale_j':1,'liveliness':1,'env_comfort':2,'ped_access':2,'safety':1,'experiential':0,'tt_level':1,'cost_level':1,'transfer_level':1}},
    (1,7):{'A':{'scale_j':2,'liveliness':1,'env_comfort':2,'ped_access':2,'safety':2,'experiential':1,'tt_level':0,'cost_level':0,'transfer_level':0},'B':{'scale_j':0,'liveliness':1,'env_comfort':1,'ped_access':0,'safety':1,'experiential':0,'tt_level':3,'cost_level':3,'transfer_level':3}},
    (1,8):{},
    (2,1):{'A':{'scale_j':1,'liveliness':2,'env_comfort':1,'ped_access':0,'safety':0,'experiential':2,'tt_level':0,'cost_level':0,'transfer_level':0},'B':{'scale_j':2,'liveliness':0,'env_comfort':2,'ped_access':2,'safety':1,'experiential':0,'tt_level':4,'cost_level':4,'transfer_level':4}},
    (2,2):{'A':{'scale_j':0,'liveliness':2,'env_comfort':1,'ped_access':1,'safety':2,'experiential':2,'tt_level':1,'cost_level':1,'transfer_level':1},'B':{'scale_j':1,'liveliness':1,'env_comfort':0,'ped_access':2,'safety':1,'experiential':1,'tt_level':4,'cost_level':4,'transfer_level':4}},
    (2,3):{'A':{'scale_j':0,'liveliness':0,'env_comfort':1,'ped_access':1,'safety':0,'experiential':0,'tt_level':0,'cost_level':0,'transfer_level':0},'B':{'scale_j':2,'liveliness':1,'env_comfort':2,'ped_access':0,'safety':1,'experiential':1,'tt_level':1,'cost_level':1,'transfer_level':1}},
    (2,4):{'A':{'scale_j':2,'liveliness':1,'env_comfort':2,'ped_access':0,'safety':2,'experiential':2,'tt_level':4,'cost_level':4,'transfer_level':4},'B':{'scale_j':1,'liveliness':0,'env_comfort':1,'ped_access':2,'safety':1,'experiential':1,'tt_level':1,'cost_level':1,'transfer_level':1}},
    (2,5):{'A':{'scale_j':1,'liveliness':1,'env_comfort':1,'ped_access':2,'safety':2,'experiential':0,'tt_level':3,'cost_level':3,'transfer_level':3},'B':{'scale_j':2,'liveliness':0,'env_comfort':2,'ped_access':1,'safety':1,'experiential':2,'tt_level':0,'cost_level':0,'transfer_level':0}},
    (2,6):{'A':{'scale_j':0,'liveliness':1,'env_comfort':2,'ped_access':2,'safety':1,'experiential':1,'tt_level':2,'cost_level':2,'transfer_level':2},'B':{'scale_j':2,'liveliness':2,'env_comfort':0,'ped_access':0,'safety':2,'experiential':2,'tt_level':0,'cost_level':0,'transfer_level':0}},
    (2,7):{'A':{'scale_j':2,'liveliness':0,'env_comfort':2,'ped_access':1,'safety':2,'experiential':2,'tt_level':1,'cost_level':1,'transfer_level':1},'B':{'scale_j':0,'liveliness':1,'env_comfort':0,'ped_access':0,'safety':0,'experiential':0,'tt_level':4,'cost_level':4,'transfer_level':4}},
    (2,8):{},
    (3,1):{'A':{'scale_j':1,'liveliness':0,'env_comfort':2,'ped_access':0,'safety':0,'experiential':2,'tt_level':4,'cost_level':4,'transfer_level':4},'B':{'scale_j':0,'liveliness':2,'env_comfort':1,'ped_access':2,'safety':2,'experiential':0,'tt_level':1,'cost_level':1,'transfer_level':1}},
    (3,2):{'A':{'scale_j':2,'liveliness':2,'env_comfort':1,'ped_access':2,'safety':2,'experiential':1,'tt_level':1,'cost_level':1,'transfer_level':1},'B':{'scale_j':0,'liveliness':1,'env_comfort':0,'ped_access':1,'safety':0,'experiential':2,'tt_level':2,'cost_level':2,'transfer_level':2}},
    (3,3):{'A':{'scale_j':0,'liveliness':2,'env_comfort':0,'ped_access':1,'safety':1,'experiential':1,'tt_level':3,'cost_level':3,'transfer_level':3},'B':{'scale_j':1,'liveliness':0,'env_comfort':2,'ped_access':2,'safety':2,'experiential':0,'tt_level':0,'cost_level':0,'transfer_level':0}},
    (3,4):{'A':{'scale_j':0,'liveliness':2,'env_comfort':1,'ped_access':0,'safety':0,'experiential':1,'tt_level':0,'cost_level':0,'transfer_level':0},'B':{'scale_j':2,'liveliness':0,'env_comfort':2,'ped_access':1,'safety':2,'experiential':0,'tt_level':3,'cost_level':3,'transfer_level':3}},
    (3,5):{'A':{'scale_j':0,'liveliness':0,'env_comfort':2,'ped_access':2,'safety':2,'experiential':2,'tt_level':0,'cost_level':0,'transfer_level':0},'B':{'scale_j':2,'liveliness':1,'env_comfort':0,'ped_access':0,'safety':0,'experiential':1,'tt_level':1,'cost_level':1,'transfer_level':1}},
    (3,6):{'A':{'scale_j':2,'liveliness':0,'env_comfort':1,'ped_access':2,'safety':1,'experiential':1,'tt_level':2,'cost_level':2,'transfer_level':2},'B':{'scale_j':1,'liveliness':2,'env_comfort':0,'ped_access':1,'safety':0,'experiential':2,'tt_level':3,'cost_level':3,'transfer_level':3}},
    (3,7):{'A':{'scale_j':1,'liveliness':1,'env_comfort':2,'ped_access':2,'safety':2,'experiential':1,'tt_level':0,'cost_level':0,'transfer_level':0},'B':{'scale_j':0,'liveliness':1,'env_comfort':1,'ped_access':0,'safety':1,'experiential':0,'tt_level':4,'cost_level':4,'transfer_level':4}},
    (3,8):{},
    (4,1):{'A':{'scale_j':0,'liveliness':1,'env_comfort':2,'ped_access':0,'safety':0,'experiential':1,'tt_level':3,'cost_level':3,'transfer_level':3},'B':{'scale_j':2,'liveliness':0,'env_comfort':1,'ped_access':1,'safety':1,'experiential':2,'tt_level':4,'cost_level':4,'transfer_level':4}},
    (4,2):{'A':{'scale_j':0,'liveliness':0,'env_comfort':2,'ped_access':1,'safety':0,'experiential':0,'tt_level':1,'cost_level':1,'transfer_level':1},'B':{'scale_j':1,'liveliness':1,'env_comfort':0,'ped_access':0,'safety':2,'experiential':2,'tt_level':2,'cost_level':2,'transfer_level':2}},
    (4,3):{'A':{'scale_j':0,'liveliness':2,'env_comfort':0,'ped_access':2,'safety':1,'experiential':0,'tt_level':0,'cost_level':0,'transfer_level':0},'B':{'scale_j':1,'liveliness':0,'env_comfort':1,'ped_access':0,'safety':2,'experiential':1,'tt_level':2,'cost_level':2,'transfer_level':2}},
    (4,4):{'A':{'scale_j':2,'liveliness':1,'env_comfort':2,'ped_access':1,'safety':2,'experiential':0,'tt_level':2,'cost_level':2,'transfer_level':2},'B':{'scale_j':1,'liveliness':2,'env_comfort':0,'ped_access':0,'safety':0,'experiential':1,'tt_level':3,'cost_level':3,'transfer_level':3}},
    (4,5):{'A':{'scale_j':1,'liveliness':0,'env_comfort':2,'ped_access':2,'safety':2,'experiential':1,'tt_level':4,'cost_level':4,'transfer_level':4},'B':{'scale_j':2,'liveliness':1,'env_comfort':1,'ped_access':1,'safety':0,'experiential':2,'tt_level':0,'cost_level':0,'transfer_level':0}},
    (4,6):{'A':{'scale_j':2,'liveliness':2,'env_comfort':1,'ped_access':2,'safety':1,'experiential':0,'tt_level':2,'cost_level':2,'transfer_level':2},'B':{'scale_j':1,'liveliness':1,'env_comfort':0,'ped_access':1,'safety':0,'experiential':2,'tt_level':3,'cost_level':3,'transfer_level':3}},
    (4,7):{'A':{'scale_j':2,'liveliness':1,'env_comfort':2,'ped_access':2,'safety':2,'experiential':2,'tt_level':1,'cost_level':1,'transfer_level':1},'B':{'scale_j':0,'liveliness':0,'env_comfort':1,'ped_access':0,'safety':0,'experiential':0,'tt_level':3,'cost_level':3,'transfer_level':3}},
    (4,8):{},
}

for blk, src in RETEST_SOURCE.items():
    SP_DESIGN[(blk, 8)] = SP_DESIGN[(blk, src)]

CHOICE_TEXT_MAP = {1:'A', 2:'B', 3:'C'}

INCOME_MIDPOINT = {
    1:0,2:500,3:1500,4:2500,5:3500,6:4500,7:5500,8:6500,9:7500,
    10:8500,11:9500,12:10500,13:11500,14:13500,15:17500,16:25000,17:np.nan,
}
AGE_MIDPOINT = {1:24, 2:35, 3:45, 4:55, 5:65}

ATTENTION_COL    = 'Q2_24'
ATTENTION_VALUE  = 4
MIN_DURATION_SEC = 180

SUBSCALES = {
    'LV_Pleasantness': ['Q2_5','Q2_8','Q2_7','Q2_9'],
    'LV_Vibrancy':     ['Q2_3','Q2_4','Q2_20'],
    'LV_Walkability':  ['Q2_12','Q2_16','Q2_10'],
    'LV_Experiential': ['Q2_19','Q2_22','Q2_23'],
}
REVERSE_ITEMS = []

IMAGE_COUNT = 12
IMAGE_DIMS  = ['liveliness','env_comfort','ped_access','safety','experiential']
IMAGE_COLS  = {
    (img, dim_idx+1): f'Q4-{img}_{dim_idx+1}'
    for img in range(1, IMAGE_COUNT+1)
    for dim_idx in range(len(IMAGE_DIMS))
}


# =============================================================================
# HELPERS
# =============================================================================

def apply_delta(baseline, var_name, level):
    if pd.isna(baseline) or pd.isna(level):
        return np.nan
    return round(float(baseline) + TRAVEL_DELTA[var_name].get(int(level), 0), 2)

def detect_block(person_series, choice_cols):
    for blk in range(1, 5):
        col = choice_cols.get((blk, 1))
        if col and col in person_series.index:
            val = person_series[col]
            if pd.notna(val) and str(val).strip() not in ('','nan'):
                return blk
    return np.nan

def count_params(names_dict):
    return sum(len(v) if isinstance(v, list) else 1 for v in names_dict.values())


# =============================================================================
# SECTION 1: DATA LOADING AND CLEANING
# =============================================================================

print("=" * 70)
print("SECTION 1: Data Loading and Cleaning")
print("=" * 70)

input_path = os.path.join(BASE_DIR, "SurveyV4.csv")
raw = pd.read_csv(input_path, skiprows=[1, 2])
print(f"Raw rows loaded: {len(raw)}")

df = raw.copy()
df['Finished'] = pd.to_numeric(df['Finished'], errors='coerce')
df = df[df['Finished'] == 1].copy()
print(f"After completion filter: {len(df)}")

df['Duration'] = pd.to_numeric(df['Duration (in seconds)'], errors='coerce')
df = df[df['Duration'] > MIN_DURATION_SEC].copy()
print(f"After duration filter (>{MIN_DURATION_SEC}s): {len(df)}")

df[ATTENTION_COL] = pd.to_numeric(df[ATTENTION_COL], errors='coerce')
df = df[df[ATTENTION_COL] == ATTENTION_VALUE].copy()
print(f"After attention check filter: {len(df)}")

df = df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns})
df = df.reset_index(drop=True)
df['person_id'] = df.index + 1
print(f"\nFinal valid sample: {len(df)} respondents")

if len(df) == 0:
    raise ValueError("No valid respondents after cleaning.")


# =============================================================================
# SECTION 2: SOCIO-DEMOGRAPHIC VARIABLES (Z_n)
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 2: Socio-Demographic Variables (Z_n)")
print("=" * 70)

df['age_num']    = pd.to_numeric(df['age_band'],    errors='coerce').map(AGE_MIDPOINT)
df['income_num'] = pd.to_numeric(df['income_band'], errors='coerce').map(INCOME_MIDPOINT)

q14b = pd.to_numeric(df.get('car_access', pd.Series([np.nan]*len(df))), errors='coerce')
df['car_avail'] = q14b.apply(lambda x: 1 if x in [1, 2] else 0)

q1_5 = pd.to_numeric(df.get('work_arrangement', pd.Series([np.nan]*len(df))), errors='coerce')
df['is_wfh'] = q1_5.apply(lambda x: 1 if x in [2, 3] else 0)

q1_5a = pd.to_numeric(df.get('wfh_days', pd.Series([np.nan]*len(df))), errors='coerce')
df['wfh_days_num'] = np.where(df['is_wfh'] == 1, (q1_5a - 1).clip(lower=0), 0.0)
df['wfh_days_num'] = pd.to_numeric(df['wfh_days_num'], errors='coerce').fillna(0)

df['hh_size'] = pd.to_numeric(
    df.get('household_size', pd.Series([np.nan]*len(df))), errors='coerce'
)

zn_cols   = ['age_num','income_num','car_avail','hh_size','wfh_days_num']
scaler_zn = StandardScaler()
zn_valid  = df[zn_cols].dropna()
df.loc[zn_valid.index, [f'{c}_z' for c in zn_cols]] = scaler_zn.fit_transform(zn_valid)

print("car_avail distribution:")
print(df['car_avail'].value_counts().sort_index().to_string())
print("\nis_wfh distribution:")
print(df['is_wfh'].value_counts().sort_index().to_string())
print("\nZ_n summary:")
print(df[zn_cols].describe().round(2))


# =============================================================================
# SECTION 3: LATENT VARIABLE SCORING
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 3: Latent Variable Scoring (EFA-revised)")
print("=" * 70)

all_items = [item for items in SUBSCALES.values() for item in items]
for col in all_items:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

for col in REVERSE_ITEMS:
    if col in df.columns:
        df[col] = 6 - df[col]

for lv, items in SUBSCALES.items():
    available = [c for c in items if c in df.columns]
    df[lv]    = df[available].mean(axis=1)
    print(f"  {lv}: {len(available)} items ({available})")

lv_names   = list(SUBSCALES.keys())
scaler_lv  = StandardScaler()
lv_valid   = df[lv_names].dropna()
lv_z_names = [f'{lv}_z' for lv in lv_names]
df.loc[lv_valid.index, lv_z_names] = scaler_lv.fit_transform(lv_valid)

print("\nLatent variable summary:")
print(df[lv_names].describe().round(3))


# =============================================================================
# SECTION 4: RELIABILITY AND VALIDITY
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 4: Reliability and Validity")
print("=" * 70)

# =============================================================================
# 4.1 Cronbach's Alpha
# =============================================================================

reliability_rows = []

for lv, items in SUBSCALES.items():
    available = [c for c in items if c in df.columns]
    subset    = df[available].dropna()

    if len(subset) < 10 or len(available) < 2:
        continue

    alpha_val, ci = pg.cronbach_alpha(data=subset)

    reliability_rows.append({
        'Latent Variable':   lv,
        'N Items':           len(available),
        'N Valid':           len(subset),
        "Cronbach's α":      round(alpha_val, 3),
        '95% CI Lower':      round(ci[0], 3),
        '95% CI Upper':      round(ci[1], 3),
        'Acceptable (≥.70)': '✓' if alpha_val >= 0.70 else '✗',
    })

reliability_df = pd.DataFrame(reliability_rows)

print("\nTable 1 – Reliability (Cronbach's Alpha)")
print(reliability_df.to_string(index=False))


# =============================================================================
# 4.2 Factor Loadings + AVE + CR
# =============================================================================

validity_rows = []
all_loadings  = {}

for lv, items in SUBSCALES.items():
    available = [c for c in items if c in df.columns]
    subset    = df[available].dropna()

    if len(subset) < 10 or len(available) < 2:
        continue

    fa = FactorAnalyzer(n_factors=1, rotation=None)
    fa.fit(subset)

    loadings = fa.loadings_[:, 0]
    all_loadings[lv] = dict(zip(available, loadings))

    ave = float(np.mean(loadings ** 2))
    cr  = float(
        np.sum(np.abs(loadings)) ** 2 /
        (np.sum(np.abs(loadings)) ** 2 + np.sum(1 - loadings ** 2))
    )

    validity_rows.append({
        'Latent Variable':       lv,
        'Mean |Loading|':        round(float(np.mean(np.abs(loadings))), 3),
        'Min |Loading|':         round(float(np.min(np.abs(loadings))), 3),
        'AVE':                   round(ave, 3),
        'Composite Reliability': round(cr, 3),
        'AVE ≥ .50':             '✓' if ave >= 0.50 else '✗',
        'CR ≥ .70':              '✓' if cr  >= 0.70 else '✗',
    })

validity_df = pd.DataFrame(validity_rows)

print("\nTable 2 – Convergent Validity (AVE & CR)")
print(validity_df.to_string(index=False))


# =============================================================================
# 4.3 Discriminant Validity (Fornell–Larcker)
# =============================================================================

lv_names = list(SUBSCALES.keys())

lv_corr = df[lv_names].corr().round(3)

ave_lookup = {
    row['Latent Variable']: row['AVE']
    for _, row in validity_df.iterrows()
}

disc_matrix = lv_corr.copy()

for lv in lv_names:
    if lv in ave_lookup:
        disc_matrix.loc[lv, lv] = round(np.sqrt(ave_lookup[lv]), 3)

print("\nTable 3 – Discriminant Validity (Fornell–Larcker)")
print(disc_matrix.to_string())


# =============================================================================
# 4.4 HTMT (with dual thresholds + decision rules)
# =============================================================================

def htmt_lv(df, items_a, items_b):
    vals = []
    for a in items_a:
        for b in items_b:
            if a in df.columns and b in df.columns:
                pair = df[[a, b]].dropna()
                if len(pair) > 5:
                    vals.append(abs(pair[a].corr(pair[b])))

    if len(vals) == 0:
        return np.nan
    return np.mean(vals)


htmt_matrix = pd.DataFrame(index=lv_names, columns=lv_names, dtype=float)

for i, lv1 in enumerate(lv_names):
    for j, lv2 in enumerate(lv_names):
        if i == j:
            htmt_matrix.loc[lv1, lv2] = 1.0
        elif i < j:
            items_a = [c for c in SUBSCALES[lv1] if c in df.columns]
            items_b = [c for c in SUBSCALES[lv2] if c in df.columns]

            val = htmt_lv(df, items_a, items_b)

            htmt_matrix.loc[lv1, lv2] = val
            htmt_matrix.loc[lv2, lv1] = val


print("\nTable 4 – HTMT Matrix")
print(htmt_matrix.round(3).to_string())


# -----------------------------
# HTMT decision rules
# -----------------------------

htmt_upper = htmt_matrix.where(
    np.triu(np.ones(htmt_matrix.shape), 1).astype(bool)
)

htmt_085_pass = htmt_upper < 0.85
htmt_090_pass = htmt_upper < 0.90

print("\nHTMT < 0.85 (strict criterion, upper triangle):")
print(htmt_085_pass)

print("\nHTMT < 0.90 (lenient criterion, upper triangle):")
print(htmt_090_pass)


# summary flags per construct pair
htmt_summary = []

for i, lv1 in enumerate(lv_names):
    for j, lv2 in enumerate(lv_names):
        if i < j:
            val = htmt_matrix.loc[lv1, lv2]

            htmt_summary.append({
                "Construct Pair": f"{lv1} - {lv2}",
                "HTMT": round(float(val), 3),
                "Pass <0.85": "✓" if val < 0.85 else "✗",
                "Pass <0.90": "✓" if val < 0.90 else "✗",
            })

htmt_summary_df = pd.DataFrame(htmt_summary)

print("\nTable 5 – HTMT Decision Summary")
print(htmt_summary_df.to_string(index=False))


# =============================================================================
# 4.5 McDonald's Omega
# =============================================================================

omega_rows = []

for lv, load_dict in all_loadings.items():
    loadings = np.array(list(load_dict.values()))

    if len(loadings) < 2:
        continue

    error_var = 1 - loadings**2

    omega = (np.sum(loadings))**2 / (
        (np.sum(loadings))**2 + np.sum(error_var)
    )

    omega_rows.append({
        'Latent Variable': lv,
        "McDonald's ω": round(float(omega), 3),
        'ω ≥ .70': '✓' if omega >= 0.70 else '✗'
    })

omega_df = pd.DataFrame(omega_rows)

print("\nTable 6 – McDonald's Omega")
print(omega_df.to_string(index=False))


# =============================================================================
# 4.6 Factor Loadings
# =============================================================================

loading_rows = []

for lv, load_dict in all_loadings.items():
    for item, loading in load_dict.items():
        loading_rows.append({
            'Factor': lv,
            'Item': item,
            'Loading': round(float(loading), 3),
            'Loading²': round(float(loading**2), 3),
            '≥ .40': '✓' if abs(loading) >= 0.40 else '✗',
        })

loadings_df = pd.DataFrame(loading_rows)

print("\nTable 7 – Factor Loadings")
print(loadings_df.to_string(index=False))


# =============================================================================
# SECTION 5: IMAGE PERCEPTION RATINGS
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 5: Image Perception Ratings")
print("=" * 70)

for img in range(1, IMAGE_COUNT+1):
    for dim_idx in range(len(IMAGE_DIMS)):
        col = IMAGE_COLS.get((img, dim_idx+1))
        if col and col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

image_rows = []
for _, person in df.iterrows():
    pid = person['person_id']
    for img in range(1, IMAGE_COUNT+1):
        row = {'person_id': pid, 'image': img}
        for dim_idx, dim_name in enumerate(IMAGE_DIMS):
            col = IMAGE_COLS.get((img, dim_idx+1))
            row[dim_name] = pd.to_numeric(person.get(col, np.nan), errors='coerce')
        image_rows.append(row)

image_long_df = pd.DataFrame(image_rows)

print("\n--- Sanity Check 5.1: Rating range ---")
range_issues = 0
for dim in IMAGE_DIMS:
    oob = image_long_df[image_long_df[dim].notna() & ~image_long_df[dim].isin([1,2,3])]
    if len(oob):
        print(f"  WARNING: {dim} — {len(oob)} out-of-range values")
        range_issues += 1
if range_issues == 0:
    print("  All ratings within [1, 2, 3].")

print("\n--- Sanity Check 5.2: Missing rate ---")
any_warning = False
missing_rows = []
for img in range(1, IMAGE_COUNT+1):
    subset = image_long_df[image_long_df['image'] == img]
    for dim in IMAGE_DIMS:
        miss = subset[dim].isna().mean()
        if miss > 0.10:
            print(f"  WARNING: Image {img} | {dim}: {miss:.1%} missing")
            any_warning = True
        missing_rows.append({'Image':img,'Dimension':dim,
                              'N Valid':subset[dim].notna().sum(),'Missing %':round(miss*100,1)})
if not any_warning:
    print("  All images within acceptable missing rate.")
missing_df = pd.DataFrame(missing_rows)

print("\n--- Sanity Check 5.3: Straight-lining ---")
total_cells = IMAGE_COUNT * len(IMAGE_DIMS)
sl_pids = []
for pid, grp in image_long_df.groupby('person_id'):
    vals = grp[IMAGE_DIMS].values.flatten()
    vals = vals[~np.isnan(vals)]
    if len(vals) == total_cells and len(set(vals)) == 1:
        sl_pids.append(pid)
if sl_pids:
    print(f"  WARNING: {len(sl_pids)} straight-liners: {sl_pids}")
else:
    print("  No straight-liners detected.")

print("\n--- Image-level mean ratings ---")
image_means = image_long_df.groupby('image')[IMAGE_DIMS].mean().round(2)
print(image_means.to_string())

perception_rename_map = {
    'liveliness':   'ps_vib',
    'env_comfort':  'ps_pls',
    'ped_access':   'ps_wlk',
    'safety':       'ps_saf',
    'experiential': 'ps_exp',
}
person_perception = (
    image_long_df.groupby('person_id')[IMAGE_DIMS].mean()
    .rename(columns=perception_rename_map)
    .reset_index()
)
df = df.merge(person_perception, on='person_id', how='left')

ps_cols   = ['ps_vib','ps_pls','ps_wlk','ps_saf','ps_exp']
scaler_ps = StandardScaler()
ps_valid  = df[ps_cols].dropna()
df.loc[ps_valid.index, [f'{c}_z' for c in ps_cols]] = scaler_ps.fit_transform(ps_valid)

print(f"\nImage perception data: {len(image_long_df)} rows")
print(f"\nPerception Sensitivity (PerSen_n) summary:")
print(df[ps_cols].describe().round(3))


# =============================================================================
# SECTION 6: SP DATA → LONG FORMAT
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 6: SP Data — Wide to Long Format")
print("=" * 70)

long_rows = []
vc_rows   = []
retest_rows = []

for _, person in df.iterrows():
    pid = person['person_id']
    block = (int(pd.to_numeric(person[BLOCK_COL], errors='coerce'))
             if BLOCK_COL and BLOCK_COL in person.index
             else detect_block(person, CHOICE_COLS))
    if pd.isna(block):
        print(f"  WARNING: person_id={pid} — block undetermined, skipping.")
        continue
    block = int(block)

    car_time_base    = pd.to_numeric(person.get('car_time', np.nan), errors='coerce')
    car_cost_base    = pd.to_numeric(person.get('car_cost', np.nan), errors='coerce')
    pt_time_base     = pd.to_numeric(person.get('pt_time',  np.nan), errors='coerce')
    pt_cost_base     = pd.to_numeric(person.get('pt_cost',  np.nan), errors='coerce')
    pt_transfer_base = float(PT_TRANSFER_BASELINE)

    lv_vals   = {name: person.get(name, np.nan) for name in lv_z_names}
    zn_vals   = {f'{c}_z': person.get(f'{c}_z', np.nan) for c in zn_cols}
    ps_vals   = {c: person.get(c, np.nan) for c in ps_cols}
    ps_z_vals = {f'{c}_z': person.get(f'{c}_z', np.nan) for c in ps_cols}

    task_choices = {}

    for task in range(1, 9):
        col_key    = (block, task)
        choice_col = CHOICE_COLS.get(col_key)
        if not choice_col or choice_col not in person.index:
            continue
        raw_val = pd.to_numeric(person[choice_col], errors='coerce')
        if pd.notna(raw_val):
            raw_val = raw_val - BLOCK_RECODE_OFFSET.get(block, 0)
        chosen_alt = CHOICE_TEXT_MAP.get(int(raw_val), np.nan) if pd.notna(raw_val) else np.nan
        if pd.isna(chosen_alt):
            continue

        task_choices[task] = chosen_alt
        design    = SP_DESIGN.get(col_key, {})
        is_vc     = (task == 7)
        is_retest = (task == 8)

        for alt in ['A','B','C']:
            att = design.get(alt, {})
            if alt != 'C':
                tt_level       = att.get('tt_level',       np.nan)
                cost_level     = att.get('cost_level',     np.nan)
                transfer_level = att.get('transfer_level', np.nan)
                act_car_time    = apply_delta(car_time_base,    'car_time',    tt_level)
                act_car_cost    = apply_delta(car_cost_base,    'car_cost',    cost_level)
                act_pt_time     = apply_delta(pt_time_base,     'pt_time',     tt_level)
                act_pt_cost     = apply_delta(pt_cost_base,     'pt_cost',     cost_level)
                act_pt_transfer = apply_delta(pt_transfer_base, 'pt_transfer', transfer_level)
                scale_j_raw     = att.get('scale_j', 0)
            else:
                act_car_time = act_car_cost = act_pt_time = act_pt_cost = 0.0
                act_pt_transfer = 0.0
                scale_j_raw = 0
                tt_level = cost_level = transfer_level = np.nan

            row = {
                'person_id': pid, 'block': block, 'task': task,
                'task_type': ('Extreme_VC' if is_vc else
                              'Test_Retest' if is_retest else 'Standard'),
                'alt': alt, 'chosen': 1 if alt == chosen_alt else 0,
                'car_time': act_car_time, 'car_cost': act_car_cost,
                'pt_time': act_pt_time,   'pt_cost': act_pt_cost,
                'pt_transfer': act_pt_transfer,
                'tt_level': tt_level, 'cost_level': cost_level,
                'transfer_level': transfer_level, 'scale_j_raw': scale_j_raw,
                'is_wfh':    person.get('is_wfh', 0),
                'wfh_days':  person.get('wfh_days_num', 0),
                'car_avail': person.get('car_avail', 0),
                'age_num':   person.get('age_num', np.nan),
                'income_num':person.get('income_num', np.nan),
                'hh_size':   person.get('hh_size', np.nan),
            }
            row.update(lv_vals)
            row.update(zn_vals)
            row.update(ps_vals)
            row.update(ps_z_vals)
            long_rows.append(row)

    vc_choice = task_choices.get(7, np.nan)
    vc_rows.append({'person_id': pid, 'block': block,
                    'vc_choice': vc_choice, 'vc_pass': 1 if vc_choice == 'A' else 0})

    src_task   = RETEST_SOURCE.get(block)
    src_choice = task_choices.get(src_task, np.nan)
    ret_choice = task_choices.get(8, np.nan)
    retest_rows.append({'person_id': pid, 'block': block,
                        'source_task': src_task, 'source_choice': src_choice,
                        'retest_choice': ret_choice,
                        'retest_match': 1 if src_choice == ret_choice else 0})

long_df   = pd.DataFrame(long_rows)
vc_df     = pd.DataFrame(vc_rows)
retest_df = pd.DataFrame(retest_rows)
vc_pass_ids = vc_df[vc_df['vc_pass'] == 1]['person_id'].unique()

long_df['obs_id']     = (long_df['person_id'].astype(str) + '_'
                         + long_df['block'].astype(str) + '_'
                         + long_df['task'].astype(str))
long_df['obs_id_num'] = pd.Categorical(long_df['obs_id']).codes
long_df['alt_num']    = long_df['alt'].map({'A':0,'B':1,'C':2})

print(f"Long format rows     : {len(long_df)}")
print(f"  Standard tasks     : {len(long_df[long_df['task_type']=='Standard'])}")
print(f"  Extreme_VC         : {len(long_df[long_df['task_type']=='Extreme_VC'])}")
print(f"  Test-Retest        : {len(long_df[long_df['task_type']=='Test_Retest'])}")

vc_pass_rate = vc_df['vc_pass'].mean()
print(f"\nExtreme_VC pass rate : {vc_pass_rate:.1%} ({vc_df['vc_pass'].sum()} / {len(vc_df)})")
vc_fails = vc_df[vc_df['vc_pass'] == 0]
if len(vc_fails):
    print(f"  Failing: {vc_fails['person_id'].tolist()}")

retest_rate = retest_df['retest_match'].mean()
print(f"Test-retest match    : {retest_rate:.1%} ({retest_df['retest_match'].sum()} / {len(retest_df)})")

estimation_df = long_df[long_df['task_type'] == 'Standard'].copy()
print(f"Estimation sample    : {len(estimation_df)} rows (standard tasks only)")

dest_rows = estimation_df[estimation_df['alt'] != 'C']
print("\nTravel value sanity check (dest alts):")
for var in ['car_time','car_cost','pt_time','pt_cost','pt_transfer']:
    neg = (dest_rows[var] < 0).sum()
    print(f"  {var:15s}: min={dest_rows[var].min():.2f}  max={dest_rows[var].max():.2f}  "
          f"{'WARNING: negatives!' if neg > 0 else 'OK'}")


# =============================================================================
# SECTION 7: SCALE DUMMY VARIABLES
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 7: Scale Dummy Variables (Value-based)")
print("=" * 70)

def create_scale_dummies(scale_val):
    """Map discrete scale_j values to treatment dummies.
    scale_j = 0 → Small (reference): D_med=0, D_lrg=0
    scale_j = 1 → Medium:            D_med=1, D_lrg=0
    scale_j = 2 → Large:             D_med=0, D_lrg=1
    """
    if pd.isna(scale_val):
        return 0, 0
    scale_val = int(scale_val)
    if scale_val == 1:
        return 1, 0
    elif scale_val == 2:
        return 0, 1
    return 0, 0  # Small or unknown → reference

long_df[['D_med','D_lrg']] = long_df['scale_j_raw'].apply(
    lambda x: pd.Series(create_scale_dummies(x))
)
estimation_df[['D_med','D_lrg']] = estimation_df['scale_j_raw'].apply(
    lambda x: pd.Series(create_scale_dummies(x))
)

print("Scale mapping: 0=Small (ref) | 1=Medium (D_med=1) | 2=Large (D_lrg=1)")
dest_estimation = estimation_df[estimation_df['alt'] != 'C']
scale_dist = pd.DataFrame({
    'Category': ['Small (ref)','Medium','Large'],
    'scale_j': [0,1,2], 'D_med': [0,1,0], 'D_lrg': [0,0,1],
    'Count': [
        ((dest_estimation['D_med']==0) & (dest_estimation['D_lrg']==0)).sum(),
        dest_estimation['D_med'].sum(),
        dest_estimation['D_lrg'].sum(),
    ],
})
scale_dist['Proportion'] = (scale_dist['Count'] / scale_dist['Count'].sum()).round(3)
print(scale_dist.to_string(index=False))
if (scale_dist['Count'] == 0).any():
    print("\n⚠  WARNING: Some scale categories have zero samples!")
else:
    print("\n✓ All scale categories have sufficient samples.")


# =============================================================================
# SECTION 8: MNL MODELS (WFH REMOVED FROM UTILITY)
# =============================================================================
# V1 Modification:
#   The original V3 had a Model 3 with LV × WFH interaction terms in the
#   utility function. Those terms are removed here because:
#     (a) LR test (Model 3 vs Model 2) was not significant (p = 0.62)
#     (b) WFH effects are now fully captured in the structural equations
#
# Models estimated:
#   Model 1: GTC only
#   Model 2: + Scale dummies + Latent Variables  ← final model
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 8: MNL Models — WFH Removed from Utility")
print("=" * 70)

try:
    import pylogit as pl
    from scipy import stats as scipy_stats
    from collections import OrderedDict

    def run_mnl(model_df, spec, names, label):
        """Fit an MNL model and print a formatted results table."""
        spec  = OrderedDict(spec)
        names = OrderedDict(names)
        model_df = model_df.copy()
        model_df['alt_num']    = model_df['alt_num'].astype(int)
        model_df['obs_id_num'] = model_df['obs_id_num'].astype(int)
        model_df['chosen']     = model_df['chosen'].astype(int)

        n_params = count_params(names)
        print(f"\n{'─'*60}")
        print(f"  {label}")
        print(f"  Parameters       : {n_params}")
        print(f"  Choice situations: {model_df[model_df['chosen']==1].shape[0]}")
        print(f"{'─'*60}")

        mnl = pl.create_choice_model(
            data=model_df, alt_id_col='alt_num', obs_id_col='obs_id_num',
            choice_col='chosen', specification=spec, names=names, model_type='MNL',
        )
        mnl.fit_mle(np.zeros(n_params))

        res = pd.DataFrame({
            'Model':     label,
            'Parameter': mnl.params.index,
            'Estimate':  mnl.params.values.round(4),
            'Std Error': mnl.standard_errors.values.round(4),
            't-stat':    mnl.tvalues.values.round(3),
            'p-value':   mnl.pvalues.values.round(4),
            'Sig':       ['***' if p<0.001 else '**' if p<0.01
                          else '*' if p<0.05 else '.' if p<0.10 else ''
                          for p in mnl.pvalues.values],
        })
        print(res[['Parameter','Estimate','Std Error','t-stat','p-value','Sig']]
              .to_string(index=False))

        ll_final = getattr(mnl,'llf',None) or getattr(mnl,'log_likelihood',None)
        ll_null  = getattr(mnl,'null_llf',None) or getattr(mnl,'log_likelihood_null',None)
        aic      = getattr(mnl,'aic',None)
        bic      = getattr(mnl,'bic',None)

        if ll_null is None:
            n_chosen = model_df[model_df['chosen']==1].shape[0]
            n_alts   = model_df['alt_num'].nunique()
            ll_null  = n_chosen * np.log(1.0 / n_alts)
        if aic is None:
            aic = -2 * ll_final + 2 * n_params
        if bic is None:
            bic = -2 * ll_final + n_params * np.log(model_df[model_df['chosen']==1].shape[0])

        rho2 = 1 - ll_final / ll_null
        print(f"\n  LL(final)  : {ll_final:.4f}")
        print(f"  LL(null)   : {ll_null:.4f}")
        print(f"  McFadden ρ²: {rho2:.4f}")
        print(f"  AIC        : {aic:.2f}")
        print(f"  BIC        : {bic:.2f}")

        mnl._ll_final = ll_final
        mnl._ll_null  = ll_null
        mnl._aic      = aic
        mnl._bic      = bic
        mnl._rho2     = rho2
        return mnl, res

    # Prepare estimation dataset
    model_df_base = estimation_df.copy()
    fill_zero_cols = ['car_time','car_cost','pt_time','pt_cost','pt_transfer',
                      'D_med','D_lrg'] + ps_cols
    for col in fill_zero_cols:
        if col in model_df_base.columns:
            model_df_base[col] = model_df_base[col].fillna(0)

    # -------------------------------------------------------------------
    # GTC calculation
    # GTC = car_time + pt_time
    #       + (car_cost + pt_cost) / VOT_PER_MIN   [money → equiv. minutes]
    #       + pt_transfer * TRANSFER_PENALTY        [transfer penalty in minutes]
    # For alt C (status-quo / no-destination), GTC is set to 0.
    # -------------------------------------------------------------------
    model_df_base['gtc'] = (
        model_df_base['car_time'] + model_df_base['pt_time']
        + (model_df_base['car_cost'] + model_df_base['pt_cost']) / VOT_PER_MIN
        + model_df_base['pt_transfer'] * TRANSFER_PENALTY
    )
    model_df_base.loc[model_df_base['alt_num'] == 2, 'gtc'] = 0.0

    # ---------------------------------------------------------------
    # Write gtc back to estimation_df so it is exported to Excel
    # ---------------------------------------------------------------
    estimation_df['gtc'] = model_df_base['gtc']

    model_df = model_df_base.dropna(subset=lv_z_names).copy()

    EXCLUDE_VC_FAIL = False
    if EXCLUDE_VC_FAIL:
        model_df = model_df[model_df['person_id'].isin(vc_pass_ids)].copy()
        print(f"\n [Sensitivity] VC-pass filter: {model_df['person_id'].nunique()} respondents")
    else:
        print(f"\n [Full sample] {model_df['person_id'].nunique()} respondents (VC filter disabled)")

    model_df = model_df.sort_values(['obs_id_num','alt_num']).reset_index(drop=True)
    model_df['alt_num']    = model_df['alt_num'].astype(int)
    model_df['obs_id_num'] = model_df['obs_id_num'].astype(int)
    model_df['chosen']     = model_df['chosen'].astype(int)

    check = model_df.groupby('obs_id_num').agg(
        n_alts=('alt_num','nunique'), chosen_sum=('chosen','sum'))
    assert check['n_alts'].min() >= 2
    assert (check['chosen_sum'] == 1).all()

    dest_alts = [0, 1]

    print(f"\nSample overview:")
    print(f"  Respondents      : {model_df['person_id'].nunique()}")
    print(f"  Choice situations: {model_df['obs_id_num'].nunique()}")
    print(f"  Rows             : {len(model_df)}")
    shares = model_df[model_df['chosen']==1].groupby('alt')['chosen'].count()
    print(f"  Choice shares    :\n{(shares/shares.sum()).round(3).to_string()}")

    print(f"\nGTC summary (dest alts A and B):")
    print(model_df[model_df['alt_num']!=2]['gtc'].describe().round(3).to_string())

    # ------------------------------------------------------------------
    # Model 1: GTC only
    # ------------------------------------------------------------------
    spec1  = {'intercept': dest_alts, 'gtc': 'all_same'}
    names1 = {'intercept': ['ASC_A','ASC_B'], 'gtc': 'β_GTC'}
    mnl1, res1 = run_mnl(model_df, spec1, names1, 'Model 1 — GTC only')

    # ------------------------------------------------------------------
    # Model 2: + Scale dummies + Latent Variables  (FINAL MODEL in V1)
    # WFH interaction terms removed from utility vs original V3 Model 3.
    # ------------------------------------------------------------------
    spec2 = {
        **spec1,
        'D_med':             'all_same',
        'D_lrg':             'all_same',
        'LV_Pleasantness_z': [0],
        'LV_Vibrancy_z':     [0],
        'LV_Walkability_z':  [0],
        'LV_Experiential_z': [0],
    }
    names2 = {
        **names1,
        'D_med':             'β_S_med',
        'D_lrg':             'β_S_lrg',
        'LV_Pleasantness_z': ['γ_Pleasantness'],
        'LV_Vibrancy_z':     ['γ_Vibrancy'],
        'LV_Walkability_z':  ['γ_Walkability'],
        'LV_Experiential_z': ['γ_Experiential'],
    }
    mnl2, res2 = run_mnl(model_df, spec2, names2,
                          'Model 2 — GTC + Scale + LVs  [FINAL — no WFH in utility]')

    # ------------------------------------------------------------------
    # Likelihood Ratio Test: Model 2 vs Model 1
    # ------------------------------------------------------------------
    def lr_test(ll_r, ll_u, df_diff, label):
        lr_stat = 2 * (ll_u - ll_r)
        p_val   = scipy_stats.chi2.sf(lr_stat, df_diff)
        sig     = '*** significant' if p_val < 0.05 else 'not significant'
        print(f"\n  LR test — {label}")
        print(f"    LR stat  : {lr_stat:.4f}")
        print(f"    df       : {df_diff}")
        print(f"    p-value  : {p_val:.4f}  {sig}")
        return {'Test': label, 'LR stat': round(lr_stat,4),
                'df': df_diff, 'p-value': round(p_val,4),
                'Significant': 'Yes' if p_val < 0.05 else 'No'}

    print(f"\n{'='*60}")
    print("  Likelihood Ratio Test")
    print(f"{'='*60}")

    n_p1 = count_params(names1)
    n_p2 = count_params(names2)

    lr_rows = [
        lr_test(mnl1._ll_final, mnl2._ll_final, n_p2 - n_p1,
                'Model 2 vs Model 1 (Scale + LVs added)'),
    ]
    lr_df = pd.DataFrame(lr_rows)

    print(f"\n{'='*60}")
    print("  Model Comparison Summary")
    print(f"{'='*60}")

    comparison_rows = []
    for lbl, mnl_obj, names_dict in [
        ('Model 1 — GTC only',                  mnl1, names1),
        ('Model 2 — GTC + Scale + LVs (final)', mnl2, names2),
    ]:
        comparison_rows.append({
            'Model':        lbl,
            'N Parameters': count_params(names_dict),
            'LL(final)':    round(mnl_obj._ll_final, 4),
            'LL(null)':     round(mnl_obj._ll_null,  4),
            'McFadden ρ²':  round(mnl_obj._rho2,     4),
            'AIC':          round(mnl_obj._aic,       2),
            'BIC':          round(mnl_obj._bic,       2),
        })

    comparison_df = pd.DataFrame(comparison_rows)
    print(comparison_df.to_string(index=False))

    results_df = pd.concat([res1, res2], ignore_index=True)

except ImportError:
    print("pylogit not installed. Run: pip install pylogit")
    results_df = comparison_df = lr_df = pd.DataFrame()


# =============================================================================
# SECTION 9: LV STRUCTURAL EQUATIONS (FULL SPECIFICATION — UNCHANGED)
# =============================================================================
# Full specification retained:
#   LV_nk = α_k0 + α_k^T·Z_n + β_k^T·PerSen_n + η_k^T·(PerSen_n×WFH_n) + ν_nk
#
# Note: The full spec is kept here as the confirmatory baseline.
# Significant paths discovered here motivate the V2 sparse specification.
# Multiple-testing context: 4 LVs × 16 predictors = 64 tests.
# Reported significance should be interpreted with appropriate caution.
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 9: LV Structural Equations (Full Specification)")
print("=" * 70)

zn_z_cols  = [f'{c}_z' for c in zn_cols]
ps_z_cols  = [f'{c}_z' for c in ps_cols]

# Build PerSen × WFH interaction terms using binary WFH indicator
for ps_z in ps_z_cols:
    df[f'{ps_z}_x_wfh'] = df[ps_z] * df['is_wfh']

ps_wfh_cols = [f'{c}_x_wfh' for c in ps_z_cols]

structural_results = []

print("\nEstimating full structural equations:")
print("  LV_nk = α_k0 + α_k^T·Z_n + β_k^T·PerSen_n + η_k^T·(PerSen_n×WFH_n) + ν_nk")
print(f"\n  Note: {len(lv_names)} LVs × {len(zn_z_cols)+len(ps_z_cols)+len(ps_wfh_cols)} "
      f"predictors = {len(lv_names)*(len(zn_z_cols)+len(ps_z_cols)+len(ps_wfh_cols))} "
      f"tests (multiple-testing caution applies)\n")

for lv in lv_names:
    lv_z           = f'{lv}_z'
    predictor_cols = zn_z_cols + ps_z_cols + ps_wfh_cols
    subset         = df[[lv_z] + predictor_cols].dropna()
    if len(subset) < 20:
        print(f"  {lv}: insufficient data (n={len(subset)}), skipping.")
        continue
    X   = sm.add_constant(subset[predictor_cols])
    y   = subset[lv_z]
    ols = sm.OLS(y, X).fit()

    for param, coef, se, tval, pval in zip(
        ols.params.index, ols.params, ols.bse, ols.tvalues, ols.pvalues
    ):
        ptype = ('Intercept'          if param == 'const'        else
                 'Z_n (Socio-dem)'    if param in zn_z_cols      else
                 'PerSen_n'           if param in ps_z_cols       else
                 'PerSen×WFH'         if param in ps_wfh_cols     else 'Other')
        structural_results.append({
            'LV': lv, 'Type': ptype, 'Predictor': param,
            'Coef': round(coef,4), 'SE': round(se,4),
            't': round(tval,3), 'p': round(pval,4),
            'Sig': ('***' if pval<0.001 else '**' if pval<0.01
                    else '*' if pval<0.05 else '.' if pval<0.10 else ''),
        })

    print(f"  {lv}: R²={ols.rsquared:.3f} | Adj R²={ols.rsquared_adj:.3f} "
          f"| F p={ols.f_pvalue:.4f}")

structural_df = pd.DataFrame(structural_results)
print("\nTable 6 – Full LV Structural Equation Coefficients")
print(structural_df.to_string(index=False))


# =============================================================================
# SECTION 10: EXPORT
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 10: Export")
print("=" * 70)

excel_path = os.path.join(OUTPUT_DIR, 'iclv_v1_no_wfh_utility.xlsx')

with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    reliability_df.to_excel(writer, sheet_name='T1_Reliability',         index=False)
    validity_df.to_excel(   writer, sheet_name='T2_Convergent_Validity',  index=False)
    disc_matrix.to_excel(   writer, sheet_name='T3_Discriminant_Validity')
    loadings_df.to_excel(   writer, sheet_name='T4_Factor_Loadings',      index=False)
    estimation_df.to_excel( writer, sheet_name='SP_Standard_Tasks',       index=False)  # now includes gtc
    long_df.to_excel(       writer, sheet_name='SP_All_Tasks',            index=False)
    vc_df.to_excel(         writer, sheet_name='QC_Extreme_VC',           index=False)
    retest_df.to_excel(     writer, sheet_name='QC_Test_Retest',          index=False)
    structural_df.to_excel( writer, sheet_name='T6_Structural_Full',      index=False)
    image_long_df.to_excel( writer, sheet_name='S5_Image_Ratings',        index=False)
    scale_dist.to_excel(    writer, sheet_name='S7_Scale_Distribution',   index=False)
    if not results_df.empty:
        results_df.to_excel(    writer, sheet_name='T5_MNL_Params',       index=False)
        comparison_df.to_excel( writer, sheet_name='T5_Model_Comparison', index=False)
        lr_df.to_excel(         writer, sheet_name='T5_LR_Tests',         index=False)

print(f"Excel : {excel_path}")
print("Done.")
