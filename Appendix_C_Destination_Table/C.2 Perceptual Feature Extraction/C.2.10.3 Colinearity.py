"""
C.2.10.3 – Category Dependence Diagnostic (OLS)

This script:
Assess dependence between perceptual dimensions and POI categories
using OLS regressions (CV ~ category) and report R² as a collinearity indicator.

Author: Zhang Wenyu
Date: 2026-03-13
"""

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# --------------------------------------------------
# 1. Load data
# --------------------------------------------------
path = "/Users/zhangwenyu/Desktop/FYP Pipeline/Appendix_A_Spatial_Framework/D.xlsx"
df = pd.read_excel(path)

# --------------------------------------------------
# 2. Define category (use what you have)
# --------------------------------------------------
df["category"] = df["subtype"]
df["category"] = df["category"].astype("category")

# --------------------------------------------------
# 3. CV variables to test
# --------------------------------------------------
cv_vars = [
    "vibrancy",
    "pleasantness",
    "walkability",
    "safety",
    "experiential"
]

# --------------------------------------------------
# 4. Run regression: CV ~ category
# --------------------------------------------------
print("="*60)
print("Collinearity Diagnostic: CV ~ Category")
print("="*60)

results = []

for var in cv_vars:
    sub = df[[var, "category"]].dropna()
    
    # OLS: var ~ category
    model = smf.ols(f"{var} ~ C(category)", data=sub).fit()
    
    r2 = model.rsquared
    n = len(sub)
    
    results.append((var, r2, n))
    
    print(f"{var:15s} | R² = {r2:.3f} | N = {n}")

# --------------------------------------------------
# 5. Simple interpretation
# --------------------------------------------------
print("\nInterpretation Guide:")
print("R² < 0.2  → weak dependence")
print("0.2–0.5   → moderate")
print("> 0.5     → strong collinearity with category")
