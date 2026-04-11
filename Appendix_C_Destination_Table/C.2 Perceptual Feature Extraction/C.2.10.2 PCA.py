"""
C.2.10.2 – PCA on Perceptual Dimensions

This script:
Apply PCA on five standardized perceptual dimensions to evaluate
variance structure and extract component loadings.

Author: Zhang Wenyu
Date: 2026-03-13
"""

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

file = "/Users/zhangwenyu/Desktop/NUSFYP/Stage 1/1.3 Street Level Imagery/pipeline_output/D_updated.xlsx"
df = pd.read_excel(file)

cols = ["vibrancy","pleasantness","walkability","safety","experiential"]

X = df[cols].dropna()

X_scaled = StandardScaler().fit_transform(X)

pca = PCA()
pca.fit(X_scaled)

print("Explained variance ratio:")
print(pca.explained_variance_ratio_)

loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f"PC{i+1}" for i in range(len(cols))],
    index=cols
)

print("\nLoadings:")
print(loadings)
