"""
B.7 – PCA Car Proxy Generation

This script reads car proxy outputs (Car Proxy 1 and Car Proxy 2), standardizes the data,
applies PCA to generate a single composite proxy (PCA_Car_Proxy), prints component weights
and explained variance, and saves the results to a new CSV.

Author: Zhang Wenyu
Date: 2025-12-11
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

# -------------------------------------------------------------
# Paths
# -------------------------------------------------------------
input_path = "/Users/zhangwenyu/Desktop/NUSFYP/car_proxy_outputs.csv"
output_dir = "/Users/zhangwenyu/Desktop/NUSFYP/"
output_file = "car_proxy_pca.csv"
output_path = os.path.join(output_dir, output_file)

# -------------------------------------------------------------
# Check if directory exists
# -------------------------------------------------------------
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Directory {output_dir} created.")

# -------------------------------------------------------------
# Load existing car proxy data
# -------------------------------------------------------------
df = pd.read_csv(input_path)

# Ensure the required columns exist
required_columns = ["Car Proxy 1 (Ownership%)", "Car Proxy 2 (Economic Parameter)"]
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in input CSV.")

# -------------------------------------------------------------
# Standardize the two variables
# -------------------------------------------------------------
X = df[required_columns].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------------------------------------
# Apply PCA (1 component)
# -------------------------------------------------------------
pca = PCA(n_components=1)
pca_scores = pca.fit_transform(X_scaled)  # shape (n_samples, 1)
weights = pca.components_[0]              # PCA weights for first component
explained_var_ratio = pca.explained_variance_ratio_[0]

# Add PCA_Car_Proxy to dataframe
df["PCA_Car_Proxy"] = pca_scores.flatten()

# -------------------------------------------------------------
# Output CSV
# -------------------------------------------------------------
df.to_csv(output_path, index=False)
print(f"PCA output saved to: {output_path}")

# -------------------------------------------------------------
# Print PCA weights and explained variance
# -------------------------------------------------------------
print("\nPCA Weights (First Principal Component):")
print(f"Car Proxy 1 (Ownership%): {weights[0]:.4f}")
print(f"Car Proxy 2 (Economic Parameter): {weights[1]:.4f}")
print(f"\nExplained variance by PCA_Car_Proxy: {explained_var_ratio:.4f}")
