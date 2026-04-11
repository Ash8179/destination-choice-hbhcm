"""
C.2.6.3 – Features Extraction Modification (Deprecated)

This script:
Uses log(1 + n_yolo_detections) to calculate percept_activity_diversity instead of entropy, which is deprecated for collinearity found later and the poor theoretical basis.

Author: Zhang Wenyu
Date: 2026-04-01
"""

import pandas as pd
import numpy as np

# Load CSV
file_path = "/Users/zhangwenyu/Desktop/NUSFYP/Stage 1/1.3 Street Level Imagery/perceptual_features_with_dimensions.csv"
df = pd.read_csv(file_path)

# Step 1: Log transform -> log(1 + n_yolo_detections)
df["log_n_yolo"] = np.log1p(df["n_yolo_detections"])

# Step 2: Min-Max normalization (0–1 scaling)
min_val = df["log_n_yolo"].min()
max_val = df["log_n_yolo"].max()

df["activity_diversity_new"] = (df["log_n_yolo"] - min_val) / (max_val - min_val)

# Step 3: Replace original variable
df["percept_activity_diversity"] = df["activity_diversity_new"]

# Step 4: Save updated file
output_path = "/Users/zhangwenyu/Desktop/NUSFYP/Stage 1/1.3 Street Level Imagery/perceptual_features_updated.csv"
df.to_csv(output_path, index=False)

print("Processing complete. Saved to:", output_path)
