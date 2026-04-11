"""
C.2.10.1 – Dimension Correlation Heatmap

This script:
Compute and visualize the correlation matrix of five perceptual
dimensions using a heatmap.

Author: Zhang Wenyu
Date: 2026-03-12
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

INPUT_FILE = "/Users/zhangwenyu/Desktop/NUSFYP/Stage 1/1.3 Street Level Imagery/pipeline_output/D_updated.xlsx"

df = pd.read_excel(INPUT_FILE)

dimensions = ["vibrancy", "pleasantness", "walkability", "safety", "experiential"]
df_subset = df[dimensions]

corr_matrix = df_subset.corr()

print("Correlation matrix:")
print(corr_matrix)

plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix of Street-Level Perception Dimensions")
plt.show()
