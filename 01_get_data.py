from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

# Create a folder to store the data
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# Download the Adult dataset
adult = fetch_openml(name="adult", version=2, as_frame=True)

# Separate the predictors and outcome
X = adult.data.copy()
y = adult.target.astype(str).str.strip().str.rstrip(".")

# Replace missing-value indicators with NaN
for col in X.select_dtypes(include=["object", "category"]).columns:
    X[col] = X[col].astype("object").replace("?", np.nan)

# Encode income as a binary outcome:
# 1 = annual income > $50K
# 0 = annual income <= $50K
assert set(y.unique()) == {"<=50K", ">50K"}, y.unique()
X["income_gt_50k"] = (y == ">50K").astype(int)

# Save the dataset as a CSV
file_path = DATA_DIR / "adult.csv"
X.to_csv(file_path, index=False)

# Display basic dataset information
print("Dataset saved to:", file_path)
print("Observations:", len(X))
print("Inputs:", X.shape[1] - 1)
print("Output: income_gt_50k (1 means income > $50K)")
print(
    "Rows with missing inputs:",
    X.drop(columns="income_gt_50k").isna().any(axis=1).sum()
)
print("Rows dropped: 0")