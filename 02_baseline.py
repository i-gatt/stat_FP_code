from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split, StratifiedKFold,
    GridSearchCV, cross_val_predict
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve

# -------------------------------------------------
# 1. LOAD DATA
# -------------------------------------------------

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

df = pd.read_csv(ROOT / "data" / "adult.csv")

y = df["income_gt_50k"].astype(int)
X = df.drop(columns=["income_gt_50k"])

# -------------------------------------------------
# 2. STRATIFIED 60/20/20 SPLIT
# -------------------------------------------------

X_train, X_other, y_train, y_other = train_test_split(
    X, y,
    train_size=0.60,
    stratify=y,
    random_state=42
)

X_cal, X_test, y_cal, y_test = train_test_split(
    X_other, y_other,
    test_size=0.50,
    stratify=y_other,
    random_state=42
)

# The test set is reserved for final evaluation.
# We do not use X_test or y_test below.

print("Training observations:", len(X_train))
print("Calibration observations:", len(X_cal))
print("Test observations:", len(X_test))

# -------------------------------------------------
# 3. PREPROCESSING
# -------------------------------------------------

numeric_cols = X.select_dtypes(
    include="number"
).columns.tolist()

categorical_cols = X.select_dtypes(
    exclude="number"
).columns.tolist()

preprocess = ColumnTransformer([
    ("numeric", Pipeline([
        ("fill", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ]), numeric_cols),

    ("categorical", Pipeline([
        ("fill", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_cols)
])

# -------------------------------------------------
# 4. LOGISTIC REGRESSION BASELINE
# -------------------------------------------------

model = Pipeline([
    ("preprocess", preprocess),
    ("logistic", LogisticRegression(
        max_iter=3000,
        solver="lbfgs"
    ))
])

folds = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

c_values = [0.001, 0.01, 0.1, 1, 10, 100]

search = GridSearchCV(
    model,
    param_grid={"logistic__C": c_values},
    scoring="neg_log_loss",
    cv=folds,
    refit=True,
    n_jobs=1
)

search.fit(X_train, y_train)

chosen_C = search.best_params_["logistic__C"]

# NLL for each of the five validation folds
fold_nll = -np.array([
    search.cv_results_[f"split{i}_test_score"][
        search.best_index_
    ]
    for i in range(5)
])

mean_nll = fold_nll.mean()

standard_error = fold_nll.std(ddof=1) / np.sqrt(5)

print("\nBASELINE RESULTS")
print("Selected C:", chosen_C)
print("Fold NLL values:", fold_nll)
print("Mean CV NLL:", mean_nll)
print("Standard error:", standard_error)

# -------------------------------------------------
# 5. OUT-OF-FOLD RELIABILITY DIAGRAM
# -------------------------------------------------

oof_prob = cross_val_predict(
    search.best_estimator_,
    X_train,
    y_train,
    cv=folds,
    method="predict_proba"
)[:, 1]

observed, predicted = calibration_curve(
    y_train,
    oof_prob,
    n_bins=10,
    strategy="quantile"
)

plt.figure(figsize=(6, 5))

plt.plot(
    [0, 1], [0, 1], "--",
    label="Perfect calibration"
)

plt.plot(
    predicted, observed, "o-",
    label="Logistic regression"
)

plt.xlabel("Mean predicted probability")
plt.ylabel("Observed fraction with income >$50K")
plt.title("Out-of-fold Reliability Diagram: Adult")
plt.legend()
plt.tight_layout()

plt.savefig(
    RESULTS / "reliability_diagram.png",
    dpi=180
)

plt.close()

# -------------------------------------------------
# 6. FIT CALIBRATION METHODS
# -------------------------------------------------

# Predictions on the calibration split
raw = search.best_estimator_.predict_proba(X_cal)[:, 1]

# Platt scaling
scores = search.best_estimator_.decision_function(
    X_cal
).reshape(-1, 1)

platt = LogisticRegression(
    C=1e6,
    max_iter=3000
)

platt.fit(scores, y_cal)

platt_prob = platt.predict_proba(scores)[:, 1]

# Isotonic regression
isotonic = IsotonicRegression(
    out_of_bounds="clip"
)

isotonic.fit(raw, y_cal)

isotonic_prob = isotonic.predict(raw)

# -------------------------------------------------
# 7. CLASSIFICATION COST COMPARISON
# -------------------------------------------------

def average_cost(y_true, probs, cost_fp, cost_fn):
    threshold = cost_fp / (cost_fp + cost_fn)

    decisions = probs >= threshold

    false_pos = np.sum(
        (decisions == 1) & (y_true == 0)
    )

    false_neg = np.sum(
        (decisions == 0) & (y_true == 1)
    )

    return (
        cost_fp * false_pos + cost_fn * false_neg
    ) / len(y_true)


cost_rows = []

for fp, fn in [(1, 1), (1, 5), (5, 1)]:

    for name, probs in [
        ("Raw", raw),
        ("Platt", platt_prob),
        ("Isotonic", isotonic_prob)
    ]:

        cost_rows.append({
            "FP:FN ratio": f"{fp}:{fn}",
            "method": name,
            "threshold": round(fp / (fp + fn), 4),
            "mean cost": round(
                average_cost(y_cal, probs, fp, fn), 6
            )
        })

cost_table = pd.DataFrame(cost_rows)

print("\nPRELIMINARY CALIBRATION COSTS")
print(cost_table.to_string(index=False))

cost_table.to_csv(
    RESULTS / "calibration_costs.csv",
    index=False
)

# -------------------------------------------------
# 8. SAVE BASELINE RESULTS
# -------------------------------------------------

summary = (
    f"N: {len(df)}; inputs: {X.shape[1]}\n"
    f"Train: {len(X_train)}; "
    f"calibration: {len(X_cal)}; "
    f"test untouched: {len(X_test)}\n"
    f"Random seed: 42; stratified 5-fold CV\n"
    f"C grid: {c_values}\n"
    f"Selected C: {chosen_C}\n"
    f"Fold NLLs: {np.round(fold_nll, 6).tolist()}\n"
    f"Mean CV NLL: {mean_nll:.6f}\n"
    f"CV standard error: {standard_error:.6f}\n"
    "Calibration costs are preliminary in-sample results.\n"
)

(RESULTS / "baseline_results.txt").write_text(
    summary + "\n" + cost_table.to_string(index=False)
)

print("\nSaved outputs inside the results folder.")