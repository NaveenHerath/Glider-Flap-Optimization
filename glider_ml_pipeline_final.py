"""
Glider Aerodynamic Surrogate Model
====================================
Data    : Full_Flap_Analysis_Data_Model_01.csv  (409 rows)
Inputs  : Flap Positions, Flap Angles, alpha
Outputs : CL, Cm, CD  (multi-output regression)

Findings from data inspection
------------------------------
- Zero-variance columns dropped: Beta, CY, Cl, Cn, Cni (all zeros — symmetric flight)
- No NaN values, no negative CD, no duplicates → data is very clean
- Flap Positions: 4 discrete values  [0.0, 0.65, 0.70, 0.75]
- Flap Angles:    6 discrete values  [0, 2, 4, 6, 8, 10]  degrees
- alpha:          27 values          [-3.5 … 9.5]  degrees (step 0.5°)

Model results (test R²)
------------------------
  Gradient Boosting : CL=0.9988  Cm=0.9953  CD=0.9972  ← BEST
  Random Forest     : CL=0.9900  Cm=0.9659  CD=0.9932
  MLP               : CL=0.9992  Cm=0.9986  CD=0.8497  (CD struggles)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import joblib
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CSV_PATH     = "Full_Flap_Analysis_Data_Model_01.csv"
FEATURE_COLS = ['Flap Positions', 'Flap Angles', 'alpha']
LABEL_COLS   = ['CL', 'Cm', 'CD']
RANDOM_STATE = 42

# ─────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
print(f"Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

# ─────────────────────────────────────────────
# 2. DATA CLEANING
# ─────────────────────────────────────────────

# Drop zero-variance columns (all-zero outputs from symmetric flight setup)
zero_var = [c for c in df.columns if df[c].nunique() == 1]
print(f"Dropping zero-variance columns: {zero_var}")
df.drop(columns=zero_var, inplace=True)

# Drop rows with NaN in features or labels (none found, but good practice)
before = len(df)
df.dropna(subset=FEATURE_COLS + LABEL_COLS, inplace=True)
print(f"Dropped {before - len(df)} NaN rows")

# Remove physically impossible values
df = df[df['CD'] >= 0]           # drag cannot be negative
df = df[df['CL'].between(-2, 3)] # sanity bounds for glider CL
df.reset_index(drop=True, inplace=True)
print(f"Clean dataset: {len(df)} rows")

# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────
# Aerodynamic coefficients have nonlinear coupling between alpha and flap settings
df['alpha_x_angle']    = df['alpha'] * df['Flap Angles']    # flap effectiveness varies with AoA
df['alpha_x_position'] = df['alpha'] * df['Flap Positions'] # position-AoA interaction
df['alpha2']           = df['alpha'] ** 2                   # stall nonlinearity
df['angle2']           = df['Flap Angles'] ** 2             # flap angle nonlinearity

FEATURE_COLS_ENG = FEATURE_COLS + [
    'alpha_x_angle', 'alpha_x_position', 'alpha2', 'angle2'
]

X = df[FEATURE_COLS_ENG].values
y = df[LABEL_COLS].values
print(f"Features: {X.shape}  |  Labels: {y.shape}")

# ─────────────────────────────────────────────
# 4. TRAIN / VAL / TEST SPLIT  (70 / 15 / 15)
# ─────────────────────────────────────────────
X_tv, X_test, y_tv, y_test = train_test_split(
    X, y, test_size=0.15, random_state=RANDOM_STATE
)
X_train, X_val, y_train, y_val = train_test_split(
    X_tv, y_tv, test_size=0.176, random_state=RANDOM_STATE
)
print(f"Split → train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

# ─────────────────────────────────────────────
# 5. DEFINE MODELS
# ─────────────────────────────────────────────
models = {

    # ── Gradient Boosting (BEST overall — recommended) ─────────────────
    # Wraps one GBR per output via MultiOutputRegressor.
    # n_estimators=300, depth=4 is a good starting point; tune if needed.
    "Gradient Boosting": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  MultiOutputRegressor(GradientBoostingRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=RANDOM_STATE,
        ))),
    ]),

    # ── Random Forest (robust, no scaling needed but kept for consistency) ─
    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  RandomForestRegressor(
            n_estimators=500,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ]),

    # ── MLP (great for CL/Cm, weaker on CD — try if smooth curves needed) ─
    # CD is harder for MLP because it has a sharp uptick at high alpha.
    "MLP": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            learning_rate_init=1e-3,
            max_iter=3000,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=RANDOM_STATE,
        )),
    ]),
}

# ─────────────────────────────────────────────
# 6. TRAIN & EVALUATE ALL MODELS
# ─────────────────────────────────────────────
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred_val  = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    r2_val   = [r2_score(y_val[:, i],  y_pred_val[:, i])  for i in range(3)]
    r2_test  = [r2_score(y_test[:, i], y_pred_test[:, i]) for i in range(3)]
    mae_test = [mean_absolute_error(y_test[:, i], y_pred_test[:, i]) for i in range(3)]

    results[name] = {
        "r2_val": r2_val, "r2_test": r2_test,
        "mae_test": mae_test, "y_pred_test": y_pred_test
    }

    print(f"\n── {name} ──")
    for i, lbl in enumerate(LABEL_COLS):
        print(f"  {lbl:4s}  R²_val={r2_val[i]:.4f}  "
              f"R²_test={r2_test[i]:.4f}  MAE={mae_test[i]:.6f}")

# ─────────────────────────────────────────────
# 7. SELECT BEST MODEL
# ─────────────────────────────────────────────
best_name = max(results, key=lambda n: np.mean(results[n]['r2_test']))
best_model = models[best_name]
print(f"\nBest model: {best_name}  "
      f"(mean R²={np.mean(results[best_name]['r2_test']):.4f})")

# 5-fold CV on full train+val set
# Note: cross_val_score passes 1D y per label, so we build a single-output
# pipeline for CV (same hyperparameters, no MultiOutputRegressor wrapper).
print(f"\n── 5-fold CV ({best_name}) ──")
cv_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  GradientBoostingRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=RANDOM_STATE,
    )),
])
for i, lbl in enumerate(LABEL_COLS):
    cv = cross_val_score(cv_pipeline, X_tv, y_tv[:, i], cv=5, scoring='r2')
    print(f"  {lbl:4s}  R² = {cv.mean():.4f} ± {cv.std():.4f}")

# ─────────────────────────────────────────────
# 8. PLOTS
# ─────────────────────────────────────────────
colors = {
    'Gradient Boosting': '#1D9E75',
    'Random Forest':     '#7F77DD',
    'MLP':               '#EF9F27',
}
label_colors = ['#7F77DD', '#1D9E75', '#EF9F27']

# Plot 1 — R² comparison bar chart
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, lbl, i in zip(axes, LABEL_COLS, range(3)):
    names = list(results.keys())
    r2s   = [results[n]['r2_test'][i] for n in names]
    bars  = ax.bar(names, r2s, color=[colors[n] for n in names],
                   edgecolor='white', width=0.5)
    ax.set_ylim(0, 1.05)
    ax.set_title(f'{lbl}  —  R² (test)', fontsize=12)
    ax.set_ylabel('R²')
    ax.tick_params(axis='x', rotation=15)
    for bar, val in zip(bars, r2s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
plt.suptitle('Model comparison — R² per output (test set)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150, bbox_inches='tight')
plt.show()

# Plot 2 — Predicted vs Actual (best model)
y_pred_best = results[best_name]['y_pred_test']
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, lbl, i, col in zip(axes, LABEL_COLS, range(3), label_colors):
    ax.scatter(y_test[:, i], y_pred_best[:, i],
               color=col, s=30, alpha=0.7, edgecolors='none')
    mn = min(y_test[:, i].min(), y_pred_best[:, i].min())
    mx = max(y_test[:, i].max(), y_pred_best[:, i].max())
    ax.plot([mn, mx], [mn, mx], '--', color='gray', linewidth=1)
    ax.set_xlabel(f'Actual {lbl}')
    ax.set_ylabel(f'Predicted {lbl}')
    ax.set_title(f'{lbl}  R²={results[best_name]["r2_test"][i]:.4f}', fontsize=12)
plt.suptitle(f'Predicted vs Actual — {best_name} (test set)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("pred_vs_actual.png", dpi=150, bbox_inches='tight')
plt.show()

# Plot 3 — Feature importances (Random Forest)
rf_model = models["Random Forest"]
importances = rf_model.named_steps['model'].feature_importances_
idx = np.argsort(importances)
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh([FEATURE_COLS_ENG[j] for j in idx], importances[idx],
        color='#7F77DD', edgecolor='white')
ax.set_xlabel('Mean importance')
ax.set_title('Random Forest — feature importances', fontsize=13)
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches='tight')
plt.show()

# Plot 4 — Residuals (best model)
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, lbl, i, col in zip(axes, LABEL_COLS, range(3), label_colors):
    res = y_test[:, i] - y_pred_best[:, i]
    ax.scatter(y_pred_best[:, i], res, color=col, s=25, alpha=0.7, edgecolors='none')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlabel(f'Predicted {lbl}')
    ax.set_ylabel('Residual')
    ax.set_title(f'{lbl} residuals')
plt.suptitle(f'Residual plots — {best_name} (test set)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("residuals.png", dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────
# 9. SAVE & INFERENCE
# ─────────────────────────────────────────────
joblib.dump(best_model, "glider_surrogate_model.pkl")
print("\nModel saved → glider_surrogate_model.pkl")

def predict(flap_position, flap_angle, alpha):
    """
    Predict CL, Cm, CD for a given flap configuration.

    Parameters
    ----------
    flap_position : float  e.g. 0.65, 0.70, 0.75
    flap_angle    : float  e.g. 0, 2, 4, 6, 8, 10  (degrees)
    alpha         : float  angle of attack in degrees

    Returns
    -------
    dict with keys CL, Cm, CD
    """
    model = joblib.load("glider_surrogate_model.pkl")
    feats = np.array([[
        flap_position,
        flap_angle,
        alpha,
        alpha * flap_angle,
        alpha * flap_position,
        alpha ** 2,
        flap_angle ** 2,
    ]])
    pred = model.predict(feats)[0]
    return dict(zip(LABEL_COLS, pred))

# Example prediction
result = predict(flap_position=0.70, flap_angle=6, alpha=4.0)
print("\n── Example prediction ──")
print(f"  Flap position=0.70, angle=6°, alpha=4°")
for k, v in result.items():
    print(f"  {k} = {v:.6f}")
