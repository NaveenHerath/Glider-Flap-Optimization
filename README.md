# ✈ Glider Flap Optimizer — ML Surrogate Model

[![HF Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-blue)](https://huggingface.co/spaces/YOUR_USERNAME/glider-flap-optimizer)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5%2B-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-app-red?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

> **Physics-informed ML surrogate model** that replaces CFD simulation calls in a glider flap design loop — predicting CL, Cm, CD in milliseconds and finding the optimal flap configuration for maximum glide ratio under full pitch-stability constraints.

---

## The Problem

Designing the optimal flap configuration for a fixed-wing glider normally means running a separate CFD simulation for every candidate design — each one taking significant time. With 24 flap configurations × a continuous angle-of-attack range, exhaustive simulation-based search is impractical.

**The solution:** train a surrogate ML model on a fixed set of 409 simulation runs, then use that model to predict aerodynamic coefficients for any configuration in milliseconds. The optimizer can then sweep thousands of candidates instantly.

---

## What Was Built

### Phase 1 — Surrogate Model
A **Gradient Boosting** regression model trained on XFLR5 simulation data. Takes three inputs and predicts three aerodynamic coefficients simultaneously:

| Input | Description | Range |
|---|---|---|
| Flap Position | Chord-wise hinge location | 0.0, 0.65, 0.70, 0.75 |
| Flap Angle | Deflection angle (degrees) | 0°, 2°, 4°, 6°, 8°, 10° |
| α (alpha) | Angle of attack (degrees) | −3.5° to 9.5° |

| Output | Description | Test R² |
|---|---|---|
| **CL** | Lift coefficient | **0.9988** |
| **Cm** | Pitching moment coefficient | **0.9953** |
| **CD** | Drag coefficient | **0.9972** |

### Phase 2 — Constrained Optimizer
Exhaustive search across 3,144 configurations (24 discrete flap combos × 131 alpha values) enforcing three pitch-stability constraints simultaneously:

| Constraint | Meaning |
|---|---|
| Cm < 0 | Trim condition — nose-down restoring moment at operating point |
| dCm/dα < 0 | Pitch stiffness — aircraft returns to trim after a disturbance |
| Static Margin > 0 | CG is ahead of neutral point — inherently stable |

**Objective: maximise CL/CD (glide ratio)**

---

## Results

### Model Performance

![Model Comparison](outputs/model_comparison.png)

Three models were compared — Gradient Boosting won across all three outputs:

| Model | CL R² | Cm R² | CD R² |
|---|---|---|---|
| **Gradient Boosting** | **0.9988** | **0.9953** | **0.9972** |
| Random Forest | 0.9900 | 0.9659 | 0.9932 |
| MLP Neural Net | 0.9992 | 0.9986 | 0.8497 |

MLP struggles on CD due to its sharp asymmetric curve shape near stall — Gradient Boosting's piecewise structure handles this better.

### Predicted vs Actual

![Pred vs Actual](outputs/pred_vs_actual.png)

### Optimal Flap Configuration

![Optimal Config](outputs/optimal_config.png)

| Parameter | Value |
|---|---|
| Flap Position | **0.65** |
| Flap Angle | **2°** |
| Trim Alpha | **1.4°** |
| CL | 0.5004 |
| CD | 0.01971 |
| **CL/CD (glide ratio)** | **25.39** |
| Cm | −0.008 ✓ |
| dCm/dα | −0.0045 ✓ |
| Static Margin | 0.136 (13.6% MAC) ✓ |

### Stability Heatmap

![Stability Heatmap](outputs/stability_heatmap.png)

The heatmap shows best stable CL/CD for every (Flap Position, Flap Angle) pair. Small flap angles (0–2°) consistently outperform larger deflections — higher angles increase CL but push Cm positive, violating the trim constraint.

### Feature Importance

![Feature Importance](outputs/feature_importance.png)

Alpha dominates importance, followed by the alpha² interaction term — confirming that the nonlinear stall behaviour drives most of the model's learning task.

---

## Project Structure

```
glider-flap-optimizer/
│
├── app.py                          # Streamlit web application (live demo)
├── glider_ml_pipeline_final.py     # Full ML training pipeline
├── glider_optimizer.py             # Standalone optimization script
├── requirements.txt                # Python dependencies
│
├── data/
│   └── Full_Flap_Analysis_Data_Model_01.csv   # 409 CFD simulation runs
│
├── model/
│   └── glider_surrogate_model.pkl  # Trained Gradient Boosting model
│
├── outputs/
│   ├── model_comparison.png        # R² bar chart — 3 models × 3 outputs
│   ├── pred_vs_actual.png          # Scatter plots — predicted vs actual
│   ├── feature_importance.png      # Random Forest feature importances
│   ├── residuals.png               # Residual plots — test set
│   ├── stability_heatmap.png       # Best stable CL/CD per config
│   ├── optimal_config.png          # Drag polar + summary for best config
│   ├── clcd_vs_alpha.png           # CL/CD vs alpha grid
│   ├── top5_analysis.png           # Cm and CL/CD curves — top 5 configs
│   └── stable_configs.csv          # Full ranked list of stable configs
│
└── README.md
```

---

## How It Works

### Data Cleaning
- Dropped 5 zero-variance columns (`Beta`, `CY`, `Cl`, `Cn`, `Cni`) — all zero due to symmetric flight (β = 0), carrying no information
- Removed physically impossible values (CD < 0 = diverged simulation run)
- IQR-based outlier removal with factor 3.0 (conservative, preserving small dataset)

### Feature Engineering
Raw inputs augmented with interaction and polynomial terms that capture aerodynamic coupling:

```python
alpha × Flap_Angle     # flap effectiveness changes with AoA
alpha × Flap_Position  # hinge position interacts with AoA
alpha²                 # stall nonlinearity — parabolic drag polar
Flap_Angle²            # diminishing returns in flap effectiveness
```

Without these terms, tree models need much deeper splits to approximate the same nonlinear relationships — increasing overfitting risk especially with < 500 rows.

### Model
- **Algorithm:** `GradientBoostingRegressor` (scikit-learn)
- **Multi-output:** `MultiOutputRegressor` wrapper — one independent GBR per output
- **Pipeline:** `StandardScaler` → `MultiOutputRegressor(GBR)`
- **Key hyperparameters:** `n_estimators=300`, `max_depth=4`, `learning_rate=0.05`, `subsample=0.8`
- **Split:** 70% train / 15% val / 15% test + 5-fold cross-validation

### Stability Constraints (Numerical)
All three stability metrics are computed via **central finite differences** on the surrogate model:

```python
# dCm/dα — pitch stiffness
dCm_dα = (Cm(α+0.1) - Cm(α-0.1)) / 0.2

# Static Margin = -dCm/dCL
dCm = Cm(α+0.1) - Cm(α-0.1)
dCL = CL(α+0.1) - CL(α-0.1)
SM  = -dCm / dCL   # positive = CG ahead of neutral point
```

Central differences are used over forward differences because they are second-order accurate (error ∝ Δα² rather than Δα).

---

## Run Locally

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/glider-flap-optimizer.git
cd glider-flap-optimizer

# Install
pip install -r requirements.txt

# Launch app
streamlit run app.py

# Or run the optimizer standalone
python glider_optimizer.py
```

---

## Live Demo

**→ [Try the app on Hugging Face Spaces](https://huggingface.co/spaces/YOUR_USERNAME/glider-flap-optimizer)**

The app lets you:
- Input any flap configuration and instantly predict CL, Cm, CD
- See full stability analysis with pass/fail on all three constraints
- Run the global optimizer and explore the stability heatmap
- Compare multiple configurations side by side

---

## Dataset
409 XFLR5 simulation runs of a custom fixed-wing model glider (1.5m wingspan, Clark Y airfoil).
- Symmetric flight condition (β = 0°)
- 4 flap positions × 6 flap angles × 27 alpha values (−3.5° to 9.5°, step 0.5°)

---

## Tech Stack
`Python` · `scikit-learn` · `Streamlit` · `NumPy` · `pandas` · `Matplotlib` · `joblib` · `SciPy`

---

## License
MIT — free to use, modify, and distribute.
