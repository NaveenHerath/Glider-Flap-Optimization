"""
Glider Flap Optimizer
======================
Finds the optimal (Flap Position, Flap Angle, alpha) that maximises
CL/CD subject to three pitch-stability constraints:
  1. Cm < 0          (trim: nose-down restoring moment)
  2. dCm/dα < 0      (static stability: pitch stiffness)
  3. Static Margin > 0  (SM = -dCm/dCL, CG ahead of neutral point)

Usage
-----
  python glider_optimizer.py
  → prints ranked results + saves stable_configs.csv + 4 plots
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────
MODEL_PATH     = "glider_surrogate_model.pkl"
LABEL_COLS     = ['CL', 'Cm', 'CD']
FLAP_POSITIONS = [0.0, 0.65, 0.70, 0.75]   # discrete values in dataset
FLAP_ANGLES    = [0, 2, 4, 6, 8, 10]        # degrees
ALPHA_SWEEP    = np.arange(-3.5, 9.6, 0.1)  # fine-grained alpha grid
DA             = 0.1                         # finite-difference step for derivatives

# ── Helper functions ──────────────────────────────────────────────────────
def make_features(fp, fa, alpha):
    """Build engineered feature vector matching training pipeline."""
    return np.array([[fp, fa, alpha,
                      alpha * fa,
                      alpha * fp,
                      alpha ** 2,
                      fa ** 2]])

def predict(model, fp, fa, alpha):
    """Return dict {CL, Cm, CD} for a given config."""
    pred = model.predict(make_features(fp, fa, alpha))[0]
    return dict(zip(LABEL_COLS, pred))

def dcm_dalpha(model, fp, fa, alpha):
    """Numerical dCm/dα via central difference."""
    cm_hi = predict(model, fp, fa, alpha + DA)['Cm']
    cm_lo = predict(model, fp, fa, alpha - DA)['Cm']
    return (cm_hi - cm_lo) / (2 * DA)

def static_margin(model, fp, fa, alpha):
    """
    Static Margin = -dCm/dCL  (positive = stable).
    Represents how far the CG is ahead of the neutral point,
    as a fraction of mean aerodynamic chord (MAC).
    """
    p_hi = predict(model, fp, fa, alpha + DA)
    p_lo = predict(model, fp, fa, alpha - DA)
    dCm = p_hi['Cm'] - p_lo['Cm']
    dCL = p_hi['CL'] - p_lo['CL']
    return -dCm / dCL if abs(dCL) > 1e-9 else 0.0

# ── Grid search ───────────────────────────────────────────────────────────
def run_optimization(model):
    """
    Exhaustive search over all discrete (FP, FA) pairs and fine alpha grid.
    Returns full results DataFrame and stable-only DataFrame.
    """
    rows = []
    for fp in FLAP_POSITIONS:
        for fa in FLAP_ANGLES:
            for alpha in ALPHA_SWEEP:
                p  = predict(model, fp, fa, alpha)
                CL, Cm, CD = p['CL'], p['Cm'], p['CD']
                if CD < 1e-6:
                    continue

                glide  = CL / CD
                dCm_da = dcm_dalpha(model, fp, fa, alpha)
                SM     = static_margin(model, fp, fa, alpha)

                c1 = Cm < 0        # trim
                c2 = dCm_da < 0    # pitch stiffness
                c3 = SM > 0.0      # positive static margin

                rows.append({
                    'Flap Position': fp,
                    'Flap Angle':    fa,
                    'alpha':         round(float(alpha), 1),
                    'CL':            round(CL, 4),
                    'Cm':            round(Cm, 4),
                    'CD':            round(CD, 5),
                    'CL/CD':         round(glide, 3),
                    'dCm/dalpha':    round(dCm_da, 5),
                    'Static Margin': round(SM, 4),
                    'Cm<0':          c1,
                    'dCm/da<0':      c2,
                    'SM>0':          c3,
                    'All Stable':    c1 and c2 and c3,
                })

    df_all    = pd.DataFrame(rows)
    df_stable = df_all[df_all['All Stable']].sort_values(
        'CL/CD', ascending=False).reset_index(drop=True)
    return df_all, df_stable

# ── Plotting ──────────────────────────────────────────────────────────────
CLR = {0.0: '#888780', 0.65: '#7F77DD', 0.70: '#1D9E75', 0.75: '#EF9F27'}

def plot_clcd_grid(df_all, df_stable):
    """3×3 grid: CL/CD vs alpha for representative (FP, FA) combos."""
    combos = [(fp, fa) for fp in [0.65, 0.70, 0.75] for fa in [2, 6, 10]]
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    for ax, (fp, fa) in zip(axes.flat, combos):
        sub  = df_all[(df_all['Flap Position'] == fp) & (df_all['Flap Angle'] == fa)]
        stab = sub[sub['All Stable']]
        unst = sub[~sub['All Stable']]
        ax.scatter(unst['alpha'], unst['CL/CD'], c='#D3D1C7', s=12, zorder=2)
        ax.scatter(stab['alpha'], stab['CL/CD'], c=CLR[fp],   s=22, zorder=3)
        if len(stab):
            b = stab.iloc[0]
            ax.axvline(b['alpha'], color=CLR[fp], lw=0.9, ls='--')
            ax.annotate(f"α={b['alpha']}°\n{b['CL/CD']:.1f}",
                        xy=(b['alpha'], b['CL/CD']),
                        xytext=(b['alpha'] + 0.7, b['CL/CD'] - 1.5),
                        fontsize=8, color=CLR[fp])
        ax.set_title(f'FP={fp}  FA={fa}°', fontsize=10)
        ax.set_xlabel('α (°)', fontsize=8)
        ax.set_ylabel('CL/CD', fontsize=8)
        ax.grid(True, alpha=0.2)
    plt.suptitle('CL/CD vs α  |  coloured = stable, grey = unstable', fontsize=13)
    plt.tight_layout()
    plt.savefig("clcd_vs_alpha.png", dpi=150, bbox_inches='tight')
    plt.show()

def plot_heatmap(df_stable):
    """Heatmap of best stable CL/CD for each (FP, FA) pair."""
    pivot = df_stable.groupby(['Flap Position', 'Flap Angle'])['CL/CD'].max().unstack(fill_value=np.nan)
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(pivot.values, aspect='auto', cmap='RdYlGn',
                   vmin=10, vmax=28, origin='lower')
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f'{v}°' for v in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel('Flap Angle')
    ax.set_ylabel('Flap Position')
    ax.set_title('Best stable CL/CD for each (Flap Position, Flap Angle)', fontsize=12)
    plt.colorbar(im, ax=ax, label='Max stable CL/CD')
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig("stability_heatmap.png", dpi=150, bbox_inches='tight')
    plt.show()

def plot_top5(df_all, df_stable):
    """Cm vs alpha and CL/CD vs alpha for top 5 unique configs."""
    top5    = df_stable.drop_duplicates(['Flap Position', 'Flap Angle']).head(5)
    colors5 = ['#534AB7', '#0F6E56', '#BA7517', '#A32D2D', '#444441']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for (_, row), c in zip(top5.iterrows(), colors5):
        fp, fa = row['Flap Position'], row['Flap Angle']
        sub = df_all[(df_all['Flap Position'] == fp) &
                     (df_all['Flap Angle'] == fa)].sort_values('alpha')
        ax.plot(sub['alpha'], sub['Cm'], color=c, lw=2, label=f'FP={fp} FA={fa}°')
    ax.axhline(0, color='gray', lw=0.8, ls='--')
    ax.set_xlabel('α (°)'); ax.set_ylabel('Cm')
    ax.set_title('Cm vs α — top 5 configs')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.2)

    ax = axes[1]
    for (_, row), c in zip(top5.iterrows(), colors5):
        fp, fa = row['Flap Position'], row['Flap Angle']
        sub = df_all[(df_all['Flap Position'] == fp) &
                     (df_all['Flap Angle'] == fa)].sort_values('alpha')
        ax.plot(sub['alpha'], sub['CL/CD'], color=c, lw=2, label=f'FP={fp} FA={fa}°')
        best_a = df_stable[(df_stable['Flap Position'] == fp) &
                           (df_stable['Flap Angle'] == fa)].iloc[0]['alpha']
        ax.axvline(best_a, color=c, lw=0.9, ls=':')
    ax.set_xlabel('α (°)'); ax.set_ylabel('CL/CD')
    ax.set_title('CL/CD vs α — top 5 (dotted = optimal trim α)')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.2)
    plt.suptitle('Top 5 stable configurations', fontsize=13)
    plt.tight_layout()
    plt.savefig("top5_analysis.png", dpi=150, bbox_inches='tight')
    plt.show()

def plot_optimal(df_all, df_stable, best):
    """Drag polar and summary bar for the single best config."""
    fp_b, fa_b = best['Flap Position'], best['Flap Angle']
    sub_b  = df_all[(df_all['Flap Position'] == fp_b) &
                    (df_all['Flap Angle'] == fa_b)].sort_values('alpha')
    stab_b = sub_b[sub_b['All Stable']]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(sub_b['CD'], sub_b['CL'], 'o-', color='#D3D1C7', ms=4, lw=1.5, label='All α')
    ax.plot(stab_b['CD'], stab_b['CL'], 'o', color='#1D9E75', ms=6, label='Stable α')
    ax.plot(best['CD'], best['CL'], '*', color='#534AB7', ms=16, zorder=5, label='Optimum')
    cd_t = np.linspace(0, sub_b['CD'].max() * 1.15, 100)
    ax.plot(cd_t, best['CL/CD'] * cd_t, '--', color='#534AB7', lw=1.2, label='Best-glide tangent')
    ax.set_xlabel('CD'); ax.set_ylabel('CL')
    ax.set_title(f'Drag polar — FP={fp_b}  FA={fa_b}°')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.2)

    bar_labels = ['CL', 'CD×100', '−Cm', 'CL/CD', '−dCm/dα×10', 'Static Margin']
    bar_vals   = [best['CL'], best['CD'] * 100, -best['Cm'],
                  best['CL/CD'], -best['dCm/dalpha'] * 10, best['Static Margin']]
    bar_colors = ['#534AB7', '#A32D2D', '#0F6E56', '#BA7517', '#1D9E75', '#0F6E56']
    ax = axes[1]
    bars = ax.bar(bar_labels, bar_vals, color=bar_colors, edgecolor='white')
    for bar, val in zip(bars, bar_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    ax.set_title(f'Optimal config summary\n'
                 f'FP={fp_b}  FA={fa_b}°  α={best["alpha"]}°  CL/CD={best["CL/CD"]}')
    ax.set_ylabel('Value'); ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    plt.savefig("optimal_config.png", dpi=150, bbox_inches='tight')
    plt.show()

# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading model...")
    model = joblib.load(MODEL_PATH)

    print("Running optimization grid search...")
    df_all, df_stable = run_optimization(model)

    print(f"\nTotal configs evaluated : {len(df_all)}")
    print(f"Stable configs found    : {len(df_stable)}")

    print("\n── Top 10 optimal configurations ──")
    print(df_stable[['Flap Position','Flap Angle','alpha',
                      'CL','Cm','CD','CL/CD',
                      'dCm/dalpha','Static Margin']].head(10).to_string(index=False))

    best = df_stable.iloc[0]
    print(f"\n{'='*54}")
    print(f"  OPTIMAL FLAP CONFIGURATION")
    print(f"{'='*54}")
    print(f"  Flap Position  : {best['Flap Position']}")
    print(f"  Flap Angle     : {best['Flap Angle']}°")
    print(f"  Trim alpha     : {best['alpha']}°")
    print(f"  CL             : {best['CL']}")
    print(f"  CD             : {best['CD']}")
    print(f"  Cm             : {best['Cm']}   (< 0  ✓)")
    print(f"  CL/CD          : {best['CL/CD']}  (glide ratio)")
    print(f"  dCm/dα         : {best['dCm/dalpha']}  (< 0  ✓)")
    print(f"  Static Margin  : {best['Static Margin']}  (> 0  ✓)")
    print(f"{'='*54}")

    df_stable.to_csv("stable_configs.csv", index=False)
    print("\nFull results saved → stable_configs.csv")

    print("\nGenerating plots...")
    plot_clcd_grid(df_all, df_stable)
    plot_heatmap(df_stable)
    plot_top5(df_all, df_stable)
    plot_optimal(df_all, df_stable, best)
    print("Done. Plots saved to current directory.")
