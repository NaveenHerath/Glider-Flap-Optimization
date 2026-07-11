import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Glider Flap Optimizer",
    page_icon="✈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Aerospace dark theme CSS ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;400;500;600;700&family=Share+Tech+Mono&family=Exo+2:wght@200;300;400;600&display=swap');

:root {
    --bg-primary:    #050a0f;
    --bg-secondary:  #0a1520;
    --bg-card:       #0d1e2e;
    --bg-panel:      #0f2336;
    --accent-cyan:   #00d4ff;
    --accent-green:  #00ff9d;
    --accent-amber:  #ffb347;
    --accent-red:    #ff4757;
    --text-primary:  #e8f4f8;
    --text-secondary:#8aacbe;
    --text-muted:    #4a6a80;
    --border:        #1a3a50;
    --border-bright: #00d4ff44;
    --glow-cyan:     0 0 20px #00d4ff33;
    --glow-green:    0 0 20px #00ff9d33;
}

html, body, [class*="css"] {
    font-family: 'Exo 2', sans-serif;
    background-color: var(--bg-primary);
    color: var(--text-primary);
}

/* Hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem; padding-bottom: 2rem; }

/* ── Hero header ── */
.hero {
    background: linear-gradient(135deg, #050a0f 0%, #0a1f35 50%, #050a0f 100%);
    border: 1px solid var(--border-bright);
    border-radius: 4px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent-cyan), var(--accent-green), transparent);
}
.hero::after {
    content: 'AERODYNAMIC ANALYSIS SYSTEM';
    position: absolute;
    bottom: 12px; right: 24px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 9px;
    color: var(--text-muted);
    letter-spacing: 3px;
}
.hero-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 3rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: 4px;
    text-transform: uppercase;
    line-height: 1;
    margin: 0;
}
.hero-title span { color: var(--accent-cyan); }
.hero-sub {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    color: var(--text-secondary);
    letter-spacing: 2px;
    margin-top: 0.6rem;
}
.hero-badge {
    display: inline-block;
    background: #00d4ff15;
    border: 1px solid var(--accent-cyan);
    color: var(--accent-cyan);
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 2px;
    padding: 3px 10px;
    border-radius: 2px;
    margin-top: 0.8rem;
    margin-right: 0.5rem;
}

/* ── Metric cards ── */
.metric-grid { display: flex; gap: 1rem; margin: 1.5rem 0; flex-wrap: wrap; }
.metric-card {
    flex: 1; min-width: 130px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 1rem 1.2rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}
.metric-card.cyan::before  { background: var(--accent-cyan); }
.metric-card.green::before { background: var(--accent-green); }
.metric-card.amber::before { background: var(--accent-amber); }
.metric-card.red::before   { background: var(--accent-red); }
.metric-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
    color: var(--text-muted);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.metric-value {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2rem;
    font-weight: 600;
    line-height: 1;
}
.metric-card.cyan  .metric-value { color: var(--accent-cyan); }
.metric-card.green .metric-value { color: var(--accent-green); }
.metric-card.amber .metric-value { color: var(--accent-amber); }
.metric-card.red   .metric-value { color: var(--accent-red); }
.metric-unit {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
    color: var(--text-muted);
    margin-top: 0.2rem;
}

/* ── Status badges ── */
.status-ok   { color: var(--accent-green); font-family: 'Share Tech Mono', monospace; font-size: 0.8rem; }
.status-fail { color: var(--accent-red);   font-family: 'Share Tech Mono', monospace; font-size: 0.8rem; }
.status-warn { color: var(--accent-amber); font-family: 'Share Tech Mono', monospace; font-size: 0.8rem; }

/* ── Section headers ── */
.section-header {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--accent-cyan);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
    margin: 1.5rem 0 1rem 0;
}

/* ── Info panel ── */
.info-panel {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent-cyan);
    border-radius: 0 4px 4px 0;
    padding: 1rem 1.2rem;
    margin: 1rem 0;
    font-size: 0.88rem;
    color: var(--text-secondary);
    line-height: 1.7;
}

/* ── Stability result box ── */
.stability-box {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 1.2rem;
    margin: 0.5rem 0;
}
.stability-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.4rem 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.88rem;
}
.stability-row:last-child { border-bottom: none; }
.stability-name { color: var(--text-secondary); font-family: 'Share Tech Mono', monospace; font-size: 0.75rem; }
.stability-val  { color: var(--text-primary); font-weight: 600; }

/* ── Optimal result banner ── */
.optimal-banner {
    background: linear-gradient(135deg, #00ff9d08, #00d4ff08);
    border: 1px solid var(--accent-green);
    border-radius: 4px;
    padding: 1.5rem 2rem;
    margin: 1rem 0;
    position: relative;
}
.optimal-banner::before {
    content: '★ OPTIMAL CONFIGURATION FOUND';
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 3px;
    color: var(--accent-green);
    display: block;
    margin-bottom: 0.8rem;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .stSlider > div > div {
    background: var(--accent-cyan) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border);
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: var(--text-muted);
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.85rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    border: none;
    padding: 0.6rem 1.5rem;
}
.stTabs [aria-selected="true"] {
    background: var(--bg-card) !important;
    color: var(--accent-cyan) !important;
    border-bottom: 2px solid var(--accent-cyan) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #00d4ff15, #00d4ff25) !important;
    border: 1px solid var(--accent-cyan) !important;
    color: var(--accent-cyan) !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 0.9rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    border-radius: 2px !important;
    padding: 0.5rem 2rem !important;
    transition: all 0.2s !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #00d4ff25, #00d4ff40) !important;
    box-shadow: var(--glow-cyan) !important;
}

/* ── Selectbox / sliders ── */
.stSelectbox label, .stSlider label, .stNumberInput label {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.72rem !important;
    color: var(--text-muted) !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
}

/* ── DataFrame ── */
.stDataFrame { border: 1px solid var(--border); border-radius: 4px; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--accent-cyan) !important; }

/* ── Warning / success ── */
.stAlert { border-radius: 2px !important; }
</style>
""", unsafe_allow_html=True)

# ── Matplotlib dark theme ─────────────────────────────────────────────────
plt.style.use('dark_background')
rcParams.update({
    'figure.facecolor':  '#0d1e2e',
    'axes.facecolor':    '#0a1520',
    'axes.edgecolor':    '#1a3a50',
    'axes.labelcolor':   '#8aacbe',
    'xtick.color':       '#4a6a80',
    'ytick.color':       '#4a6a80',
    'text.color':        '#e8f4f8',
    'grid.color':        '#1a3a50',
    'grid.alpha':        0.5,
    'font.family':       'monospace',
})

ACCENT_CYAN  = '#00d4ff'
ACCENT_GREEN = '#00ff9d'
ACCENT_AMBER = '#ffb347'
ACCENT_RED   = '#ff4757'
ACCENT_PURPLE= '#9b59b6'

# ── Model loading ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    path = "glider_surrogate_model.pkl"
    if not os.path.exists(path):
        return None
    return joblib.load(path)

model = load_model()

# ── Core prediction helpers ───────────────────────────────────────────────
LABEL_COLS = ['CL', 'Cm', 'CD']

def make_features(fp, fa, alpha):
    return np.array([[fp, fa, alpha,
                      alpha * fa, alpha * fp,
                      alpha ** 2, fa ** 2]])

def predict_coeffs(fp, fa, alpha):
    pred = model.predict(make_features(fp, fa, alpha))[0]
    return dict(zip(LABEL_COLS, pred))

def dcm_dalpha(fp, fa, alpha, da=0.1):
    cm_hi = predict_coeffs(fp, fa, alpha + da)['Cm']
    cm_lo = predict_coeffs(fp, fa, alpha - da)['Cm']
    return (cm_hi - cm_lo) / (2 * da)

def static_margin(fp, fa, alpha, da=0.1):
    p_hi = predict_coeffs(fp, fa, alpha + da)
    p_lo = predict_coeffs(fp, fa, alpha - da)
    dCm = p_hi['Cm'] - p_lo['Cm']
    dCL = p_hi['CL'] - p_lo['CL']
    return -dCm / dCL if abs(dCL) > 1e-9 else 0.0

def stability_verdict(cm, dcm_da, sm):
    c1 = cm < 0
    c2 = dcm_da < 0
    c3 = sm > 0
    all_ok = c1 and c2 and c3
    return c1, c2, c3, all_ok

# ── Optimizer ────────────────────────────────────────────────────────────
FLAP_POSITIONS = [0.0, 0.65, 0.70, 0.75]
FLAP_ANGLES    = [0, 2, 4, 6, 8, 10]

@st.cache_data
def run_optimizer():
    rows = []
    for fp in FLAP_POSITIONS:
        for fa in FLAP_ANGLES:
            for alpha in np.arange(-3.5, 9.6, 0.1):
                p = predict_coeffs(fp, fa, alpha)
                CL, Cm, CD = p['CL'], p['Cm'], p['CD']
                if CD < 1e-6: continue
                glide  = CL / CD
                dCm_da = dcm_dalpha(fp, fa, alpha)
                sm     = static_margin(fp, fa, alpha)
                c1, c2, c3, ok = stability_verdict(Cm, dCm_da, sm)
                rows.append({
                    'Flap Position': fp, 'Flap Angle': fa,
                    'alpha': round(float(alpha), 1),
                    'CL': round(CL, 4), 'Cm': round(Cm, 4), 'CD': round(CD, 5),
                    'CL/CD': round(glide, 3),
                    'dCm/dα': round(dCm_da, 5),
                    'Static Margin': round(sm, 4),
                    'All Stable': ok,
                })
    df = pd.DataFrame(rows)
    stable = df[df['All Stable']].sort_values('CL/CD', ascending=False).reset_index(drop=True)
    return df, stable

# ═══════════════════════════════════════════════════════════════════════════
# HERO HEADER
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <p class="hero-title">GLIDER <span>FLAP</span> OPTIMIZER</p>
  <p class="hero-sub">// SURROGATE ML MODEL · AERODYNAMIC COEFFICIENT PREDICTION · STABILITY ANALYSIS</p>
  <span class="hero-badge">GRADIENT BOOSTING</span>
  <span class="hero-badge">MULTI-OUTPUT REGRESSION</span>
  <span class="hero-badge">PITCH STABILITY</span>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error("⚠ Model file `glider_surrogate_model.pkl` not found. "
             "Run `glider_ml_pipeline_final.py` first to generate it.")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR — Input controls
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<p class="section-header">// Flight Parameters</p>', unsafe_allow_html=True)

    st.markdown('<p style="font-family:\'Share Tech Mono\',monospace;font-size:0.65rem;'
                'color:#4a6a80;letter-spacing:2px;">FLAP POSITION (chord fraction)</p>',
                unsafe_allow_html=True)
    flap_pos = st.select_slider(
        "Flap Position", options=[0.0, 0.65, 0.70, 0.75],
        value=0.65, label_visibility="collapsed"
    )

    st.markdown('<p style="font-family:\'Share Tech Mono\',monospace;font-size:0.65rem;'
                'color:#4a6a80;letter-spacing:2px;margin-top:1rem;">FLAP ANGLE (degrees)</p>',
                unsafe_allow_html=True)
    flap_angle = st.select_slider(
        "Flap Angle", options=[0, 2, 4, 6, 8, 10],
        value=2, label_visibility="collapsed"
    )

    st.markdown('<p style="font-family:\'Share Tech Mono\',monospace;font-size:0.65rem;'
                'color:#4a6a80;letter-spacing:2px;margin-top:1rem;">ANGLE OF ATTACK α (degrees)</p>',
                unsafe_allow_html=True)
    alpha = st.slider(
        "Alpha", min_value=-3.5, max_value=9.5, value=2.0, step=0.1,
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown('<p class="section-header">// Model Info</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-panel">
    Trained on <strong style="color:#00d4ff">409 CFD simulations</strong>
    across 24 flap configurations.<br><br>
    <strong>R² scores (test set)</strong><br>
    CL: 0.9988 &nbsp;|&nbsp; Cm: 0.9953<br>
    CD: 0.9972
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p style="font-family:\'Share Tech Mono\',monospace;font-size:0.6rem;'
                'color:#2a4a60;letter-spacing:1px;text-align:center;margin-top:2rem;">'
                '© GLIDER FLAP OPTIMIZER · ML SURROGATE</p>',
                unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "⬡  PREDICT",
    "⬡  STABILITY",
    "⬡  OPTIMIZER",
    "⬡  COMPARE",
])

# ───────────────────────────────────────────────────────────────────────────
# TAB 1 — Predict coefficients
# ───────────────────────────────────────────────────────────────────────────
with tab1:
    p = predict_coeffs(flap_pos, flap_angle, alpha)
    CL, Cm, CD = p['CL'], p['Cm'], p['CD']
    glide = CL / CD if CD > 0 else 0

    st.markdown('<p class="section-header">// Aerodynamic Coefficients</p>',
                unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-grid">
      <div class="metric-card cyan">
        <div class="metric-label">Lift Coefficient</div>
        <div class="metric-value">{CL:.4f}</div>
        <div class="metric-unit">CL</div>
      </div>
      <div class="metric-card amber">
        <div class="metric-label">Pitching Moment</div>
        <div class="metric-value">{Cm:.4f}</div>
        <div class="metric-unit">Cm</div>
      </div>
      <div class="metric-card red">
        <div class="metric-label">Drag Coefficient</div>
        <div class="metric-value">{CD:.5f}</div>
        <div class="metric-unit">CD</div>
      </div>
      <div class="metric-card green">
        <div class="metric-label">Glide Ratio</div>
        <div class="metric-value">{glide:.2f}</div>
        <div class="metric-unit">CL / CD</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Alpha sweep plot ──────────────────────────────────────────────────
    st.markdown('<p class="section-header">// Coefficient Curves vs α</p>',
                unsafe_allow_html=True)

    alphas = np.arange(-3.5, 9.6, 0.1)
    CLs, Cms, CDs, glides = [], [], [], []
    for a in alphas:
        pp = predict_coeffs(flap_pos, flap_angle, a)
        CLs.append(pp['CL'])
        Cms.append(pp['Cm'])
        CDs.append(pp['CD'])
        glides.append(pp['CL'] / pp['CD'] if pp['CD'] > 0 else 0)

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    fig.patch.set_facecolor('#0d1e2e')

    for ax, ys, lbl, col in zip(
        axes,
        [CLs, Cms, CDs, glides],
        ['CL', 'Cm', 'CD', 'CL/CD'],
        [ACCENT_CYAN, ACCENT_AMBER, ACCENT_RED, ACCENT_GREEN]
    ):
        ax.plot(alphas, ys, color=col, lw=2)
        ax.axvline(alpha, color='white', lw=0.8, ls='--', alpha=0.5)
        ax.axhline(0, color='#1a3a50', lw=0.8)
        idx = np.argmin(np.abs(alphas - alpha))
        ax.scatter([alpha], [ys[idx]], color=col, s=60, zorder=5)
        ax.set_xlabel('α (°)', fontsize=9)
        ax.set_title(lbl, color=col, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.tight_layout(pad=0.8)
    st.pyplot(fig)
    plt.close()

    st.markdown(f"""
    <div class="info-panel">
    <strong>Config:</strong> Flap Position = {flap_pos} &nbsp;|&nbsp;
    Flap Angle = {flap_angle}° &nbsp;|&nbsp; α = {alpha}°
    &nbsp;&nbsp;·&nbsp;&nbsp;
    The dashed vertical line marks your selected angle of attack.
    Glide ratio peaks near α = 3–5° for most configurations.
    </div>
    """, unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────────────────
# TAB 2 — Stability analysis
# ───────────────────────────────────────────────────────────────────────────
with tab2:
    p    = predict_coeffs(flap_pos, flap_angle, alpha)
    CL, Cm, CD = p['CL'], p['Cm'], p['CD']
    dCm_da = dcm_dalpha(flap_pos, flap_angle, alpha)
    sm     = static_margin(flap_pos, flap_angle, alpha)
    c1, c2, c3, all_ok = stability_verdict(Cm, dCm_da, sm)

    st.markdown('<p class="section-header">// Pitch Stability Analysis</p>',
                unsafe_allow_html=True)

    verdict_color = "var(--accent-green)" if all_ok else "var(--accent-red)"
    verdict_text  = "STABLE" if all_ok else "UNSTABLE"
    verdict_icon  = "✔" if all_ok else "✘"

    st.markdown(f"""
    <div style="background:var(--bg-card);border:1px solid {verdict_color};
                border-radius:4px;padding:1.2rem 1.5rem;margin-bottom:1.5rem;
                text-align:center;">
      <span style="font-family:'Rajdhani',sans-serif;font-size:2.5rem;
                   font-weight:700;color:{verdict_color};letter-spacing:6px;">
        {verdict_icon} {verdict_text}
      </span>
      <p style="font-family:'Share Tech Mono',monospace;font-size:0.7rem;
                color:var(--text-muted);letter-spacing:2px;margin:0.4rem 0 0 0;">
        FP={flap_pos} · FA={flap_angle}° · α={alpha}°
      </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-header">// Constraint Checks</p>',
                    unsafe_allow_html=True)

        def badge(ok, true_label, false_label):
            cls = "status-ok" if ok else "status-fail"
            txt = f"✔ {true_label}" if ok else f"✘ {false_label}"
            return f'<span class="{cls}">{txt}</span>'

        st.markdown(f"""
        <div class="stability-box">
          <div class="stability-row">
            <span class="stability-name">Cm &lt; 0 &nbsp;(trim)</span>
            <span class="stability-val">{Cm:.4f}</span>
            {badge(c1, "PASS", "FAIL")}
          </div>
          <div class="stability-row">
            <span class="stability-name">dCm/dα &lt; 0 &nbsp;(pitch stiffness)</span>
            <span class="stability-val">{dCm_da:.5f}</span>
            {badge(c2, "PASS", "FAIL")}
          </div>
          <div class="stability-row">
            <span class="stability-name">Static Margin &gt; 0</span>
            <span class="stability-val">{sm:.4f}</span>
            {badge(c3, "PASS", "FAIL")}
          </div>
        </div>
        """, unsafe_allow_html=True)

        sm_pct = sm * 100
        sm_color = ACCENT_GREEN if sm > 0.05 else (ACCENT_AMBER if sm > 0 else ACCENT_RED)
        st.markdown(f"""
        <div class="info-panel" style="margin-top:1rem;">
        <strong>Static Margin = {sm:.4f}</strong> &nbsp;({sm_pct:.1f}% MAC)<br>
        {"✔ CG is ahead of neutral point — inherently stable." if sm > 0
          else "✘ CG is aft of neutral point — divergent in pitch."}<br><br>
        <span style="color:var(--text-muted);font-size:0.8rem;">
        Typical glider design target: 10–20% MAC static margin.
        </span>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<p class="section-header">// Cm vs α Curve</p>',
                    unsafe_allow_html=True)

        alphas = np.arange(-3.5, 9.6, 0.1)
        cms    = [predict_coeffs(flap_pos, flap_angle, a)['Cm'] for a in alphas]

        fig, ax = plt.subplots(figsize=(6, 3.5))
        fig.patch.set_facecolor('#0d1e2e')
        ax.set_facecolor('#0a1520')

        stable_mask = np.array(cms) < 0
        ax.fill_between(alphas, cms, 0, where=stable_mask,
                        alpha=0.15, color=ACCENT_GREEN, label='Stable region (Cm<0)')
        ax.fill_between(alphas, cms, 0, where=~stable_mask,
                        alpha=0.15, color=ACCENT_RED, label='Unstable region')
        ax.plot(alphas, cms, color=ACCENT_AMBER, lw=2)
        ax.axhline(0, color='#4a6a80', lw=0.8, ls='--')
        ax.axvline(alpha, color='white', lw=0.8, ls=':', alpha=0.6)
        idx = np.argmin(np.abs(alphas - alpha))
        ax.scatter([alpha], [cms[idx]], color=ACCENT_AMBER, s=60, zorder=5)
        ax.set_xlabel('α (°)', fontsize=9)
        ax.set_ylabel('Cm', fontsize=9)
        ax.set_title('Pitching Moment vs Angle of Attack',
                     fontsize=9, color='#8aacbe')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ───────────────────────────────────────────────────────────────────────────
# TAB 3 — Optimizer
# ───────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<p class="section-header">// Global Flap Optimizer</p>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="info-panel">
    Exhaustive search over all 24 discrete (Flap Position, Flap Angle) combinations
    and a fine α grid [−3.5°…9.5°, step 0.1°] — 3,144 total evaluations.
    Constraints: <strong>Cm &lt; 0</strong> · <strong>dCm/dα &lt; 0</strong> ·
    <strong>Static Margin &gt; 0</strong>.<br>
    Objective: <strong>maximise CL/CD (glide ratio)</strong>.
    </div>
    """, unsafe_allow_html=True)

    run_btn = st.button("⬡  RUN OPTIMIZER")

    if run_btn or 'opt_done' in st.session_state:
        with st.spinner("Running optimization sweep across 3,144 configurations..."):
            df_all, df_stable = run_optimizer()
            st.session_state['opt_done'] = True

        best = df_stable.iloc[0]

        st.markdown(f"""
        <div class="optimal-banner">
          <div class="metric-grid">
            <div class="metric-card cyan" style="min-width:100px;">
              <div class="metric-label">Flap Position</div>
              <div class="metric-value">{best['Flap Position']}</div>
              <div class="metric-unit">chord fraction</div>
            </div>
            <div class="metric-card cyan" style="min-width:100px;">
              <div class="metric-label">Flap Angle</div>
              <div class="metric-value">{best['Flap Angle']}°</div>
              <div class="metric-unit">degrees</div>
            </div>
            <div class="metric-card cyan" style="min-width:100px;">
              <div class="metric-label">Trim Alpha</div>
              <div class="metric-value">{best['alpha']}°</div>
              <div class="metric-unit">degrees</div>
            </div>
            <div class="metric-card green" style="min-width:100px;">
              <div class="metric-label">Glide Ratio</div>
              <div class="metric-value">{best['CL/CD']}</div>
              <div class="metric-unit">CL / CD</div>
            </div>
            <div class="metric-card amber" style="min-width:100px;">
              <div class="metric-label">Static Margin</div>
              <div class="metric-value">{best['Static Margin']:.3f}</div>
              <div class="metric-unit">{best['Static Margin']*100:.1f}% MAC</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)

        with c1:
            st.markdown('<p class="section-header">// Stability Heatmap</p>',
                        unsafe_allow_html=True)
            pivot = df_stable.groupby(
                ['Flap Position', 'Flap Angle'])['CL/CD'].max().unstack(fill_value=np.nan)
            fig, ax = plt.subplots(figsize=(7, 4))
            fig.patch.set_facecolor('#0d1e2e')
            ax.set_facecolor('#0a1520')
            im = ax.imshow(pivot.values, aspect='auto', cmap='RdYlGn',
                           vmin=10, vmax=28, origin='lower')
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([f'{v}°' for v in pivot.columns], fontsize=9)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index, fontsize=9)
            ax.set_xlabel('Flap Angle', fontsize=9)
            ax.set_ylabel('Flap Position', fontsize=9)
            ax.set_title('Best stable CL/CD per config', fontsize=9, color='#8aacbe')
            plt.colorbar(im, ax=ax, label='Max stable CL/CD')
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    val = pivot.values[i, j]
                    if not np.isnan(val):
                        ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                                fontsize=8, color='white')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with c2:
            st.markdown('<p class="section-header">// Drag Polar — Optimal Config</p>',
                        unsafe_allow_html=True)
            fp_b, fa_b = best['Flap Position'], best['Flap Angle']
            mask = (df_all['Flap Position'] == fp_b) & (df_all['Flap Angle'] == fa_b)
            sub_b  = df_all[mask].sort_values('alpha')
            stab_b = sub_b[sub_b['All Stable']]

            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor('#0d1e2e')
            ax.set_facecolor('#0a1520')
            ax.plot(sub_b['CD'], sub_b['CL'], 'o-',
                    color='#1a3a50', ms=3, lw=1.5, label='All α')
            ax.plot(stab_b['CD'], stab_b['CL'], 'o',
                    color=ACCENT_GREEN, ms=5, label='Stable α')
            ax.plot(best['CD'], best['CL'], '*',
                    color=ACCENT_CYAN, ms=14, zorder=5, label='Optimum')
            cd_t = np.linspace(0, sub_b['CD'].max() * 1.15, 100)
            ax.plot(cd_t, best['CL/CD'] * cd_t, '--',
                    color=ACCENT_CYAN, lw=1, label='Best-glide tangent')
            ax.set_xlabel('CD', fontsize=9)
            ax.set_ylabel('CL', fontsize=9)
            ax.set_title(f'FP={fp_b}  FA={fa_b}°', fontsize=9, color='#8aacbe')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown('<p class="section-header">// Top 20 Stable Configurations</p>',
                    unsafe_allow_html=True)
        display_cols = ['Flap Position', 'Flap Angle', 'alpha',
                        'CL', 'Cm', 'CD', 'CL/CD', 'dCm/dα', 'Static Margin']
        st.dataframe(
            df_stable[display_cols].head(20).style
            .background_gradient(subset=['CL/CD'], cmap='RdYlGn')
            .format({'CL': '{:.4f}', 'Cm': '{:.4f}', 'CD': '{:.5f}',
                     'CL/CD': '{:.3f}', 'dCm/dα': '{:.5f}', 'Static Margin': '{:.4f}'}),
            use_container_width=True, height=420
        )

        csv = df_stable[display_cols].to_csv(index=False)
        st.download_button(
            label="⬡  DOWNLOAD STABLE CONFIGS CSV",
            data=csv,
            file_name="stable_flap_configs.csv",
            mime="text/csv",
        )

# ───────────────────────────────────────────────────────────────────────────
# TAB 4 — Compare configs
# ───────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<p class="section-header">// Compare Flap Configurations</p>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="info-panel">
    Add up to 4 configurations to compare their coefficient curves side by side.
    </div>
    """, unsafe_allow_html=True)

    if 'compare_list' not in st.session_state:
        st.session_state.compare_list = []

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        cmp_fp = st.select_slider("FP", options=[0.0, 0.65, 0.70, 0.75],
                                   value=0.65, key='cmp_fp')
    with col_b:
        cmp_fa = st.select_slider("FA (°)", options=[0, 2, 4, 6, 8, 10],
                                   value=2, key='cmp_fa')
    with col_c:
        cmp_alpha = st.slider("α (°)", -3.5, 9.5, 2.0, 0.1, key='cmp_alpha')
    with col_d:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("⬡  ADD CONFIG"):
            entry = {'fp': cmp_fp, 'fa': cmp_fa, 'alpha': cmp_alpha}
            if len(st.session_state.compare_list) < 4:
                st.session_state.compare_list.append(entry)
        if st.button("⬡  CLEAR ALL"):
            st.session_state.compare_list = []

    if st.session_state.compare_list:
        colors_cmp = [ACCENT_CYAN, ACCENT_GREEN, ACCENT_AMBER, ACCENT_RED]
        alphas_plt  = np.arange(-3.5, 9.6, 0.1)

        fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
        fig.patch.set_facecolor('#0d1e2e')
        for ax in axes:
            ax.set_facecolor('#0a1520')

        labels_plt = ['CL', 'Cm', 'CD', 'CL/CD']
        label_colors = [ACCENT_CYAN, ACCENT_AMBER, ACCENT_RED, ACCENT_GREEN]

        for cfg, col in zip(st.session_state.compare_list, colors_cmp):
            CLs, Cms, CDs, Gs = [], [], [], []
            for a in alphas_plt:
                pp = predict_coeffs(cfg['fp'], cfg['fa'], a)
                CLs.append(pp['CL'])
                Cms.append(pp['Cm'])
                CDs.append(pp['CD'])
                Gs.append(pp['CL'] / pp['CD'] if pp['CD'] > 0 else 0)

            lbl = f"FP={cfg['fp']} FA={cfg['fa']}°"
            for ax, ys in zip(axes, [CLs, Cms, CDs, Gs]):
                ax.plot(alphas_plt, ys, color=col, lw=2, label=lbl)
                ax.axvline(cfg['alpha'], color=col, lw=0.6, ls=':', alpha=0.5)

        for ax, lbl, col in zip(axes, labels_plt, label_colors):
            ax.axhline(0, color='#1a3a50', lw=0.8)
            ax.set_xlabel('α (°)', fontsize=9)
            ax.set_title(lbl, color=col, fontsize=11, fontweight='bold')
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        plt.tight_layout(pad=0.8)
        st.pyplot(fig)
        plt.close()

        # Comparison table
        rows = []
        for cfg in st.session_state.compare_list:
            p   = predict_coeffs(cfg['fp'], cfg['fa'], cfg['alpha'])
            dcd = dcm_dalpha(cfg['fp'], cfg['fa'], cfg['alpha'])
            sm  = static_margin(cfg['fp'], cfg['fa'], cfg['alpha'])
            _, _, _, ok = stability_verdict(p['Cm'], dcd, sm)
            rows.append({
                'Config': f"FP={cfg['fp']} FA={cfg['fa']}° α={cfg['alpha']}°",
                'CL': round(p['CL'], 4), 'Cm': round(p['Cm'], 4),
                'CD': round(p['CD'], 5),
                'CL/CD': round(p['CL'] / p['CD'], 3) if p['CD'] > 0 else 0,
                'Static Margin': round(sm, 4),
                'Stable': '✔ YES' if ok else '✘ NO',
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.markdown("""
        <div style="text-align:center;padding:3rem;color:var(--text-muted);
                    font-family:'Share Tech Mono',monospace;font-size:0.8rem;
                    letter-spacing:2px;">
        NO CONFIGURATIONS ADDED YET<br>
        <span style="font-size:0.65rem;color:#2a4a60;">
        SELECT A CONFIG ABOVE AND PRESS ADD CONFIG
        </span>
        </div>
        """, unsafe_allow_html=True)
