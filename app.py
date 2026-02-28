"""
AutoHeal AI – Intelligent Self-Healing Infrastructure Engine
Streamlit dashboard entry point.

Layout
------
  Sidebar  : simulation controls
  Row 1    : executive KPI cards
  Row 2    : health score gauge  |  failure probability trend
  Row 3    : CPU + Memory charts
  Row 4    : Disk I/O + Network Latency charts
  Row 5    : recovery action log  |  stability index
  Row 6    : before/after recovery comparison (when recoveries exist)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import (
    CHART_HEIGHT,
    DEFAULT_STEPS,
    HEALTH_WEIGHTS,
    HISTORY_DISPLAY_STEPS,
    ANOMALY_INJECT_PROB,
    METRIC_BASELINES,
    RECOVERY_RISK_THRESHOLD,
    MetricSnapshot,
    RecoveryEvent,
)
from detection import AnomalyDetector, MIN_FIT_SAMPLES as DETECT_MIN
from prediction import FailurePredictor, MIN_FIT_SAMPLES as PREDICT_MIN
from recovery import RecoveryEngine, compute_health_score
from simulation import InfrastructureSimulator

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AutoHeal AI",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Dark-theme CSS injection
# ---------------------------------------------------------------------------

DARK_CSS = """
<style>
    /* Base */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0d1117;
        color: #c9d1d9;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    /* KPI cards */
    .kpi-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 16px 20px;
        margin-bottom: 8px;
    }
    .kpi-label {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #8b949e;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 28px;
        font-weight: 700;
        color: #e6edf3;
        line-height: 1.1;
    }
    .kpi-sub {
        font-size: 12px;
        color: #8b949e;
        margin-top: 4px;
    }
    .kpi-value.danger  { color: #f85149; }
    .kpi-value.warning { color: #d29922; }
    .kpi-value.good    { color: #3fb950; }
    /* Section headers */
    .section-header {
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #8b949e;
        border-bottom: 1px solid #30363d;
        padding-bottom: 6px;
        margin-bottom: 12px;
        margin-top: 4px;
    }
    /* Recovery log table */
    .recovery-row {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 4px;
        padding: 10px 14px;
        margin-bottom: 6px;
        font-size: 13px;
    }
    .recovery-action { font-weight: 600; color: #79c0ff; }
    .recovery-meta   { color: #8b949e; font-size: 12px; margin-top: 2px; }
    /* Plotly chart container */
    .stPlotlyChart { border: 1px solid #30363d; border-radius: 6px; }
    /* Divider */
    hr { border-color: #30363d; }
    /* Streamlit element overrides */
    .stButton > button {
        background-color: #238636;
        color: #ffffff;
        border: none;
        border-radius: 6px;
        font-size: 14px;
        font-weight: 600;
        padding: 8px 20px;
    }
    .stButton > button:hover { background-color: #2ea043; }
    div[data-testid="metric-container"] { display: none; }
</style>
"""

st.markdown(DARK_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Plotly theme helpers
# ---------------------------------------------------------------------------

PLOTLY_LAYOUT_BASE = dict(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#0d1117",
    font=dict(color="#c9d1d9", size=11),
    margin=dict(l=40, r=16, t=36, b=36),
    xaxis=dict(gridcolor="#21262d", zerolinecolor="#30363d"),
    yaxis=dict(gridcolor="#21262d", zerolinecolor="#30363d"),
)

PALETTE = {
    "cpu":         "#58a6ff",
    "memory":      "#bc8cff",
    "disk":        "#ffa657",
    "network":     "#3fb950",
    "error":       "#f85149",
    "availability":"#79c0ff",
    "response":    "#d2a8ff",
    "probability": "#f85149",
    "health":      "#3fb950",
}


def _line_chart(
    df: pd.DataFrame,
    columns: List[Tuple[str, str, str]],
    title: str,
    yrange: Optional[List[float]] = None,
) -> go.Figure:
    """
    Build a compact Plotly line chart with the dark theme.

    Parameters
    ----------
    df : DataFrame
        Must have a 'step' column plus the metric columns.
    columns : list of (col_name, display_label, colour)
    title : str
    yrange : optional [min, max]
    """
    fig = go.Figure()
    for col, label, colour in columns:
        if col not in df.columns:
            continue
        fig.add_trace(go.Scatter(
            x=df["step"],
            y=df[col],
            name=label,
            mode="lines",
            line=dict(color=colour, width=1.8),
        ))
    layout = dict(**PLOTLY_LAYOUT_BASE, title=dict(text=title, font=dict(size=13)))
    if yrange:
        layout["yaxis"] = dict(
            **PLOTLY_LAYOUT_BASE["yaxis"],
            range=yrange,
        )
    fig.update_layout(**layout, height=CHART_HEIGHT, showlegend=True)
    return fig


def _gauge(value: float, title: str) -> go.Figure:
    """Render a Plotly gauge for the health score."""
    colour = (
        "#3fb950" if value >= 0.70 else
        "#d29922" if value >= 0.40 else
        "#f85149"
    )
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(value * 100, 1),
        number=dict(suffix="%", font=dict(size=40, color=colour)),
        title=dict(text=title, font=dict(size=13, color="#8b949e")),
        gauge=dict(
            axis=dict(
                range=[0, 100],
                tickcolor="#30363d",
                tickfont=dict(color="#8b949e", size=10),
            ),
            bar=dict(color=colour),
            bgcolor="#161b22",
            bordercolor="#30363d",
            steps=[
                dict(range=[0, 40],  color="#2d1a1a"),
                dict(range=[40, 70], color="#2d2513"),
                dict(range=[70, 100], color="#122d1a"),
            ],
        ),
    ))
    fig.update_layout(
        paper_bgcolor="#0d1117",
        font=dict(color="#c9d1d9"),
        height=CHART_HEIGHT,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


# ---------------------------------------------------------------------------
# Colour helpers for KPI cards
# ---------------------------------------------------------------------------

def _health_class(score: float) -> str:
    if score >= 0.70:
        return "good"
    if score >= 0.40:
        return "warning"
    return "danger"


def _prob_class(prob: float) -> str:
    if prob <= 0.35:
        return "good"
    if prob <= 0.60:
        return "warning"
    return "danger"


def _kpi_html(label: str, value: str, sub: str, css_class: str = "") -> str:
    cls = f"kpi-value {css_class}".strip()
    return (
        f'<div class="kpi-card">'
        f'  <div class="kpi-label">{label}</div>'
        f'  <div class="{cls}">{value}</div>'
        f'  <div class="kpi-sub">{sub}</div>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------

def _init_session() -> None:
    if "simulator" not in st.session_state:
        st.session_state["simulator"] = None
    if "detector" not in st.session_state:
        st.session_state["detector"] = AnomalyDetector()
    if "predictor" not in st.session_state:
        st.session_state["predictor"] = FailurePredictor()
    if "recovery_engine" not in st.session_state:
        st.session_state["recovery_engine"] = RecoveryEngine()
    if "health_scores" not in st.session_state:
        st.session_state["health_scores"] = []
    if "anomaly_scores" not in st.session_state:
        st.session_state["anomaly_scores"] = []
    if "failure_probs" not in st.session_state:
        st.session_state["failure_probs"] = []
    if "prob_steps" not in st.session_state:
        st.session_state["prob_steps"] = []
    if "running" not in st.session_state:
        st.session_state["running"] = False
    if "completed" not in st.session_state:
        st.session_state["completed"] = False


_init_session()


# ---------------------------------------------------------------------------
# Core simulation runner
# ---------------------------------------------------------------------------

def run_simulation(n_steps: int, seed: int, anomaly_prob: float) -> None:
    """Execute a full simulation run, fitting models and triggering recovery."""
    sim = InfrastructureSimulator(
        seed=seed,
        anomaly_prob=anomaly_prob,
    )
    detector = AnomalyDetector()
    predictor = FailurePredictor()
    engine = RecoveryEngine()

    health_scores: List[float] = []
    anomaly_scores: List[float] = []
    failure_probs: List[float] = []
    prob_steps: List[int] = []

    # Progress display
    progress_bar = st.progress(0, text="Initialising simulation ...")
    status_text = st.empty()

    for i in range(n_steps):
        snapshot = sim.step()
        health = compute_health_score(snapshot)
        health_scores.append(health)

        # Anomaly detection (requires enough history)
        history = sim.history
        if len(history) >= DETECT_MIN:
            if not detector.is_fitted:
                detector.fit(history[:len(history)])
            a_score = detector.score(snapshot)
        else:
            a_score = 0.0
        anomaly_scores.append(a_score)

        # Failure prediction (requires enough labelled history)
        if len(history) >= PREDICT_MIN:
            if not predictor.is_fitted:
                predictor.fit(history[:len(history)], anomaly_scores[:len(history)])
            fp = predictor.predict_proba(snapshot)
        else:
            fp = 0.0
        failure_probs.append(fp)
        prob_steps.append(snapshot.step)

        # Recovery check
        if predictor.is_fitted and engine.should_trigger(fp):
            action = engine.select_action(snapshot, snapshot.step)
            if action is not None:
                event, bias = engine.execute(action, snapshot, fp, snapshot.step)
                sim.apply_recovery_bias(bias)
                sim.log_recovery_event(event)

        # Re-fit models periodically to incorporate new distribution
        if (i + 1) % 20 == 0 and len(history) >= PREDICT_MIN:
            scores_so_far = anomaly_scores[:len(history)]
            detector.fit(history)
            predictor.fit(history, scores_so_far)

        progress_bar.progress(
            (i + 1) / n_steps,
            text=f"Step {i + 1} / {n_steps}  |  Health: {health:.2f}  |  "
                 f"Failure prob: {fp:.2f}",
        )

    progress_bar.empty()
    status_text.empty()

    # Persist to session state
    st.session_state["simulator"] = sim
    st.session_state["detector"] = detector
    st.session_state["predictor"] = predictor
    st.session_state["recovery_engine"] = engine
    st.session_state["health_scores"] = health_scores
    st.session_state["anomaly_scores"] = anomaly_scores
    st.session_state["failure_probs"] = failure_probs
    st.session_state["prob_steps"] = prob_steps
    st.session_state["running"] = False
    st.session_state["completed"] = True


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        '<p style="font-size:20px;font-weight:700;color:#e6edf3;margin-bottom:4px;">'
        "AutoHeal AI</p>"
        '<p style="font-size:12px;color:#8b949e;margin-top:0;">Infrastructure Engine</p>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    n_steps = st.slider(
        "Simulation steps",
        min_value=40,
        max_value=300,
        value=DEFAULT_STEPS,
        step=10,
        help="Number of time-steps in one simulation run.",
    )
    seed = st.number_input(
        "Random seed",
        min_value=0,
        max_value=9999,
        value=42,
        step=1,
    )
    anomaly_prob = st.slider(
        "Anomaly injection probability",
        min_value=0.01,
        max_value=0.30,
        value=ANOMALY_INJECT_PROB,
        step=0.01,
        help="Per-step probability of a synthetic anomaly spike.",
    )

    st.markdown("---")
    run_btn = st.button("Run Simulation", use_container_width=True)
    if run_btn:
        st.session_state["running"] = True
        st.session_state["completed"] = False

    st.markdown("---")
    st.markdown(
        '<p class="kpi-label">Risk threshold</p>'
        f'<p style="font-size:20px;font-weight:700;color:#e6edf3;">'
        f'{int(RECOVERY_RISK_THRESHOLD * 100)}%</p>'
        '<p class="kpi-sub">Recovery triggers above this failure probability.</p>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main content area
# ---------------------------------------------------------------------------

st.markdown(
    '<h1 style="font-size:22px;font-weight:700;color:#e6edf3;margin-bottom:2px;">'
    "AutoHeal AI &ndash; Intelligent Self-Healing Infrastructure Engine</h1>"
    '<p style="font-size:13px;color:#8b949e;margin-top:0;margin-bottom:24px;">'
    "Real-time anomaly detection, failure prediction, and autonomous recovery.</p>",
    unsafe_allow_html=True,
)

# Run simulation if triggered
if st.session_state.get("running", False):
    run_simulation(n_steps=n_steps, seed=int(seed), anomaly_prob=anomaly_prob)
    st.rerun()

# Show placeholder when nothing has run yet
if not st.session_state.get("completed", False):
    st.info(
        "Configure the simulation parameters in the sidebar and click "
        "\"Run Simulation\" to begin."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Data extraction from session state
# ---------------------------------------------------------------------------

sim: InfrastructureSimulator = st.session_state["simulator"]
engine: RecoveryEngine = st.session_state["recovery_engine"]
health_scores: List[float] = st.session_state["health_scores"]
failure_probs: List[float] = st.session_state["failure_probs"]
prob_steps: List[int] = st.session_state["prob_steps"]
recovery_log: List[RecoveryEvent] = sim.recovery_log

df_full = sim.to_dataframe()
n_display = min(HISTORY_DISPLAY_STEPS, len(df_full))
df = df_full.tail(n_display).copy()

current_snapshot: MetricSnapshot = sim.current_snapshot()
current_health: float = health_scores[-1] if health_scores else 0.0
current_fp: float = failure_probs[-1] if failure_probs else 0.0
stability: float = engine.compute_stability_index(health_scores)
total_recoveries: int = len(recovery_log)

# Build probability trend DataFrame
df_prob = pd.DataFrame({"step": prob_steps, "failure_probability": failure_probs})

# Align health scores with step numbers
df["health_score"] = health_scores[-n_display:]
df["failure_probability"] = (
    failure_probs[-n_display:] if len(failure_probs) >= n_display
    else [0.0] * (n_display - len(failure_probs)) + failure_probs
)

# ---------------------------------------------------------------------------
# Row 1 – KPI cards
# ---------------------------------------------------------------------------

st.markdown('<div class="section-header">System Status</div>', unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(
        _kpi_html(
            "Health Score",
            f"{current_health:.2f}",
            "Composite weighted index",
            _health_class(current_health),
        ),
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        _kpi_html(
            "Failure Probability",
            f"{current_fp:.1%}",
            "Current step estimate",
            _prob_class(current_fp),
        ),
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        _kpi_html(
            "Stability Index",
            f"{stability:.2f}",
            "Mean health x low variance",
            _health_class(stability),
        ),
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        _kpi_html(
            "Recovery Actions",
            str(total_recoveries),
            "Triggered this run",
            "good" if total_recoveries == 0 else "warning",
        ),
        unsafe_allow_html=True,
    )

with col5:
    avail = current_snapshot.service_availability if current_snapshot else 0.0
    st.markdown(
        _kpi_html(
            "Service Availability",
            f"{avail:.1%}",
            "Last recorded value",
            "good" if avail >= 0.90 else ("warning" if avail >= 0.75 else "danger"),
        ),
        unsafe_allow_html=True,
    )

st.markdown("")

# ---------------------------------------------------------------------------
# Row 2 – Gauge + Failure probability trend
# ---------------------------------------------------------------------------

st.markdown('<div class="section-header">Health and Risk Overview</div>', unsafe_allow_html=True)

col_gauge, col_prob = st.columns([1, 2])

with col_gauge:
    fig_gauge = _gauge(current_health, "Current Health Score")
    st.plotly_chart(fig_gauge, use_container_width=True)

with col_prob:
    fig_prob = _line_chart(
        df,
        [("failure_probability", "Failure Probability", PALETTE["probability"]),
         ("health_score", "Health Score", PALETTE["health"])],
        "Failure Probability & Health Score Trend",
        yrange=[0.0, 1.05],
    )
    # Add recovery event markers
    if recovery_log:
        rec_steps = [e.step for e in recovery_log]
        rec_probs = [e.failure_probability for e in recovery_log]
        fig_prob.add_trace(go.Scatter(
            x=rec_steps,
            y=rec_probs,
            mode="markers",
            name="Recovery Action",
            marker=dict(color="#d29922", size=8, symbol="triangle-up"),
        ))
    st.plotly_chart(fig_prob, use_container_width=True)

# ---------------------------------------------------------------------------
# Row 3 – CPU + Memory
# ---------------------------------------------------------------------------

st.markdown('<div class="section-header">Compute Metrics</div>', unsafe_allow_html=True)

col_cpu, col_mem = st.columns(2)

with col_cpu:
    fig_cpu = _line_chart(
        df,
        [("cpu_usage", "CPU Usage", PALETTE["cpu"])],
        "CPU Utilisation",
        yrange=[0.0, 1.05],
    )
    st.plotly_chart(fig_cpu, use_container_width=True)

with col_mem:
    fig_mem = _line_chart(
        df,
        [("memory_usage", "Memory Usage", PALETTE["memory"])],
        "Memory Utilisation",
        yrange=[0.0, 1.05],
    )
    st.plotly_chart(fig_mem, use_container_width=True)

# ---------------------------------------------------------------------------
# Row 4 – Disk I/O + Network latency + Error rate
# ---------------------------------------------------------------------------

st.markdown('<div class="section-header">I/O and Network Metrics</div>', unsafe_allow_html=True)

col_disk, col_net, col_err = st.columns(3)

with col_disk:
    fig_disk = _line_chart(
        df,
        [("disk_io", "Disk I/O", PALETTE["disk"])],
        "Disk I/O",
        yrange=[0.0, 1.05],
    )
    st.plotly_chart(fig_disk, use_container_width=True)

with col_net:
    fig_net = _line_chart(
        df,
        [("network_latency", "Network Latency", PALETTE["network"])],
        "Network Latency",
        yrange=[0.0, 1.05],
    )
    st.plotly_chart(fig_net, use_container_width=True)

with col_err:
    fig_err = _line_chart(
        df,
        [("error_rate", "Error Rate", PALETTE["error"]),
         ("response_time", "Response Time", PALETTE["response"])],
        "Error Rate and Response Time",
        yrange=[0.0, 1.05],
    )
    st.plotly_chart(fig_err, use_container_width=True)

# ---------------------------------------------------------------------------
# Row 5 – Recovery log + Stability breakdown
# ---------------------------------------------------------------------------

st.markdown('<div class="section-header">Recovery and Stability</div>', unsafe_allow_html=True)

col_log, col_stab = st.columns([2, 1])

with col_log:
    st.markdown("**Recovery Action Log**", unsafe_allow_html=False)
    if not recovery_log:
        st.markdown(
            '<p style="color:#8b949e;font-size:13px;">No recovery actions were '
            "triggered during this simulation run.</p>",
            unsafe_allow_html=True,
        )
    else:
        for evt in reversed(recovery_log[-20:]):
            deltas = ", ".join(
                f"{k}: {'+' if v >= 0 else ''}{v:.3f}"
                for k, v in evt.impact_delta.items()
            )
            st.markdown(
                f'<div class="recovery-row">'
                f'  <span class="recovery-action">{evt.action_label}</span>'
                f'  <div class="recovery-meta">'
                f"    Step {evt.step} &nbsp;|&nbsp; "
                f"    Failure prob at trigger: {evt.failure_probability:.1%} &nbsp;|&nbsp; "
                f"    Cost: {evt.cost:.1f}"
                f"  </div>"
                f'  <div class="recovery-meta" style="margin-top:3px;">'
                f"    Impact delta: {deltas}"
                f"  </div>"
                f"</div>",
                unsafe_allow_html=True,
            )

with col_stab:
    # Rolling stability over time
    window = 10
    rolling_stability: List[float] = []
    for k in range(len(health_scores)):
        window_scores = health_scores[max(0, k - window + 1): k + 1]
        rolling_stability.append(engine.compute_stability_index(window_scores))

    df_stab = pd.DataFrame({
        "step": list(range(len(rolling_stability))),
        "stability": rolling_stability,
    }).tail(n_display)

    fig_stab = _line_chart(
        df_stab,
        [("stability", "Stability Index", "#79c0ff")],
        "Rolling Stability Index",
        yrange=[0.0, 1.05],
    )
    st.plotly_chart(fig_stab, use_container_width=True)

# ---------------------------------------------------------------------------
# Row 6 – Before/After recovery comparison
# ---------------------------------------------------------------------------

if recovery_log:
    st.markdown(
        '<div class="section-header">Before / After Recovery Comparison</div>',
        unsafe_allow_html=True,
    )

    # Pick the most recent recovery event for comparison
    latest_evt: RecoveryEvent = recovery_log[-1]
    # Snapshot immediately before the recovery
    before_step = max(0, latest_evt.step - 1)
    after_step = min(latest_evt.step + 5, len(df_full) - 1)

    history_list = sim.history
    snap_before = history_list[before_step] if before_step < len(history_list) else None
    snap_after = history_list[after_step] if after_step < len(history_list) else None

    if snap_before is not None and snap_after is not None:
        metrics = [
            "cpu_usage", "memory_usage", "disk_io",
            "network_latency", "error_rate", "response_time",
        ]
        before_vals = [getattr(snap_before, m) for m in metrics]
        after_vals = [getattr(snap_after, m) for m in metrics]

        fig_ba = go.Figure()
        fig_ba.add_trace(go.Bar(
            name=f"Before (step {before_step})",
            x=metrics,
            y=before_vals,
            marker_color="#f85149",
        ))
        fig_ba.add_trace(go.Bar(
            name=f"After (step {after_step})",
            x=metrics,
            y=after_vals,
            marker_color="#3fb950",
        ))
        fig_ba.update_layout(
            **PLOTLY_LAYOUT_BASE,
            barmode="group",
            height=CHART_HEIGHT,
            title=dict(
                text=f"Recovery: {latest_evt.action_label}",
                font=dict(size=13),
            ),
        )
        st.plotly_chart(fig_ba, use_container_width=True)

        # Summary text
        health_before = compute_health_score(snap_before)
        health_after = compute_health_score(snap_after)
        delta = health_after - health_before
        direction = "improved" if delta > 0 else "declined"
        st.markdown(
            f'<p style="font-size:13px;color:#8b949e;">'
            f"Health score {direction} by <strong>{abs(delta):.3f}</strong> points "
            f"following <strong>{latest_evt.action_label}</strong> "
            f"(cost index: {latest_evt.cost:.1f}).</p>",
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown(
    '<p style="font-size:11px;color:#8b949e;text-align:center;">'
    "AutoHeal AI – Intelligent Self-Healing Infrastructure Engine"
    "</p>",
    unsafe_allow_html=True,
)
