import math
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ---------- Page Config ----------
st.set_page_config(
    page_title="Lead Decay Forecaster",
    layout="wide",
    menu_items={
        "Get help": None,
        "Report a bug": None,
        "About": None,
    },
)


# ---------- Custom CSS (Financial Noir) ----------
CUSTOM_CSS = """
<style>
/* Overall app background */
body, .stApp {
    background-color: #0e1112;
    color: #ffffff;
}

/* Main app container */
.block-container {
    padding-top: 2rem;
}

/* Sidebar styling */
section[data-testid="stSidebar"] > div {
    background: #050708;
    border-right: 3px solid #D4AF37;
}

/* Sidebar widget labels */
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
section[data-testid="stSidebar"] small {
    color: #E0E0E0 !important;
}

/* Sidebar title visibility */
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #FFFFFF !important;
}

/* Global markdown/subtext readability */
[data-testid="stMarkdownContainer"] p,
small {
    color: #E0E0E0 !important;
}

/* Hide default Streamlit menu, header, footer */
#MainMenu {visibility: hidden;}
header[data-testid="stHeader"] {visibility: hidden;}
footer {visibility: hidden;}

/* Metric cards container */
.metric-row {
    display: flex;
    gap: 1.5rem;
    flex-wrap: wrap;
}

.metric-card {
    flex: 1 1 250px;
    background: #1a1e20;
    border-left: 6px solid #D4AF37;
    padding: 1rem 1.5rem;
    border-radius: 6px;
    color: #f5f5f5;
    box-shadow: 0 0 18px rgba(0, 0, 0, 0.7);
}

.metric-label {
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #D4AF37;
    margin-bottom: 0.25rem;
}

.metric-value {
    font-size: 1.6rem;
    font-weight: 700;
}

.metric-subtext {
    font-size: 0.8rem;
    color: #E0E0E0;
    margin-top: 0.25rem;
}

/* Section titles */
.section-title {
    font-size: 1.2rem;
    font-weight: 600;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: #D4AF37;
    margin-bottom: 0.25rem;
}

.section-subtitle {
    font-size: 0.9rem;
    color: #cccccc;
    margin-bottom: 1rem;
}

/* Insights panel */
.insights-panel {
    background: #050708;
    border-left: 4px solid #D4AF37;
    padding: 1.25rem 1.5rem;
    border-radius: 6px;
    box-shadow: 0 0 18px rgba(0, 0, 0, 0.8);
    color: #f7f7f7;
}

.insights-highlight {
    color: #D4AF37;
    font-weight: 600;
}

/* Plotly container tweaks */
div[data-testid="stPlotlyChart"] {
    background-color: transparent;
}

/* Udaller ecosystem link - bold, white on hover */
.udaller-protocol-link {
    color: #E0E0E0;
    font-weight: bold;
    text-decoration: none;
}
.udaller-protocol-link:hover {
    color: #FFFFFF !important;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------- Mathematical Engine ----------
DECAY_CONSTANT = 0.322  # per minute


def intent_probability(t_minutes: float) -> float:
    """P(t) = e^(-k * t). Returns probability in [0,1]."""
    return math.exp(-DECAY_CONSTANT * max(t_minutes, 0.0))


def cost_of_delay(
    adv: float, lcr: float, cpl: float, t_minutes: float
) -> float:
    """
    Cost = (ADV * LCR * (1 - D(t))) + CPL
    Where D(t) is the decay factor P(t).
    """
    d_t = intent_probability(t_minutes)
    return (adv * lcr * (1.0 - d_t)) + cpl


def opportunity_lost(
    adv: float, lcr: float, t_current: float, t_best: float = 1.0
) -> float:
    """
    Opportunity lost vs best-case (e.g. 1-minute response).
    Computed on expected revenue only, not CPL.
    """
    d_best = intent_probability(t_best)
    d_current = intent_probability(t_current)
    return max(0.0, adv * lcr * (d_best - d_current))


def build_decay_curve(max_minutes: int = 60, step: float = 0.5) -> pd.DataFrame:
    t_values = np.arange(0, max_minutes + step, step)
    probs = np.exp(-DECAY_CONSTANT * t_values)
    df = pd.DataFrame(
        {
            "minutes": t_values,
            "probability": probs * 100.0,  # percentage
        }
    )
    return df


def make_qualification_cliff_chart(
    df: pd.DataFrame, response_time: float
) -> go.Figure:
    fig = go.Figure()

    # Main decay line
    fig.add_trace(
        go.Scatter(
            x=df["minutes"],
            y=df["probability"],
            mode="lines",
            name="Qualification Probability",
            line=dict(color="#00AEEF", width=3),
            fill="tozeroy",
            fillcolor="rgba(0, 174, 239, 0.18)",
            hovertemplate="<b>%{x:.1f} Minutes</b><br>Probability: %{y:.1f}%<extra></extra>",
        )
    )

    # Marker for the user's response time
    response_prob = intent_probability(response_time) * 100.0
    fig.add_trace(
        go.Scatter(
            x=[response_time],
            y=[response_prob],
            mode="markers",
            name="Your Response Time",
            marker=dict(
                color="#D4AF37",
                size=14,
                symbol="diamond",
                line=dict(color="#ffffff", width=1),
            ),
            hovertemplate="<b>Your Avg: %{x} Min</b><br>Probability: %{y:.1f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text="THE QUALIFICATION CLIFF",
            x=0.01,
            xanchor="left",
            font=dict(color="#ffffff", size=20, family="Helvetica, Arial"),
        ),
        xaxis_title="Response Time (minutes)",
        yaxis_title="Qualification Probability (%)",
        font=dict(color="#E0E0E0"),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            range=[0, 60],
            gridcolor="rgba(255,255,255,0.05)",
            zeroline=False,
            fixedrange=True,
        ),
        yaxis=dict(
            range=[0, 100],
            gridcolor="rgba(255,255,255,0.05)",
            zeroline=False,
            fixedrange=True,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.01,
            font=dict(color="#E0E0E0"),
            itemclick=False,
            itemdoubleclick=False,
        ),
        margin=dict(l=50, r=50, t=100, b=50),
    )

    return fig


def format_currency(value: float) -> str:
    return f"${value:,.0f}"


def format_percent(value: float) -> str:
    return f"{value:,.1f}%"


# ---------- Sidebar: Parameters ----------
with st.sidebar:
    st.title("Parameters", anchor=False)

    adv = st.number_input(
        "Average Deal Value",
        min_value=0.0,
        value=15000.0,
        step=1000.0,
        format="%.2f",
        help="Average revenue from a closed deal.",
    )

    lcr_percent = st.number_input(
        "Lead-to-Close Rate (%)",
        min_value=0.0,
        max_value=100.0,
        value=25.0,
        step=1.0,
        help="Historical percentage of leads that eventually close.",
    )

    cpl = st.number_input(
        "Cost Per Lead",
        min_value=0.0,
        value=250.0,
        step=50.0,
        format="%.2f",
        help="Fully loaded acquisition cost per marketing-qualified lead.",
    )

    response_time = st.slider(
        "Avg Response Time (minutes)",
        min_value=1,
        max_value=120,
        value=10,
        step=1,
    )


# Convert to model-friendly values
lcr = lcr_percent / 100.0


# ---------- Main Dashboard (Action Hero Kernel - Udaller V8.4) ----------
st.markdown(
    """
<div style="text-align:center; margin-bottom:0.5rem;">
  <div style="color:#9e9e9e; font-size:0.9rem;">
    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 6px; margin-bottom: 2px;"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect><path d="M7 11V7a5 5 0 0 1 10 0v4"></path></svg>
    100% Private - Processed Securely &amp; Never Stored
  </div>
  <h1 style="color:#FFFFFF; font-size:2rem; font-weight:700; margin:0.5rem 0 0.35rem 0; letter-spacing:0.02em;">Calculate Your Pipeline Latency Tax</h1>
  <div style="color:#b0b0b0; font-size:1.1rem;">Identify the exact financial cost of your manual follow-up speed before the lead goes cold.</div>
</div>
""",
    unsafe_allow_html=True,
)

# Top row: Metric cards
decay_prob = intent_probability(response_time)
qualification_prob_pct = decay_prob * 100.0

current_cost = cost_of_delay(adv, lcr, cpl, response_time)
best_cost = cost_of_delay(adv, lcr, cpl, 1.0)
delta_cost = max(0.0, current_cost - best_cost)

opp_lost = opportunity_lost(adv, lcr, response_time, 1.0)

metric_cards_html = f"""
<div class="metric-row">
  <div class="metric-card">
    <div class="metric-label">Qualification Probability</div>
    <div class="metric-value">{format_percent(qualification_prob_pct)}</div>
    <div class="metric-subtext">Likelihood this lead still converts at your current response time.</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Cost of Delay</div>
    <div class="metric-value">{format_currency(current_cost)}</div>
    <div class="metric-subtext">Expected value leakage + acquisition cost at this response time.</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Opportunity Lost</div>
    <div class="metric-value">{format_currency(opp_lost)}</div>
    <div class="metric-subtext">Expected revenue sacrificed vs responding in 1 minute.</div>
  </div>
</div>
"""

st.markdown(metric_cards_html, unsafe_allow_html=True)


# Middle row: Qualification Cliff chart
st.markdown("<br>", unsafe_allow_html=True)
decay_df = build_decay_curve(max_minutes=60, step=0.5)
fig = make_qualification_cliff_chart(decay_df, response_time)
st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


# Bottom row: System Architect Insights
st.markdown("### 🔍 System Architect Insights")

best_prob = intent_probability(1.0) * 100.0
prob_drop = max(0.0, best_prob - qualification_prob_pct)

per_lead_cost_gap = delta_cost
per_lead_revenue_gap = opp_lost

insights_lines = [
    f"At **{response_time} minutes**, your qualification probability sits at "
    f"**{format_percent(qualification_prob_pct)}**, down from **{format_percent(best_prob)}** "
    f"if you responded in 1 minute.",
    f"That is a **{format_percent(prob_drop)}** drop in win probability per lead.",
    "",
    f"On a per-lead basis, you're burning roughly "
    f"<span class='insights-highlight'>{format_currency(per_lead_revenue_gap)}</span> "
    f"in expected revenue and taking on an incremental "
    f"<span class='insights-highlight'>{format_currency(per_lead_cost_gap)}</span> "
    f"in delay-driven value leakage compared with a 1-minute response.",
]

st.markdown(
    f"""
<div class="insights-panel">
  <p>{"<br>".join(insights_lines)}</p>
  <p>
    Scale that across just 100 inbound leads a month and you're staring at
    <span class="insights-highlight">{format_currency(per_lead_revenue_gap * 100)}</span>
    in silent pipeline erosion — purely due to response latency.
  </p>
  <p>
    The architecture verdict: shave minutes off your response time
    and the cliff turns back into a ramp. Treat speed-to-lead as a
    first-class SLO, not a convenience metric.
  </p>
</div>
""",
    unsafe_allow_html=True,
)

# ---------- Zeigarnik Conversion Bridge (Udaller V8.4) ----------
st.markdown(
    """
<div style="
  max-width:780px;
  margin:2rem auto 1.25rem auto;
  padding:1.25rem 1.5rem;
  border-radius:8px;
  border:1px solid #343a40;
  background:#161B22;
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:1rem;
  box-shadow:0 4px 20px rgba(0, 0, 0, 0.5);
  font-size:0.9rem;
  color:#E0E0E0;
">
  <div style="display:flex; align-items:flex-start; gap:0.75rem;">
    <div style="margin-top:2px;">
      <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#F4C430" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>
    </div>
    <div>
      <h3 style="color:#FFFFFF; font-size:1.1rem; font-weight:700; margin:0 0 0.35rem 0;">Stop burning inbound leads.</h3>
      <div style="font-size:0.85rem; color:#b0b0b0; line-height:1.4;">
        You are paying a massive latency tax. Join the newsletter to unlock The Vault and get the 3-step Automated Routing Blueprint to drop your response time under 60 seconds.
      </div>
    </div>
  </div>
  <div style="flex-shrink:0;">
    <a href="https://udallerprotocol.com/subscribe" target="_blank" style="
      display:inline-block;
      padding:0.5rem 1rem;
      border-radius:6px;
      border:none;
      color:#0e1112;
      background:#F4C430;
      font-size:0.85rem;
      font-weight:600;
      letter-spacing:0.04em;
      text-decoration:none;
      white-space:nowrap;
    ">
      Get the Blueprint ↗
    </a>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ---------- Legal & Ecosystem Footer (Udaller V8.4) ----------
st.markdown(
    """
<div style="text-align:center; color:#9ca3af; font-size:0.85rem; margin-top:0.5rem; margin-bottom:0.75rem;">
  The Lead Decay Forecaster provides probability analysis based on standard industry models for educational purposes only.
  It is not a substitute for financial or operational counsel. Always verify your own pipeline metrics.
</div>
<div style="text-align:center; color:#9ca3af; font-size:0.9rem;">
  This tool is part of the <a href="https://udaller.one" class="udaller-protocol-link">Udaller</a> ecosystem. Build your machine at <a href="https://udallerprotocol.com" class="udaller-protocol-link">The Udaller Protocol</a>.
</div>
""",
    unsafe_allow_html=True,
)

