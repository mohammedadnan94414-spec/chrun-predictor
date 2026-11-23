"""
ChurnGuard - Telecom Customer Churn Prediction Dashboard
========================================================
Run with:   streamlit run frontend.py

Talks to your existing FastAPI backend:
    POST <API URL>  with the 10 feature fields  ->  {"churn_probability": ..., "prediction": ...}

Nothing in this file changes your model or backend. Every number shown on the
dashboard comes from predictions your API actually returned.
"""

from __future__ import annotations

import datetime as dt
import html as html_lib
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# =============================================================================
# 0. PAGE CONFIG (must be the first Streamlit call)
# =============================================================================
st.set_page_config(
    page_title="ChurnGuard | Churn Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# 1. CONFIGURATION  (edit these if you need to)
# =============================================================================
DEFAULT_API_URL = "http://127.0.0.1:8000/predict"
DEFAULT_TIMEOUT = 10
HISTORY_FILE = Path(__file__).with_name("prediction_history.csv")

PAGES = [
    "Dashboard Overview",
    "Predict Churn",
    "Customer Analytics",
    "Prediction History",
    "Model Information",
    "Settings",
]
PAGE_ICONS = {
    "Dashboard Overview": "📊",
    "Predict Churn": "🔮",
    "Customer Analytics": "📈",
    "Prediction History": "🕘",
    "Model Information": "🧠",
    "Settings": "⚙️",
}

# Colors
PURPLE, BLUE, CYAN = "#8b5cf6", "#3b82f6", "#22d3ee"
RISK_COLORS = {"Low": "#22c55e", "Medium": "#f59e0b", "High": "#ef4444"}
OUTCOME_COLORS = {"Churn": "#f43f5e", "Stay": "#22d3ee"}

# The 10 input features. "key" is the EXACT name your API expects.
FEATURES = [
    dict(key="AccountWeeks", label="Account weeks", kind="int",
         help="How many weeks the customer account has been active."),
    dict(key="ContractRenewal", label="Contract renewal", kind="bool",
         help="Whether the customer recently renewed their contract."),
    dict(key="DataPlan", label="Data plan", kind="bool",
         help="Whether the customer has a data plan."),
    dict(key="DataUsage", label="Data usage", kind="float", step=0.1,
         help="Monthly data usage. Use the same units as your training data (GB in the common telecom dataset)."),
    dict(key="CustServCalls", label="Customer service calls", kind="int",
         help="Number of calls the customer made to customer service."),
    dict(key="DayMins", label="Day minutes", kind="float", step=1.0,
         help="Average daytime call minutes per month."),
    dict(key="DayCalls", label="Day calls", kind="int",
         help="Average number of daytime calls per month."),
    dict(key="MonthlyCharge", label="Monthly charge", kind="float", step=0.5,
         help="Average monthly bill."),
    dict(key="OverageFee", label="Overage fee", kind="float", step=0.5,
         help="Overage (extra usage) fees charged."),
    dict(key="RoamMins", label="Roaming minutes", kind="float", step=1.0,
         help="Average roaming call minutes per month."),
]
FEATURE_KEYS = [f["key"] for f in FEATURES]
FEATURE_BY_KEY = {f["key"]: f for f in FEATURES}
HISTORY_COLUMNS = ["Timestamp", "ChurnProbability", "Prediction"] + FEATURE_KEYS

# Same defaults as your original form
DEFAULTS = dict(
    AccountWeeks=100, ContractRenewal=1, DataPlan=1, DataUsage=0.0,
    CustServCalls=4, DayMins=250.5, DayCalls=100, MonthlyCharge=80.0,
    OverageFee=25.0, RoamMins=15.0,
)

# Illustrative INPUT examples (these are just form values, not predictions)
PRESETS = {
    "Default values": DEFAULTS,
    "Example A - long tenure, renewed contract": dict(
        AccountWeeks=150, ContractRenewal=1, DataPlan=1, DataUsage=2.5,
        CustServCalls=1, DayMins=180.0, DayCalls=90, MonthlyCharge=55.0,
        OverageFee=3.0, RoamMins=8.0,
    ),
    "Example B - short tenure, many support calls": dict(
        AccountWeeks=40, ContractRenewal=0, DataPlan=0, DataUsage=0.0,
        CustServCalls=5, DayMins=300.0, DayCalls=110, MonthlyCharge=95.0,
        OverageFee=30.0, RoamMins=18.0,
    ),
}

DEFAULT_SETTINGS = dict(
    api_url=DEFAULT_API_URL,
    timeout=DEFAULT_TIMEOUT,
    model_name="Not specified",
    low_max=30,   # below this % = Low risk
    high_min=60,  # at/above this % = High risk
)

ICONS = {  # simple line icons (Lucide style)
    "activity": '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
    "alert": '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
    "check": '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>',
    "percent": '<line x1="19" y1="5" x2="5" y2="19"/><circle cx="6.5" cy="6.5" r="2.5"/><circle cx="17.5" cy="17.5" r="2.5"/>',
    "server": '<rect x="2" y="2" width="20" height="8" rx="2" ry="2"/><rect x="2" y="14" width="20" height="8" rx="2" ry="2"/><line x1="6" y1="6" x2="6.01" y2="6"/><line x1="6" y1="18" x2="6.01" y2="18"/>',
}

# =============================================================================
# 2. STYLES
# =============================================================================
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
@property --p {syntax: '<number>'; inherits: false; initial-value: 0;}

html, body, [class*="css"], .stApp {font-family: 'Inter', sans-serif;}
.stApp {
  background:
    radial-gradient(1100px 560px at 8% -10%, rgba(99,102,241,.28), transparent 60%),
    radial-gradient(900px 480px at 100% 0%, rgba(34,211,238,.14), transparent 55%),
    #0b1020;
  color: #e6eaf5;
}
#MainMenu, footer, [data-testid="stDecoration"], .stDeployButton, [data-testid="stAppDeployButton"] {display: none !important;}
header[data-testid="stHeader"] {background: transparent;}
.block-container {padding-top: 2rem; padding-bottom: 4rem; max-width: 1280px;}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0d1330 0%, #090e24 100%);
  border-right: 1px solid rgba(255,255,255,.06);
}
section[data-testid="stSidebar"] div[role="radiogroup"] {gap: .3rem;}
section[data-testid="stSidebar"] div[role="radiogroup"] > label {
  width: 100%; padding: .65rem .9rem; border-radius: 12px; cursor: pointer;
  border: 1px solid transparent; transition: all .2s ease;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {background: rgba(139,92,246,.12);}
section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child {display: none;}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
  background: linear-gradient(90deg, rgba(99,102,241,.30), rgba(34,211,238,.10));
  border-color: rgba(139,92,246,.45);
}
.brand {display: flex; align-items: center; gap: .75rem; margin: .4rem 0 1.4rem;}
.brand-mark {
  width: 42px; height: 42px; border-radius: 13px; display: grid; place-items: center;
  background: linear-gradient(135deg, #6366f1, #8b5cf6 55%, #22d3ee); color: #fff;
  box-shadow: 0 10px 28px -10px rgba(99,102,241,.9);
}
.brand-name {font-size: 1.2rem; font-weight: 800; letter-spacing: -.02em; line-height: 1.1;}
.brand-tag {font-size: .72rem; color: #8f9bb8; font-weight: 500;}
.nav-label {font-size: .7rem; text-transform: uppercase; letter-spacing: .12em; color: #6b7794; font-weight: 700; margin: 0 0 .5rem .3rem;}
.side-status {margin-top: 1.6rem; padding: .9rem 1rem; border-radius: 14px; background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.08);}
.side-status .t {font-size: .7rem; text-transform: uppercase; letter-spacing: .1em; color: #8f9bb8; font-weight: 700;}
.side-status .s {display: flex; align-items: center; gap: .5rem; font-weight: 600; margin: .35rem 0 .2rem;}
.side-status .u {font-size: .72rem; color: #8f9bb8; word-break: break-all;}

/* ---------- Page header ---------- */
.page-head {display: flex; justify-content: space-between; align-items: flex-end; flex-wrap: wrap; gap: 1rem; margin-bottom: 1.5rem;}
.page-title {
  font-size: 1.9rem; font-weight: 800; letter-spacing: -.03em; line-height: 1.15;
  background: linear-gradient(90deg, #fff, #a5b4fc 60%, #67e8f9);
  -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
}
.page-sub {color: #8f9bb8; margin-top: .3rem; font-size: .95rem;}
.pill {
  display: inline-flex; align-items: center; gap: .5rem; padding: .4rem .9rem; border-radius: 999px;
  font-size: .8rem; font-weight: 600; background: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.1);
}
.dot {width: 8px; height: 8px; border-radius: 50%; display: inline-block;}
.dot.on {background: #22c55e; animation: pulse 2s infinite;}
.dot.off {background: #f59e0b;}
@keyframes pulse {0% {box-shadow: 0 0 0 0 rgba(34,197,94,.6);} 70% {box-shadow: 0 0 0 8px rgba(34,197,94,0);} 100% {box-shadow: 0 0 0 0 rgba(34,197,94,0);}}

/* ---------- Cards ---------- */
.glass {
  background: rgba(20,28,58,.55); backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
  border: 1px solid rgba(255,255,255,.08); border-radius: 18px; padding: 1.2rem 1.4rem;
  transition: transform .25s ease, border-color .25s ease, box-shadow .25s ease;
}
.glass:hover {transform: translateY(-3px); border-color: rgba(139,92,246,.45); box-shadow: 0 18px 40px -22px rgba(99,102,241,.75);}
.kpi-grid {display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 1.5rem;}
.kpi {position: relative; overflow: hidden;}
.kpi::after {content: ""; position: absolute; right: -40px; bottom: -70px; width: 170px; height: 170px; border-radius: 50%; background: radial-gradient(circle, var(--a), transparent 70%); opacity: .2;}
.kpi .top {display: flex; align-items: center; gap: .65rem;}
.kpi .icon {width: 36px; height: 36px; border-radius: 11px; display: grid; place-items: center; color: var(--a); background: rgba(255,255,255,.06);}
.kpi .label {font-size: .74rem; font-weight: 700; text-transform: uppercase; letter-spacing: .07em; color: #8f9bb8;}
.kpi .value {font-size: 2rem; font-weight: 800; letter-spacing: -.02em; margin-top: .7rem; line-height: 1.1;}
.kpi .sub {color: #8f9bb8; font-size: .82rem; margin-top: .25rem;}
.card-title {font-size: 1rem; font-weight: 700; margin-bottom: .6rem;}
.muted {color: #8f9bb8; font-size: .9rem;}
.section-title {font-size: .75rem; text-transform: uppercase; letter-spacing: .1em; color: #8f9bb8; font-weight: 700; margin: .2rem 0 .6rem;}
.info-grid {display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; margin-bottom: 1.2rem;}
.info-grid .k {font-size: .72rem; text-transform: uppercase; letter-spacing: .08em; color: #8f9bb8; font-weight: 700;}
.info-grid .v {font-size: 1.05rem; font-weight: 700; margin-top: .3rem; word-break: break-all;}
.empty {text-align: center; padding: 2.5rem 1rem;}
.empty .big {font-size: 2.2rem;}

/* ---------- Result ---------- */
.result {text-align: center;}
.result-top {font-size: .75rem; color: #8f9bb8; text-transform: uppercase; letter-spacing: .1em; font-weight: 700; margin-bottom: 1rem;}
.ring {
  --p: 0; --c: #22c55e; width: 190px; height: 190px; border-radius: 50%; margin: 0 auto 1.1rem;
  display: grid; place-items: center; animation: ringfill 1.3s ease-out both;
  background:
    radial-gradient(closest-side, #0f1631 79%, transparent 80% 100%),
    conic-gradient(var(--c) calc(var(--p) * 1%), rgba(148,163,184,.16) 0);
  filter: drop-shadow(0 0 18px var(--c));
}
.ring-inner span {display: block; font-size: 2.3rem; font-weight: 800; letter-spacing: -.02em;}
.ring-inner small {display: block; font-size: .72rem; color: #8f9bb8; margin-top: -.15rem;}
.badges {display: flex; justify-content: center; gap: .6rem; flex-wrap: wrap;}
.outcome, .badge {
  display: inline-block; padding: .4rem 1rem; border-radius: 999px; font-weight: 800; font-size: .85rem; letter-spacing: .05em;
  color: var(--c); background: rgba(255,255,255,.05); border: 1px solid var(--c);
}
.badge {font-weight: 700; letter-spacing: .02em;}
.chip {display: inline-block; margin: .2rem .35rem .2rem 0; padding: .35rem .8rem; border-radius: 10px; font-size: .82rem; font-weight: 500; background: rgba(139,92,246,.14); border: 1px solid rgba(139,92,246,.3);}
.actions {margin: .4rem 0 0; padding-left: 1.1rem; color: #cbd5ef; font-size: .92rem; line-height: 1.7;}

/* ---------- Streamlit widgets ---------- */
div[data-testid="stForm"] {background: rgba(20,28,58,.55); border: 1px solid rgba(255,255,255,.08); border-radius: 18px; padding: 1.4rem;}
div[data-testid="stPlotlyChart"] {background: rgba(20,28,58,.55); border: 1px solid rgba(255,255,255,.08); border-radius: 18px; padding: .4rem .6rem; transition: border-color .25s ease;}
div[data-testid="stPlotlyChart"]:hover {border-color: rgba(139,92,246,.45);}
div[data-testid="stDataFrame"] {border-radius: 14px; overflow: hidden; border: 1px solid rgba(255,255,255,.08);}
div[data-testid="stWidgetLabel"] p {color: #b6c0da; font-weight: 500;}
div[data-baseweb="input"], div[data-baseweb="select"] > div {border-radius: 10px;}
.stButton > button, div[data-testid="stFormSubmitButton"] > button, .stDownloadButton > button {
  width: 100%; border: 0; border-radius: 12px; padding: .7rem 1rem; font-weight: 700; color: #fff;
  background: linear-gradient(135deg, #6366f1, #8b5cf6 55%, #22d3ee); background-size: 150% 150%;
  transition: transform .15s ease, box-shadow .15s ease, background-position .3s ease;
}
.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover, .stDownloadButton > button:hover {
  transform: translateY(-2px); background-position: 100% 0; color: #fff; box-shadow: 0 12px 28px -12px rgba(99,102,241,.9);
}
.stButton > button:disabled {opacity: .4;}

/* ---------- Responsive ---------- */
@media (max-width: 768px) {
  .block-container {padding: 1rem 1rem 3rem;}
  .page-title {font-size: 1.5rem;}
  .kpi .value {font-size: 1.6rem;}
  .ring {width: 160px; height: 160px;}
}
"""


def render_html(markup: str) -> None:
    """Render HTML safely: strips indentation and blank lines so Markdown never treats it as code."""
    cleaned = "\n".join(line.strip() for line in markup.splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


def inject_css() -> None:
    render_html(f"<style>{CSS}</style>")


def esc(text) -> str:
    return html_lib.escape(str(text))


def svg(name: str) -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="19" height="19" viewBox="0 0 24 24" fill="none" '
        'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        f"{ICONS[name]}</svg>"
    )


def stretch(fn, *args, **kwargs):
    """Call a Streamlit element full-width on both old and new Streamlit versions."""
    try:
        return fn(*args, use_container_width=True, **kwargs)
    except Exception:
        return fn(*args, width="stretch", **kwargs)


# =============================================================================
# 3. STATE, SETTINGS AND HISTORY STORAGE
# =============================================================================
def load_history() -> list[dict]:
    """Load saved predictions from disk (so history survives a browser refresh)."""
    if not HISTORY_FILE.exists():
        return []
    try:
        df = pd.read_csv(HISTORY_FILE)
        if not set(HISTORY_COLUMNS).issubset(df.columns):
            return []
        return df[HISTORY_COLUMNS].to_dict("records")
    except Exception:
        return []


def save_history(history: list[dict]) -> None:
    try:
        pd.DataFrame(history, columns=HISTORY_COLUMNS).to_csv(HISTORY_FILE, index=False)
    except OSError:
        pass  # history still lives in session state if the disk is not writable


def init_state() -> None:
    ss = st.session_state
    if "settings" not in ss:
        ss["settings"] = dict(DEFAULT_SETTINGS)
    if "history" not in ss:
        ss["history"] = load_history()
    ss.setdefault("last_result", None)
    ss.setdefault("nav", PAGES[0])


def settings() -> dict:
    return st.session_state["settings"]


def history_df() -> pd.DataFrame:
    df = pd.DataFrame(st.session_state["history"], columns=HISTORY_COLUMNS)
    if df.empty:
        return df
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["Outcome"] = df["Prediction"].map({1: "Churn", 0: "Stay"})
    df["Risk"] = df["ChurnProbability"].map(risk_label)
    return df


def display_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Timestamp"] = out["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out["Churn %"] = out["ChurnProbability"].round(2)
    return out[["Timestamp", "Outcome", "Churn %", "Risk"] + FEATURE_KEYS]


def go_to(page: str) -> None:
    st.session_state["nav"] = page


# =============================================================================
# 4. API LAYER
# =============================================================================
class ApiError(Exception):
    """Raised with a friendly message when the prediction API fails."""


def predict_api(payload: dict) -> tuple[float, int]:
    cfg = settings()
    try:
        response = requests.post(cfg["api_url"], json=payload, timeout=cfg["timeout"])
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise ApiError(f"Cannot connect to {cfg['api_url']}. Is your FastAPI server running?")
    except requests.exceptions.Timeout:
        raise ApiError(f"The API did not respond within {cfg['timeout']} seconds.")
    except requests.exceptions.HTTPError as err:
        body = err.response.text[:200] if err.response is not None else ""
        code = err.response.status_code if err.response is not None else "?"
        raise ApiError(f"The API returned HTTP {code}. {body}")
    except requests.exceptions.RequestException as err:
        raise ApiError(f"Request failed: {err}")

    try:
        data = response.json()
    except ValueError:
        raise ApiError("The API response was not valid JSON.")

    if not isinstance(data, dict) or "churn_probability" not in data or "prediction" not in data:
        raise ApiError("The API response is missing 'churn_probability' or 'prediction'.")

    try:
        prob = float(data["churn_probability"])
        pred = int(data["prediction"])
    except (TypeError, ValueError):
        raise ApiError("The API returned values that are not numbers.")

    if not 0 <= prob <= 100:
        raise ApiError(f"churn_probability ({prob}) is outside 0-100. The dashboard expects a percentage.")
    if pred not in (0, 1):
        raise ApiError(f"prediction ({pred}) must be 0 or 1.")
    return prob, pred


@st.cache_data(ttl=15, show_spinner=False)
def check_api(url: str) -> bool:
    """True if the FastAPI server answers. Cached for 15 seconds."""
    base = url.rsplit("/predict", 1)[0].rstrip("/")
    for path in ("/health", "/docs", ""):
        try:
            if requests.get(base + path, timeout=2).status_code < 500:
                return True
        except requests.exceptions.RequestException:
            return False
    return False


# =============================================================================
# 5. BUSINESS HELPERS
# =============================================================================
def risk_label(prob: float) -> str:
    cfg = settings()
    if prob < cfg["low_max"]:
        return "Low"
    if prob < cfg["high_min"]:
        return "Medium"
    return "High"


def validate_inputs(c: dict) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
    if c["DayMins"] > 0 and c["DayCalls"] == 0:
        errors.append("Day minutes are above 0 but day calls is 0. Please check these two values.")
    if c["DayCalls"] > 0 and c["DayMins"] == 0:
        warnings.append("Day calls are above 0 but day minutes is 0.")
    if c["DataPlan"] == 0 and c["DataUsage"] > 0:
        warnings.append("Data usage is above 0 although the customer has no data plan.")
    if c["MonthlyCharge"] == 0:
        warnings.append("Monthly charge is 0.")
    if c["AccountWeeks"] == 0:
        warnings.append("Account weeks is 0 (brand new account).")
    return errors, warnings


def risk_signals(c: dict) -> list[tuple[str, str]]:
    """
    Simple rule-of-thumb flags (label, suggested action) based ONLY on the inputs.
    These are NOT model explanations. Adjust the thresholds to match your business.
    """
    out = []
    if c["CustServCalls"] >= 4:
        out.append(("📞 Frequent support calls", "Have a senior agent proactively follow up on open issues."))
    if c["ContractRenewal"] == 0:
        out.append(("📄 Contract not renewed", "Offer a renewal incentive or a contract upgrade."))
    if c["DataPlan"] == 0:
        out.append(("📶 No data plan", "Promote a data plan that fits the customer's usage."))
    if c["OverageFee"] >= 20:
        out.append(("💸 High overage fees", "Review plan fit and suggest a larger allowance to reduce overage fees."))
    if c["MonthlyCharge"] >= 90:
        out.append(("💳 High monthly charge", "Consider a loyalty discount or a bundled offer."))
    if c["DayMins"] >= 280:
        out.append(("☎️ Heavy daytime usage", "Check whether an unlimited-minutes plan would suit them better."))
    if c["RoamMins"] >= 15:
        out.append(("🌍 High roaming usage", "Offer a roaming add-on or international pack."))
    if c["AccountWeeks"] < 50:
        out.append(("🆕 Short account tenure", "Run an early-life check-in to make sure they are satisfied."))
    return out


def general_action(risk: str, pred: int) -> str:
    if risk == "High" or pred == 1:
        return "Escalate to the retention team for personal outreach."
    if risk == "Medium":
        return "Monitor this customer and consider a targeted offer."
    return "No action needed. Continue standard engagement."


# =============================================================================
# 6. REUSABLE UI COMPONENTS
# =============================================================================
def page_header(title: str, subtitle: str) -> None:
    online = check_api(settings()["api_url"])
    dot, text = ("on", "API online") if online else ("off", "API unreachable")
    render_html(f"""
    <div class="page-head">
      <div><div class="page-title">{esc(title)}</div><div class="page-sub">{esc(subtitle)}</div></div>
      <div class="pill"><span class="dot {dot}"></span>{text}</div>
    </div>
    """)


def kpi_card(label: str, value, sub: str, icon: str, accent: str) -> str:
    return (
        f'<div class="glass kpi" style="--a:{accent}"><div class="top"><div class="icon">{svg(icon)}</div>'
        f'<div class="label">{esc(label)}</div></div><div class="value">{value}</div>'
        f'<div class="sub">{esc(sub)}</div></div>'
    )


def kpi_grid(cards: list[str]) -> None:
    render_html(f'<div class="kpi-grid">{"".join(cards)}</div>')


def empty_state(message: str, key: str) -> None:
    render_html(f"""
    <div class="glass empty"><div class="big">📭</div>
    <div class="card-title">No predictions yet</div><div class="muted">{esc(message)}</div></div>
    """)
    st.button("Go to Predict Churn", on_click=go_to, args=("Predict Churn",), key=key)


def style_fig(fig: go.Figure, title: str, height: int = 340) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0.02, font=dict(size=15, color="#e6eaf5")),
        height=height,
        margin=dict(l=20, r=20, t=55, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#a8b3cf"),
        legend=dict(orientation="h", y=-0.15),
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,.12)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(148,163,184,.12)", zeroline=False)
    return fig


def show_chart(fig: go.Figure) -> None:
    stretch(st.plotly_chart, fig, config={"displayModeBar": False})


def show_table(df: pd.DataFrame) -> None:
    stretch(st.dataframe, df, hide_index=True)


# ----- charts (all built from real prediction history) -----
def fig_outcome_donut(df: pd.DataFrame) -> go.Figure:
    counts = df["Outcome"].value_counts()
    fig = go.Figure(go.Pie(
        labels=list(counts.index), values=list(counts.values), hole=0.68, sort=False,
        marker=dict(colors=[OUTCOME_COLORS[x] for x in counts.index], line=dict(color="#0b1020", width=3)),
        textinfo="percent",
    ))
    fig.add_annotation(text=f"<b>{len(df)}</b><br>predictions", showarrow=False,
                       font=dict(size=15, color="#e6eaf5"))
    return style_fig(fig, "Churn vs Stay")


def fig_risk_bar(df: pd.DataFrame) -> go.Figure:
    order = ["Low", "Medium", "High"]
    counts = df["Risk"].value_counts().reindex(order).fillna(0).astype(int)
    fig = go.Figure(go.Bar(
        x=order, y=list(counts.values), text=list(counts.values), textposition="outside",
        marker=dict(color=[RISK_COLORS[r] for r in order]),
    ))
    fig.update_yaxes(title="Customers", rangemode="tozero")
    return style_fig(fig, "Risk category distribution")


def fig_trend(df: pd.DataFrame) -> go.Figure:
    d = df.sort_values("Timestamp").reset_index(drop=True)
    fig = go.Figure(go.Scatter(
        x=list(d.index + 1), y=list(d["ChurnProbability"]), mode="lines+markers",
        line=dict(color=CYAN, width=3), marker=dict(size=8, color=PURPLE),
        fill="tozeroy", fillcolor="rgba(139,92,246,.15)",
        customdata=list(d["Timestamp"].dt.strftime("%d %b %H:%M:%S")),
        hovertemplate="Prediction #%{x}<br>%{y:.1f}%<br>%{customdata}<extra></extra>",
    ))
    fig.add_hline(y=float(d["ChurnProbability"].mean()), line_dash="dot", line_color="#94a3b8",
                  annotation_text="Average", annotation_font_color="#94a3b8")
    fig.update_yaxes(range=[0, 100], title="Churn probability (%)")
    fig.update_xaxes(title="Prediction number")
    return style_fig(fig, "Prediction trend")


def fig_hist(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(df, x="ChurnProbability", nbins=10, range_x=[0, 100],
                       color_discrete_sequence=[PURPLE])
    fig.update_traces(marker_line_color="#0b1020", marker_line_width=2)
    fig.update_xaxes(title="Churn probability (%)")
    fig.update_yaxes(title="Customers")
    return style_fig(fig, "Churn probability distribution")


def fig_scatter(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(df, x="DayMins", y="MonthlyCharge", color="Outcome",
                     color_discrete_map=OUTCOME_COLORS, hover_data=["ChurnProbability"])
    fig.update_traces(marker=dict(size=11, line=dict(color="#0b1020", width=1)))
    return style_fig(fig, "Day minutes vs monthly charge")


def fig_box(df: pd.DataFrame, feature: str) -> go.Figure:
    fig = px.box(df, x="Outcome", y=feature, color="Outcome", points="all",
                 color_discrete_map=OUTCOME_COLORS)
    fig.update_layout(showlegend=False)
    return style_fig(fig, f"{feature} by predicted outcome")


# =============================================================================
# 7. SIDEBAR
# =============================================================================
def render_sidebar() -> str:
    with st.sidebar:
        render_html(f"""
        <div class="brand">
          <div class="brand-mark">{svg("activity")}</div>
          <div><div class="brand-name">ChurnGuard</div><div class="brand-tag">Churn intelligence platform</div></div>
        </div>
        <div class="nav-label">Navigation</div>
        """)
        st.radio("Navigation", PAGES, key="nav", label_visibility="collapsed",
                 format_func=lambda p: f"{PAGE_ICONS[p]}  {p}")

        online = check_api(settings()["api_url"])
        dot, text = ("on", "Connected") if online else ("off", "Not connected")
        render_html(f"""
        <div class="side-status">
          <div class="t">API connection</div>
          <div class="s"><span class="dot {dot}"></span>{text}</div>
          <div class="u">{esc(settings()["api_url"])}</div>
        </div>
        """)
    return st.session_state["nav"]


# =============================================================================
# 8. PAGES
# =============================================================================
def page_overview() -> None:
    page_header("Dashboard Overview", "A live summary of the predictions made through ChurnGuard.")
    df = history_df()
    total = len(df)
    churn = int((df["Prediction"] == 1).sum()) if total else 0
    stay = total - churn
    online = check_api(settings()["api_url"])

    kpi_grid([
        kpi_card("Total predictions", total, "All saved predictions", "activity", PURPLE),
        kpi_card("Predicted to churn", churn,
                 f"{churn / total:.0%} of predictions" if total else "No data yet", "alert", "#f43f5e"),
        kpi_card("Likely to stay", stay,
                 f"{stay / total:.0%} of predictions" if total else "No data yet", "check", CYAN),
        kpi_card("Avg churn probability",
                 f"{df['ChurnProbability'].mean():.1f}%" if total else "-",
                 "Mean of all predictions" if total else "No data yet", "percent", BLUE),
        kpi_card("Model / API status", "Online" if online else "Offline",
                 "Prediction API reachable" if online else "Start your FastAPI server", "server",
                 "#22c55e" if online else "#f59e0b"),
    ])

    if df.empty:
        empty_state("Run your first prediction to see charts and statistics here.", "goto_overview")
        return

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        show_chart(fig_outcome_donut(df))
    with c2:
        show_chart(fig_trend(df))

    st.markdown('<div class="section-title">Most recent predictions</div>', unsafe_allow_html=True)
    recent = display_frame(df.sort_values("Timestamp", ascending=False).head(5))
    show_table(recent[["Timestamp", "Outcome", "Churn %", "Risk", "AccountWeeks", "MonthlyCharge", "CustServCalls"]])


def feature_input(container, key: str) -> None:
    f = FEATURE_BY_KEY[key]
    state_key = f"f_{key}"
    if f["kind"] == "int":
        container.number_input(f["label"], min_value=0, step=1, key=state_key, help=f["help"])
    elif f["kind"] == "float":
        container.number_input(f["label"], min_value=0.0, step=f.get("step", 1.0),
                               format="%.2f", key=state_key, help=f["help"])
    else:
        container.selectbox(f["label"], [1, 0], key=state_key, help=f["help"],
                            format_func=lambda x: "Yes" if x == 1 else "No")


def set_form_values(values: dict) -> None:
    for f in FEATURES:
        v = values[f["key"]]
        st.session_state[f"f_{f['key']}"] = float(v) if f["kind"] == "float" else int(v)


def on_load_preset() -> None:
    set_form_values(PRESETS[st.session_state["preset_choice"]])


def on_reset_form() -> None:
    set_form_values(DEFAULTS)
    st.session_state["last_result"] = None


def result_card(res: dict) -> None:
    prob, pred = res["prob"], res["pred"]
    risk = risk_label(prob)
    color = RISK_COLORS[risk]
    outcome = "CHURN" if pred == 1 else "STAY"
    ocolor = OUTCOME_COLORS["Churn" if pred == 1 else "Stay"]

    render_html(f"""
    <style>@keyframes ringfill {{from {{--p: 0;}} to {{--p: {prob:.2f};}}}}</style>
    <div class="glass result">
      <div class="result-top">Prediction result &middot; {esc(res["time"])}</div>
      <div class="ring" style="--p:{prob:.2f};--c:{color}">
        <div class="ring-inner"><span style="color:{color}">{prob:.1f}%</span><small>churn probability</small></div>
      </div>
      <div class="badges">
        <span class="outcome" style="--c:{ocolor}">{outcome}</span>
        <span class="badge" style="--c:{color}">{risk} risk</span>
      </div>
    </div>
    """)
    st.caption(
        "The outcome is the model's prediction. The risk level is based on the probability bands in Settings "
        f"(Low < {settings()['low_max']}%, High >= {settings()['high_min']}%)."
    )

    signals = risk_signals(res["payload"])
    chips = "".join(f'<span class="chip">{s}</span>' for s, _ in signals) or \
        '<span class="chip">No rule-based risk signals detected</span>'
    actions = [a for _, a in signals] or [general_action(risk, pred)]
    if signals and (risk == "High" or pred == 1):
        actions.insert(0, "Escalate to the retention team for personal outreach.")
    items = "".join(f"<li>{esc(a)}</li>" for a in actions)

    render_html(f"""
    <div class="glass" style="margin-top:1rem">
      <div class="card-title">Customer risk signals</div>
      <div>{chips}</div>
      <div class="card-title" style="margin-top:1rem">Recommended retention actions</div>
      <ul class="actions">{items}</ul>
      <div class="muted" style="margin-top:.8rem;font-size:.78rem">
        Signals and actions are simple rules applied to the inputs. They are suggestions, not model explanations.
      </div>
    </div>
    """)


def page_predict() -> None:
    page_header("Predict Churn", "Enter a customer profile to get a churn prediction from your ML model.")

    for f in FEATURES:  # initialise the form fields once
        st.session_state.setdefault(f"f_{f['key']}", DEFAULTS[f["key"]])

    left, right = st.columns([1.25, 1], gap="large")

    with left:
        st.markdown('<div class="section-title">Quick start</div>', unsafe_allow_html=True)
        st.selectbox("Example customer profile", list(PRESETS), key="preset_choice",
                     help="Fills the form with sample INPUT values. These are not predictions.")
        b1, b2 = st.columns(2)
        b1.button("Load profile", on_click=on_load_preset)
        b2.button("Reset form", on_click=on_reset_form)

        with st.form("customer_form"):
            st.markdown('<div class="section-title">Account information</div>', unsafe_allow_html=True)
            cols = st.columns(3)
            for col, key in zip(cols, ["AccountWeeks", "ContractRenewal", "DataPlan"]):
                feature_input(col, key)

            st.markdown('<div class="section-title">Usage information</div>', unsafe_allow_html=True)
            cols = st.columns(2)
            for col, key in zip(cols * 2, ["DataUsage", "DayMins", "DayCalls", "RoamMins"]):
                feature_input(col, key)

            st.markdown('<div class="section-title">Billing &amp; customer support</div>', unsafe_allow_html=True)
            cols = st.columns(3)
            for col, key in zip(cols, ["MonthlyCharge", "OverageFee", "CustServCalls"]):
                feature_input(col, key)

            submitted = st.form_submit_button("🔮  Predict churn risk")

    with right:
        if submitted:
            payload = {k: st.session_state[f"f_{k}"] for k in FEATURE_KEYS}
            errors, warnings = validate_inputs(payload)
            for msg in errors:
                st.error(msg)
            for msg in warnings:
                st.warning(msg)

            if not errors:
                try:
                    with st.spinner("Analyzing customer profile..."):
                        prob, pred = predict_api(payload)
                    now = dt.datetime.now()
                    st.session_state["last_result"] = dict(
                        prob=prob, pred=pred, payload=payload, time=now.strftime("%d %b %Y, %H:%M:%S"))
                    st.session_state["history"].append(dict(
                        Timestamp=now.strftime("%Y-%m-%d %H:%M:%S"),
                        ChurnProbability=prob, Prediction=pred, **payload))
                    save_history(st.session_state["history"])
                except ApiError as err:
                    st.error(str(err))
                    check_api.clear()

        res = st.session_state.get("last_result")
        if res:
            result_card(res)
        else:
            render_html("""
            <div class="glass empty"><div class="big">🔮</div>
            <div class="card-title">Ready when you are</div>
            <div class="muted">Fill in the customer details and press <b>Predict churn risk</b>.
            The result will appear here.</div></div>
            """)


def page_analytics() -> None:
    page_header("Customer Analytics", "Insights built only from the predictions you have made.")
    df = history_df()
    if df.empty:
        empty_state("Analytics appear after you run predictions.", "goto_analytics")
        return

    kpi_grid([
        kpi_card("Predictions analysed", len(df), "From saved history", "activity", PURPLE),
        kpi_card("Average churn probability", f"{df['ChurnProbability'].mean():.1f}%",
                 f"Highest: {df['ChurnProbability'].max():.1f}%", "percent", BLUE),
        kpi_card("High-risk customers", int((df["Risk"] == "High").sum()),
                 f"Risk >= {settings()['high_min']}%", "alert", "#f43f5e"),
    ])

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        show_chart(fig_outcome_donut(df))
    with c2:
        show_chart(fig_risk_bar(df))

    c3, c4 = st.columns(2, gap="medium")
    with c3:
        show_chart(fig_trend(df))
    with c4:
        show_chart(fig_hist(df))

    st.markdown('<div class="section-title">Customer usage insights</div>', unsafe_allow_html=True)
    feature = st.selectbox("Feature to compare by predicted outcome", FEATURE_KEYS, index=5)
    c5, c6 = st.columns(2, gap="medium")
    with c5:
        show_chart(fig_scatter(df))
    with c6:
        show_chart(fig_box(df, feature))
    st.caption(f"Based on {len(df)} prediction(s) made in this app. Small samples can be misleading.")


def page_history() -> None:
    page_header("Prediction History", "Search, filter and export every prediction.")
    df = history_df()
    if df.empty:
        empty_state("Your prediction history will be listed here.", "goto_history")
        return

    disp = display_frame(df.sort_values("Timestamp", ascending=False))

    f1, f2, f3 = st.columns([2, 1, 1])
    query = f1.text_input("Search", placeholder="Search any value, date or outcome...")
    outcomes = f2.multiselect("Outcome", ["Churn", "Stay"], default=["Churn", "Stay"])
    risks = f3.multiselect("Risk level", ["Low", "Medium", "High"], default=["Low", "Medium", "High"])
    lo, hi = st.slider("Churn probability range (%)", 0, 100, (0, 100))

    view = disp[disp["Outcome"].isin(outcomes) & disp["Risk"].isin(risks) & disp["Churn %"].between(lo, hi)]
    if query:
        mask = view.astype(str).apply(lambda c: c.str.contains(query, case=False, regex=False)).any(axis=1)
        view = view[mask]

    st.caption(f"Showing {len(view)} of {len(disp)} predictions")
    show_table(view)

    d1, d2 = st.columns(2)
    d1.download_button("⬇️  Download CSV", view.to_csv(index=False).encode("utf-8"),
                       "churnguard_predictions.csv", "text/csv", disabled=view.empty)
    with st.expander("Danger zone"):
        confirm = st.checkbox("I understand this permanently deletes all saved predictions")
        if st.button("🗑️  Clear history", disabled=not confirm):
            st.session_state["history"] = []
            st.session_state["last_result"] = None
            save_history([])
            st.rerun()


def page_model() -> None:
    page_header("Model Information", "Details about the prediction service this dashboard connects to.")
    cfg = settings()
    online = check_api(cfg["api_url"])
    status = "🟢 Connected" if online else "🟠 Not reachable"

    render_html(f"""
    <div class="info-grid">
      <div class="glass"><div class="k">Model name</div><div class="v">{esc(cfg["model_name"])}</div></div>
      <div class="glass"><div class="k">Prediction endpoint</div><div class="v">{esc(cfg["api_url"])}</div></div>
      <div class="glass"><div class="k">Connection status</div><div class="v">{status}</div></div>
    </div>
    """)

    st.markdown('<div class="section-title">Input features</div>', unsafe_allow_html=True)
    feats = pd.DataFrame([
        dict(Feature=f["key"], Label=f["label"],
             Type={"int": "Whole number", "float": "Decimal", "bool": "1 = Yes, 0 = No"}[f["kind"]],
             Description=f["help"]) for f in FEATURES
    ])
    show_table(feats)

    render_html("""
    <div class="info-grid" style="margin-top:1.2rem">
      <div class="glass"><div class="card-title">Prediction output</div>
        <ul class="actions">
          <li><b>churn_probability</b>: the model's churn probability as a percentage (0-100).</li>
          <li><b>prediction</b>: 1 means the customer is predicted to churn, 0 means likely to stay.</li>
          <li>The Low / Medium / High risk label is added by this dashboard from the probability bands in Settings.</li>
        </ul></div>
      <div class="glass"><div class="card-title">Limitations</div>
        <ul class="actions">
          <li>Predictions are probabilities, not certainties.</li>
          <li>Quality depends on the data the model was trained on.</li>
          <li>Inputs must use the same units and scale as the training data.</li>
          <li>The model does not explain individual predictions.</li>
          <li>Risk signals shown here are simple rules, not model output.</li>
          <li>Use alongside human judgement for customer decisions.</li>
        </ul></div>
    </div>
    <div class="section-title">Model performance (placeholders)</div>
    <div class="kpi-grid">
      <div class="glass kpi" style="--a:#8b5cf6"><div class="label">Accuracy</div><div class="value">-</div><div class="sub">Not provided</div></div>
      <div class="glass kpi" style="--a:#3b82f6"><div class="label">Precision</div><div class="value">-</div><div class="sub">Not provided</div></div>
      <div class="glass kpi" style="--a:#22d3ee"><div class="label">Recall</div><div class="value">-</div><div class="sub">Not provided</div></div>
      <div class="glass kpi" style="--a:#22c55e"><div class="label">ROC-AUC</div><div class="value">-</div><div class="sub">Not provided</div></div>
    </div>
    """)
    st.caption("Your API does not return model metrics. Add real values from your training notebook if you want to show them.")


def page_settings() -> None:
    page_header("Settings", "Configure the API connection and how risk levels are displayed.")
    cfg = settings()

    with st.form("settings_form"):
        st.markdown('<div class="section-title">API connection</div>', unsafe_allow_html=True)
        api_url = st.text_input("Prediction API URL", cfg["api_url"],
                                help="The full URL of your FastAPI /predict endpoint.")
        timeout = st.number_input("Request timeout (seconds)", 1, 120, int(cfg["timeout"]))
        model_name = st.text_input("Model display name", cfg["model_name"],
                                   help="Only used as a label on the Model Information page.")

        st.markdown('<div class="section-title">Risk bands (display only)</div>', unsafe_allow_html=True)
        s1, s2 = st.columns(2)
        low_max = s1.slider("Low risk below (%)", 1, 99, int(cfg["low_max"]))
        high_min = s2.slider("High risk from (%)", 1, 99, int(cfg["high_min"]))
        saved = st.form_submit_button("Save settings")

    if saved:
        if not api_url.strip().startswith(("http://", "https://")):
            st.error("The API URL must start with http:// or https://")
        elif low_max >= high_min:
            st.error("'Low risk below' must be smaller than 'High risk from'.")
        else:
            st.session_state["settings"] = dict(
                api_url=api_url.strip(), timeout=int(timeout), model_name=model_name.strip() or "Not specified",
                low_max=int(low_max), high_min=int(high_min))
            check_api.clear()
            st.rerun()

    c1, c2 = st.columns(2)
    if c1.button("🔄  Re-check API connection"):
        check_api.clear()
        st.rerun()
    if c2.button("↩️  Restore default settings"):
        st.session_state["settings"] = dict(DEFAULT_SETTINGS)
        check_api.clear()
        st.rerun()

    st.caption(f"Prediction history is saved to: {HISTORY_FILE}")


# =============================================================================
# 9. APP ENTRY POINT
# =============================================================================
PAGE_FUNCTIONS = {
    "Dashboard Overview": page_overview,
    "Predict Churn": page_predict,
    "Customer Analytics": page_analytics,
    "Prediction History": page_history,
    "Model Information": page_model,
    "Settings": page_settings,
}


def main() -> None:
    init_state()
    inject_css()
    page = render_sidebar()
    PAGE_FUNCTIONS[page]()


main()