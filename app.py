# ================================================================
#  FloodSense AI  ·  app.py
#  ★  YOUR API KEY → Line 4  ★
DEFAULT_API_KEY = "3258f51cbbcca4a16589beac35c9ee6c"
DEFAULT_CITY    = "London"
# ================================================================

import os, time
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from utils       import load_model_and_scaler, predict_flood, add_to_history, history_to_dataframe
from weather_api import fetch_weather

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title            = "FloodSense AI",
    page_icon             = "🌊",
    layout                = "wide",
    initial_sidebar_state = "expanded",
)

# ── Session state ─────────────────────────────────────────────────
def _s(k, v):
    if k not in st.session_state:
        st.session_state[k] = v

_s("dark",    True)
_s("api_key", DEFAULT_API_KEY)
_s("city",    DEFAULT_CITY)
_s("weather", None)
_s("result",  None)
_s("history", [])
_s("v_rain",  60.0)
_s("v_temp",  28.0)
_s("v_hum",   65.0)
_s("v_river", 4.5)
_s("v_soil",  50.0)

# ── Theme ─────────────────────────────────────────────────────────
D = st.session_state.dark
if D:
    BG=  "#070d1b"; CARD= "rgba(255,255,255,0.05)"; CARD2="rgba(255,255,255,0.025)"
    CHVR="rgba(255,255,255,0.08)"; BDR="rgba(255,255,255,0.08)"; BDRA="rgba(56,189,248,0.4)"
    TP=  "#eaf4fc"; TS=  "#5ba3c2"; TM= "#2e4f63"
    SBG= "#050b16"; AC=  "#0ea5e9"; AC2="#38bdf8"; ACGL="rgba(14,165,233,0.15)"
    OK=  "#10b981"; WARN="#f59e0b"; DNG="#ef4444"
    SH=  "0 8px 40px rgba(0,0,0,0.5)"; SHS="0 2px 14px rgba(0,0,0,0.3)"
    IB=  "rgba(255,255,255,0.04)";  TR= "rgba(255,255,255,0.09)";  PT="plotly_dark"
    INPUT_TEXT = "#eaf4fc"
    INPUT_BG = "rgba(255,255,255,0.07)"
    INPUT_PH = "#5ba3c2"
else:
    BG=  "#f4f8ff"; CARD= "rgba(255,255,255,0.94)"; CARD2="rgba(255,255,255,0.6)"
    CHVR="#ffffff";  BDR= "rgba(14,165,233,0.12)";  BDRA="rgba(14,165,233,0.5)"
    TP=  "#091e35"; TS=  "#1460a0"; TM= "#5580a0"
    SBG= "#e8f2fc"; AC=  "#0277bd"; AC2="#0ea5e9"; ACGL="rgba(14,165,233,0.10)"
    OK=  "#059669"; WARN="#d97706"; DNG="#dc2626"
    SH=  "0 4px 28px rgba(14,165,233,0.10)"; SHS="0 2px 10px rgba(14,165,233,0.07)"
    IB=  "rgba(14,165,233,0.04)";  TR= "rgba(14,165,233,0.09)";   PT="plotly_white"
    INPUT_TEXT = "#091e35"
    INPUT_BG = "#ffffff"
    INPUT_PH = "#5580a0"


# ─────────────────────────────────────────────────────────────────
#  CSS  (JS auto-open block REMOVED — it fought Streamlit's sidebar)
# ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
/* ── Fonts & Reset ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
*, *::before, *::after {{ box-sizing: border-box; }}
html, body, [class*="css"] {{
    font-family: 'Inter', system-ui, sans-serif !important;
    -webkit-font-smoothing: antialiased;
}}

/* ── App shell ── */
.stApp {{ background: {BG} !important; color: {TP} !important; }}
footer {{ display: none !important; }}
[data-testid="stDecoration"] {{ display: none !important; }}

/* ── Style the header bar to match dark theme (do NOT hide it) ── */
[data-testid="stHeader"] {{
    background: {BG} !important;
    border-bottom: 1px solid {BDR} !important;
}}
/* Style the hamburger menu icon inside header */
[data-testid="stHeader"] button {{
    color: {TP} !important;
}}
[data-testid="stHeader"] svg {{
    fill: {TP} !important;
    color: {TP} !important;
}}

.block-container {{
    padding: 4rem 2.4rem 3rem !important;
    max-width: 1380px !important;
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: {SBG} !important;
    border-right: 1px solid {BDR} !important;
}}
[data-testid="stSidebar"] > div:first-child {{
    padding: 1.5rem 1.2rem 1.2rem !important;
}}

/* ── Collapse button inside sidebar ── */
[data-testid="stSidebarCollapseButton"] button {{
    background: {IB} !important;
    border: 1px solid {BDR} !important;
    border-radius: 8px !important;
    color: {TS} !important;
    transition: background 0.15s, border-color 0.15s !important;
}}
[data-testid="stSidebarCollapseButton"] button:hover {{
    background: {ACGL} !important;
    border-color: {AC} !important;
}}
[data-testid="stSidebarCollapseButton"] svg {{ color: {TS} !important; fill: {TS} !important; }}

/* ── Collapsed arrow — guaranteed visible ── */
[data-testid="stSidebarCollapsedControl"] {{
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    z-index: 99999 !important;
}}
[data-testid="stSidebarCollapsedControl"] button {{
    width: 28px !important;
    height: 64px !important;
    background: #e53935 !important;
    border-radius: 0 10px 10px 0 !important;
    border: none !important;
    cursor: pointer !important;
    box-shadow: 2px 2px 14px rgba(229,57,53,0.6) !important;
}}
[data-testid="stSidebarCollapsedControl"] svg {{
    color: #ffffff !important;
    fill: #ffffff !important;
}}

/* ── Sidebar text colours ── */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div {{ color: {TP} !important; }}

/* ── Sidebar city input — theme-aware ── */
[data-testid="stSidebar"] .stTextInput > div {{
    background: {INPUT_BG} !important;
    border: 1px solid {BDR} !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}}
[data-testid="stSidebar"] .stTextInput > div:focus-within {{
    border-color: {AC} !important;
    box-shadow: 0 0 0 3px {ACGL} !important;
}}
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] [data-baseweb="input"] input,
[data-testid="stSidebar"] [data-baseweb="base-input"] input,
[data-testid="stSidebar"] [class*="Input"] input {{
    background: transparent !important;
    color: {INPUT_TEXT} !important;
    -webkit-text-fill-color: {INPUT_TEXT} !important;
    caret-color: {INPUT_TEXT} !important;
    opacity: 1 !important;
    font-size: 0.875rem !important;
    font-family: 'Inter', sans-serif !important;
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    padding: 0.55rem 0.9rem !important;
    width: 100% !important;
}}
[data-testid="stSidebar"] input:focus,
[data-testid="stSidebar"] .stTextInput input:focus {{
    color: {INPUT_TEXT} !important;
    -webkit-text-fill-color: {INPUT_TEXT} !important;
}}
[data-testid="stSidebar"] input::placeholder,
[data-testid="stSidebar"] .stTextInput input::placeholder {{
    color: {INPUT_PH} !important;
    -webkit-text-fill-color: {INPUT_PH} !important;
    opacity: 1 !important;
}}
[data-testid="stSidebar"] input:-webkit-autofill,
[data-testid="stSidebar"] input:-webkit-autofill:hover,
[data-testid="stSidebar"] input:-webkit-autofill:focus {{
    -webkit-text-fill-color: {INPUT_TEXT} !important;
    -webkit-box-shadow: 0 0 0px 1000px {INPUT_BG} inset !important;
    transition: background-color 5000s ease-in-out 0s !important;
}}

/* ── Sliders — green→yellow→red risk gradient ── */
[data-testid="stSlider"] > label,
[data-testid="stSlider"] > label p {{
    color: {TP} !important; font-size:0.875rem !important;
    font-weight:500 !important; opacity:1 !important;
    transition: color 0.2s !important;
}}
[data-testid="stSlider"] > div > div > div {{
    background: {TR} !important;
    height: 6px !important;
    border-radius: 6px !important;
}}
[data-testid="stSlider"] > div > div > div > div:first-child {{
    background: linear-gradient(90deg,
        #16a34a 0%,
        #22c55e 25%,
        #facc15 50%,
        #f97316 75%,
        #dc2626 100%
    ) !important;
    height: 6px !important;
    border-radius: 6px !important;
}}
[data-testid="stSlider"] [role="slider"] {{
    background: #ffffff !important;
    border: 2.5px solid #6b7280 !important;
    box-shadow: 0 0 0 3px {ACGL}, 0 2px 8px rgba(0,0,0,0.2) !important;
    width: 18px !important; height: 18px !important;
    border-radius: 50% !important;
    cursor: grab !important;
    transition: box-shadow 0.15s !important;
}}
[data-testid="stSlider"] [role="slider"]:active {{ cursor: grabbing !important; }}
[data-testid="stSlider"] [role="slider"]:hover {{
    box-shadow: 0 0 0 5px {ACGL}, 0 3px 10px rgba(0,0,0,0.25) !important;
}}

/* ── Buttons ── */
.stButton > button {{
    background:linear-gradient(135deg,{AC} 0%,{AC2} 100%) !important;
    color:#fff !important; border:none !important; border-radius:11px !important;
    font-family:'Inter',sans-serif !important; font-weight:600 !important;
    font-size:0.9rem !important; padding:0.65rem 1.5rem !important;
    width:100% !important; box-shadow:0 4px 16px rgba(14,165,233,0.32) !important;
    transition:all 0.2s ease !important; cursor:pointer !important;
}}
.stButton > button:hover {{
    transform:translateY(-2px) !important;
    box-shadow:0 8px 24px rgba(14,165,233,0.48) !important;
}}
.stButton > button:active {{ transform:translateY(0) !important; }}

/* ── Metric cards ── */
[data-testid="metric-container"] {{
    background:{CARD} !important; border:1px solid {BDR} !important;
    border-radius:14px !important; padding:1rem 1.15rem !important;
    box-shadow:{SHS} !important; transition:border-color 0.2s, box-shadow 0.2s !important;
}}
[data-testid="metric-container"]:hover {{
    border-color:{BDRA} !important; box-shadow:0 0 20px {ACGL} !important;
}}
[data-testid="metric-container"] label {{
    color:{TS} !important; font-size:0.7rem !important;
    font-weight:600 !important; text-transform:uppercase !important; letter-spacing:0.7px !important;
}}
[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    color:{TP} !important; font-weight:700 !important; font-size:1.3rem !important;
}}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    background:transparent !important; border-bottom:1px solid {BDR} !important;
    padding:0 !important; gap:0 !important;
}}
.stTabs [data-baseweb="tab"] {{
    background:transparent !important; border:none !important;
    border-bottom:2px solid transparent !important; color:{TM} !important;
    font-family:'Inter',sans-serif !important; font-weight:500 !important;
    font-size:0.875rem !important; padding:0.8rem 1.4rem !important;
    border-radius:0 !important; margin-bottom:-1px !important;
    transition:color 0.15s, border-color 0.15s !important;
}}
.stTabs [data-baseweb="tab"]:hover {{ color:{AC2} !important; }}
.stTabs [aria-selected="true"] {{
    color:{AC} !important; border-bottom:2px solid {AC} !important;
    font-weight:600 !important; background:transparent !important;
}}
.stTabs [data-baseweb="tab-panel"] {{
    background:transparent !important; border:none !important;
    padding:1.8rem 0 0 !important;
}}

/* ── Toggle & Divider ── */
[data-testid="stToggle"] label {{
    color:{TP} !important; font-size:0.875rem !important; font-weight:500 !important;
}}
hr {{ border-color:{BDR} !important; margin:0.85rem 0 !important; }}

/* ── COMPONENTS ── */
.sb-brand {{
    display:flex; align-items:center; gap:10px;
    padding-bottom:16px; border-bottom:1px solid {BDR}; margin-bottom:16px;
}}
.sb-icon {{ font-size:1.7rem; line-height:1; }}
.sb-name {{
    font-size:1.1rem; font-weight:800; letter-spacing:-0.3px;
    background:linear-gradient(135deg,{AC},{AC2});
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
}}
.sb-tag {{ font-size:0.6rem; color:{TM}; font-weight:600; letter-spacing:0.9px; text-transform:uppercase; margin-top:1px; }}
.sb-wx {{
    background:{IB}; border:1px solid {BDR}; border-radius:12px;
    padding:10px 12px; margin-top:8px;
    display:flex; align-items:center; justify-content:space-between;
}}
.sb-wx-city {{ font-size:0.78rem; font-weight:600; color:{TP}; }}
.sb-wx-temp {{ font-size:1.2rem; font-weight:800; color:{AC2}; }}
.sb-wx-dtl  {{ font-size:0.68rem; color:{TM}; margin-top:1px; }}
.sb-status  {{ display:flex; flex-direction:column; gap:6px; margin-top:10px; }}
.sb-row     {{ display:flex; align-items:center; gap:7px; font-size:0.75rem; color:{TS}; }}
.dot-on     {{ width:7px; height:7px; border-radius:50%; background:{OK}; flex-shrink:0; box-shadow:0 0 6px {OK}88; }}
.dot-off    {{ width:7px; height:7px; border-radius:50%; background:{TM}; flex-shrink:0; }}

.ph {{
    display:flex; align-items:flex-end; justify-content:space-between;
    padding-bottom:1.2rem; border-bottom:1px solid {BDR}; margin-bottom:1.4rem;
    flex-wrap:wrap; gap:0.5rem;
}}
.ph-title {{
    font-size:1.65rem; font-weight:900; letter-spacing:-0.8px; line-height:1.15;
    background:linear-gradient(130deg,{AC} 0%,{AC2} 55%,#a5f3fc 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
}}
.ph-sub {{ font-size:0.82rem; color:{TM}; margin-top:3px; }}
.gc {{
    background:{CARD}; border:1px solid {BDR}; border-radius:20px;
    padding:1.4rem 1.6rem; box-shadow:{SHS}; margin-bottom:0.9rem;
    transition:border-color 0.22s, box-shadow 0.22s; backdrop-filter:blur(6px);
}}
.gc:hover {{ border-color:{BDRA}; box-shadow:{SH}; }}
.ttl {{ font-size:0.95rem; font-weight:700; color:{TP}; margin-bottom:0.9rem; display:flex; align-items:center; gap:6px; }}
.lbl {{ font-size:0.66rem; font-weight:700; letter-spacing:1.1px; text-transform:uppercase; color:{TM}; margin-bottom:0.3rem; }}

.wx {{
    background:{CARD}; border:1px solid {BDR}; border-radius:20px;
    padding:1.3rem 1.6rem; box-shadow:{SHS};
    display:flex; align-items:flex-start; justify-content:space-between;
    flex-wrap:wrap; gap:1rem; margin-bottom:1.2rem; transition:border-color 0.2s;
}}
.wx:hover {{ border-color:{BDRA}; }}
.wx-city {{ font-size:0.9rem; font-weight:700; color:{TP}; }}
.wx-temp {{ font-size:2.8rem; font-weight:900; letter-spacing:-2.5px; color:{TP}; line-height:1; margin:5px 0; }}
.wx-desc {{ font-size:0.78rem; color:{TS}; font-style:italic; margin-top:2px; }}
.live-tag {{
    background:rgba(16,185,129,0.11); border:1px solid {OK}; border-radius:5px;
    padding:1px 7px; font-size:0.58rem; font-weight:800; color:{OK};
    letter-spacing:0.9px; vertical-align:middle; margin-left:7px;
}}
.wx-grid {{ display:grid; grid-template-columns:repeat(3,1fr); gap:8px; min-width:196px; align-self:center; }}
.wx-stat {{ background:{IB}; border:1px solid {BDR}; border-radius:10px; padding:8px 6px; text-align:center; transition:border-color 0.15s; }}
.wx-stat:hover {{ border-color:{BDRA}; }}
.wxs-v {{ font-size:0.9rem; font-weight:700; color:{TP}; }}
.wxs-l {{ font-size:0.62rem; color:{TM}; text-transform:uppercase; letter-spacing:0.4px; margin-top:2px; }}
.wx-none {{
    text-align:center; padding:1.1rem 1.4rem; color:{TM};
    background:{CARD2}; border:1px dashed {BDR}; border-radius:16px;
    font-size:0.83rem; margin-bottom:1.2rem;
}}
.prob-wrap {{ text-align:center; padding:1.8rem 1.5rem 1.3rem; }}
.prob-num {{
    font-size:5.2rem; font-weight:900; letter-spacing:-4px; line-height:1;
    animation:popIn 0.4s cubic-bezier(0.175,0.885,0.32,1.275);
}}
@keyframes popIn {{ 0%{{opacity:0;transform:scale(0.7)}} 65%{{transform:scale(1.06)}} 100%{{opacity:1;transform:scale(1)}} }}
.risk-badge {{
    display:inline-flex; align-items:center; gap:5px; border-radius:50px;
    padding:5px 18px; font-size:0.8rem; font-weight:700; letter-spacing:0.4px; margin-top:10px;
}}
.prog-track {{ background:{TR}; border-radius:50px; height:8px; overflow:hidden; margin:1.1rem 0 0.3rem; }}
.prog-bar   {{ height:100%; border-radius:50px; transition:width 1s cubic-bezier(0.4,0,0.2,1); }}
.prog-lbl   {{ display:flex; justify-content:space-between; font-size:0.64rem; color:{TM}; }}
.expl       {{ background:{IB}; border-left:3px solid {AC}; border-radius:0 12px 12px 0; padding:0.75rem 1rem; font-size:0.86rem; color:{TS}; line-height:1.65; margin-top:0.6rem; }}
.await-wrap {{ text-align:center; padding:3.5rem 1.5rem; color:{TM}; }}
.await-icon {{ font-size:2.8rem; margin-bottom:0.7rem; }}
.await-ttl  {{ font-size:0.95rem; font-weight:600; color:{TS}; }}
.await-sub  {{ font-size:0.8rem; margin-top:0.3rem; line-height:1.6; }}
.h-row {{
    display:flex; align-items:center; gap:12px; padding:0.9rem 1.1rem;
    border-radius:14px; background:{CARD}; border:1px solid {BDR}; margin-bottom:7px;
    transition:border-color 0.15s, background 0.15s, transform 0.15s; cursor:default;
}}
.h-row:hover {{ border-color:{BDRA}; background:{CHVR}; transform:translateX(3px); }}
.h-em {{ font-size:1.2rem; flex-shrink:0; }}
.h-pct{{ font-size:1.05rem; font-weight:800; min-width:60px; }}
.h-bdg{{ min-width:88px; }}
.h-meta{{ flex:1; font-size:0.75rem; color:{TM}; }}
.h-ts {{ font-size:0.69rem; color:{TM}; white-space:nowrap; }}
.empty-wrap {{ text-align:center; padding:4rem 2rem; color:{TM}; }}
.empty-icon {{ font-size:2.6rem; margin-bottom:0.7rem; }}
.empty-ttl  {{ font-size:0.95rem; font-weight:600; color:{TS}; }}
.empty-sub  {{ font-size:0.8rem; margin-top:0.3rem; }}

@media screen and (max-width:768px) {{
    .block-container {{ padding:1rem 0.9rem 2rem !important; }}
    .prob-num {{ font-size:3.6rem !important; letter-spacing:-2px !important; }}
    .wx-temp  {{ font-size:2.2rem !important; }}
    .ph-title {{ font-size:1.3rem !important; }}
    .wx-grid  {{ min-width:unset; }}
    .stButton > button {{ font-size:1rem !important; padding:0.75rem !important; }}
    .h-meta {{ display:none; }}
}}
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model …")
def get_model():
    return load_model_and_scaler()

model, scaler   = get_model()
_api_ok   = st.session_state.api_key not in ["", "YOUR_API_KEY_HERE"]
_model_ok = model is not None

# ════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div class="sb-brand">
      <div class="sb-icon">🌊</div>
      <div><div class="sb-name">FloodSense AI</div>
           <div class="sb-tag">Flood Intelligence</div></div>
    </div>
    """, unsafe_allow_html=True)

    _dark = st.toggle("🌙 Dark Mode", value=st.session_state.dark, key="tog")
    if _dark != st.session_state.dark:
        st.session_state.dark = _dark
        st.rerun()

    st.divider()

    _city = st.text_input("Location", value=st.session_state.city,
                          placeholder="📍 Enter city …",
                          label_visibility="collapsed", key="in_city")
    st.session_state.city = _city

    if st.button("☁️  Fetch Live Weather", use_container_width=True, key="btn_fetch"):
        k = st.session_state.api_key
        if not k.strip() or k == "YOUR_API_KEY_HERE":
            st.error("Add your API key on line 4 of app.py")
        else:
            with st.spinner("Fetching …"):
                _wd = fetch_weather(st.session_state.city, k)
            if _wd.success:
                st.session_state.weather = _wd
                st.session_state.v_rain  = max(0.0,  min(300.0, float(_wd.rainfall_1h)))
                st.session_state.v_temp  = max(10.0, min(45.0,  float(_wd.temperature)))
                st.session_state.v_hum   = max(20.0, min(100.0, float(_wd.humidity)))
                st.success(f"✅ {_wd.city}, {_wd.country}")
                time.sleep(0.6); st.rerun()
            else:
                st.error(_wd.error_message)

    _wd = st.session_state.weather
    if _wd and _wd.success:
        st.markdown(f"""
        <div class="sb-wx">
          <div>
            <div class="sb-wx-city">{_wd.city}</div>
            <div class="sb-wx-dtl">💧{_wd.rainfall_1h}mm · 💨{_wd.humidity}%</div>
          </div>
          <div class="sb-wx-temp">{_wd.temperature}°</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown(f"""
    <div class="sb-status">
      <div class="sb-row"><div class="{'dot-on' if _model_ok else 'dot-off'}"></div>
        <span>Model {'active' if _model_ok else 'not loaded'}</span></div>
      <div class="sb-row"><div class="{'dot-on' if _api_ok else 'dot-off'}"></div>
        <span>API {'configured' if _api_ok else 'not set'}</span></div>
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
#  PAGE HEADER + WEATHER CARD
# ════════════════════════════════════════════════════════
st.markdown(f"""
<div class="ph">
  <div>
    <div class="ph-title">FloodSense AI</div>
    <div class="ph-sub">AI-Powered Real-Time Flood Intelligence Platform</div>
  </div>
</div>
""", unsafe_allow_html=True)

wd = st.session_state.weather
if wd and wd.success:
    st.markdown(f"""
    <div class="wx">
      <div>
        <div><span class="wx-city">{wd.city}, {wd.country}</span><span class="live-tag">LIVE</span></div>
        <div class="wx-temp">{wd.temperature}°C</div>
        <div class="wx-desc">{wd.description}</div>
      </div>
      <div class="wx-grid">
        <div class="wx-stat"><div class="wxs-v">💧 {wd.rainfall_1h}</div><div class="wxs-l">Rain mm/h</div></div>
        <div class="wx-stat"><div class="wxs-v">💨 {wd.humidity}%</div><div class="wxs-l">Humidity</div></div>
        <div class="wx-stat"><div class="wxs-v">🌬 {wd.wind_speed}</div><div class="wxs-l">Wind m/s</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="wx-none">🌤️ &nbsp; Open the sidebar → enter your city → click
      <b style="color:{AC};">Fetch Live Weather</b>
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
#  TABS
# ════════════════════════════════════════════════════════
tab_pred, tab_data, tab_hist = st.tabs([
    "🎯  Prediction", "📊  Data Insights", "📋  History"
])

# ════════════════════════════════════════════════════════
#  TAB 1 — PREDICTION
# ════════════════════════════════════════════════════════
with tab_pred:
    lc, rc = st.columns([1, 1], gap="large")

    with lc:
        st.markdown('<div class="ttl">🌡️ Environmental Parameters</div>',
                    unsafe_allow_html=True)

        _rain = st.slider("💧 Rainfall (mm/day)", 0.0, 300.0,
                          float(st.session_state.v_rain), 0.5, key="sl_rain")

        _temp = st.slider("🌡️ Temperature (°C)", 10.0, 45.0,
                          float(st.session_state.v_temp), 0.5, key="sl_temp")

        _hum  = st.slider("💨 Humidity (%)", 20.0, 100.0,
                          float(st.session_state.v_hum), 0.5, key="sl_hum")

        _river = st.slider("🌊 River Level (m)", 0.0, 15.0,
                           float(st.session_state.v_river), 0.1, key="sl_river")

        _soil = st.slider("🌱 Soil Moisture (%)", 0.0, 100.0,
                          float(st.session_state.v_soil), 0.5, key="sl_soil")

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        if st.button("🔍  Estimate Flood Risk", use_container_width=True, key="btn_pred"):
            st.session_state.v_rain  = _rain
            st.session_state.v_temp  = _temp
            st.session_state.v_hum   = _hum
            st.session_state.v_river = _river
            st.session_state.v_soil  = _soil
            with st.spinner("Analysing …"):
                time.sleep(0.2)
                _r = predict_flood(model, scaler, _rain, _temp, _hum, _river, _soil)
            st.session_state.result = _r
            add_to_history(st.session_state, _r)

    with rc:
        st.markdown('<div class="ttl">📊 Risk Assessment</div>',
                    unsafe_allow_html=True)
        res = st.session_state.result

        if res is None:
            st.markdown(f"""
            <div class="gc await-wrap">
              <div class="await-icon">🤖</div>
              <div class="await-ttl">No Prediction Yet</div>
              <div class="await-sub">Adjust sliders on the left,<br/>
                then click <b style="color:{AC};">Estimate Flood Risk</b>.</div>
            </div>""", unsafe_allow_html=True)
        else:
            p=res["probability"]; lv=res["risk_level"]; em=res["emoji"]
            if p < 35:
                tc=OK;   bgb="rgba(16,185,129,0.10)"; bdb=OK;   bar="linear-gradient(90deg,#059669,#34d399)"
            elif p < 65:
                tc=WARN; bgb="rgba(245,158,11,0.10)"; bdb=WARN; bar="linear-gradient(90deg,#b45309,#fbbf24)"
            else:
                tc=DNG;  bgb="rgba(239,68,68,0.10)";  bdb=DNG;  bar="linear-gradient(90deg,#b91c1c,#f87171)"

            st.markdown(f"""
            <div class="gc prob-wrap">
              <div class="lbl">Flood Probability</div>
              <div class="prob-num" style="color:{tc};">{p}%</div>
              <div><span class="risk-badge"
                style="background:{bgb};border:1.5px solid {bdb};color:{tc};">
                {em}&nbsp;{lv.upper()}</span></div>
              <div style="margin-top:1.2rem;">
                <div class="prog-track">
                  <div class="prog-bar" style="width:{p}%;background:{bar};box-shadow:0 0 12px {bdb}55;"></div>
                </div>
                <div class="prog-lbl"><span>Safe</span><span>50%</span><span>Critical</span></div>
              </div>
            </div>""", unsafe_allow_html=True)

            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=p,
                number={"suffix":"%","font":{"size":26,"color":TP,"family":"Inter"}},
                gauge={
                    "axis":{"range":[0,100],"tickcolor":TM,"tickwidth":1,"tickfont":{"color":TM,"size":9}},
                    "bar":{"color":tc,"thickness":0.22},
                    "bgcolor":"rgba(0,0,0,0)","borderwidth":0,
                    "steps":[
                        {"range":[0,35],"color":"rgba(16,185,129,0.07)"},
                        {"range":[35,65],"color":"rgba(245,158,11,0.07)"},
                        {"range":[65,100],"color":"rgba(239,68,68,0.07)"},
                    ],
                    "threshold":{"line":{"color":tc,"width":2},"value":p},
                },
            ))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                              height=190,margin=dict(l=24,r=24,t=14,b=6),
                              font={"color":TP,"family":"Inter"})
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

            st.markdown(f'<div class="expl">{res["explanation"]}</div>', unsafe_allow_html=True)
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
            m1,m2,m3 = st.columns(3)
            m1.metric("💧 Rain",     f"{res['inputs']['rainfall']:.0f} mm")
            m2.metric("🌊 River",    f"{res['inputs']['river_level']:.1f} m")
            m3.metric("💨 Humidity", f"{res['inputs']['humidity']:.0f}%")

# ════════════════════════════════════════════════════════
#  TAB 2 — DATA INSIGHTS
# ════════════════════════════════════════════════════════
with tab_data:
    CSV = os.path.join("data", "flood_dataset.csv")
    if not os.path.exists(CSV):
        st.warning("Dataset not found — run `python model_training.py` first.")
    else:
        df = pd.read_csv(CSV)

        k1,k2,k3,k4 = st.columns(4)
        k1.metric("Total Samples", f"{len(df):,}")
        k2.metric("Flood Events",  f"{df['Flood_Risk'].sum():,}")
        k3.metric("Non-Flood",     f"{(df['Flood_Risk']==0).sum():,}")
        k4.metric("Flood Rate",    f"{df['Flood_Risk'].mean()*100:.1f}%")

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2, gap="medium")
        with c1:
            st.markdown('<div class="ttl">Rainfall Distribution</div>', unsafe_allow_html=True)
            fig_b = px.box(df, x="Flood_Risk", y="Rainfall_mm", color="Flood_Risk",
                           color_discrete_map={0:"#10b981",1:"#ef4444"},
                           labels={"Flood_Risk":"Flood","Rainfall_mm":"Rainfall (mm/day)"},
                           template=PT)
            fig_b.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                                showlegend=False,height=300,
                                font={"color":TP,"family":"Inter"},
                                margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig_b, use_container_width=True, config={"displayModeBar":False})

        with c2:
            st.markdown('<div class="ttl">River Level vs Soil Moisture</div>', unsafe_allow_html=True)
            smp = df.sample(min(500,len(df)), random_state=7)
            fig_s = px.scatter(smp, x="River_Level_m", y="Soil_Moisture_pct",
                               color=smp["Flood_Risk"].map({0:"No Flood",1:"Flood"}),
                               color_discrete_map={"No Flood":"#10b981","Flood":"#ef4444"},
                               opacity=0.6, template=PT,
                               labels={"River_Level_m":"River Level (m)","Soil_Moisture_pct":"Soil Moisture (%)"})
            fig_s.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                                height=300,legend_title_text="",
                                font={"color":TP,"family":"Inter"},
                                margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig_s, use_container_width=True, config={"displayModeBar":False})

        st.markdown('<div class="ttl">Feature Importance</div>', unsafe_allow_html=True)
        fi_df = pd.DataFrame({
            "Feature":["Rainfall","Temperature","Humidity","River Level","Soil Moisture"],
            "Importance": model.feature_importances_
        }).sort_values("Importance")
        fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                        color="Importance",
                        color_continuous_scale=[[0,AC],[0.5,AC2],[1,"#a5f3fc"]],
                        template=PT)
        fig_fi.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                             height=260,showlegend=False,coloraxis_showscale=False,
                             font={"color":TP,"family":"Inter"},
                             margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig_fi, use_container_width=True, config={"displayModeBar":False})

        st.markdown('<div class="ttl">Feature Correlation</div>', unsafe_allow_html=True)
        corr = df[["Rainfall_mm","Temperature_C","Humidity_pct",
                   "River_Level_m","Soil_Moisture_pct","Flood_Risk"]].corr().round(2)
        fig_hm = px.imshow(corr, text_auto=True, aspect="auto",
                           color_continuous_scale="RdBu_r", template=PT)
        fig_hm.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                             height=340,font={"color":TP,"family":"Inter"},
                             margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig_hm, use_container_width=True, config={"displayModeBar":False})

# ════════════════════════════════════════════════════════
#  TAB 3 — HISTORY
# ════════════════════════════════════════════════════════
with tab_hist:
    hist = st.session_state.history

    if not hist:
        st.markdown(f"""
        <div class="empty-wrap">
          <div class="empty-icon">📋</div>
          <div class="empty-ttl">No predictions yet</div>
          <div class="empty-sub">Run an assessment and results will appear here.</div>
        </div>""", unsafe_allow_html=True)
    else:
        probs = [h["probability"] for h in hist]
        sc1,sc2,sc3,sc4 = st.columns(4)
        sc1.metric("Total Runs", len(hist))
        sc2.metric("Avg Risk",   f"{sum(probs)/len(probs):.1f}%")
        sc3.metric("Peak Risk",  f"{max(probs):.1f}%")
        sc4.metric("Safe Runs",  sum(1 for p in probs if p < 35))

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="ttl">📋 Prediction Timeline</div>', unsafe_allow_html=True)

        for h in hist:
            hc=h["color"]; hp=h["probability"]
            if hp<35:  bbg="rgba(16,185,129,0.10)"; bbd=OK
            elif hp<65: bbg="rgba(245,158,11,0.10)"; bbd=WARN
            else:       bbg="rgba(239,68,68,0.10)";  bbd=DNG

            st.markdown(f"""
            <div class="h-row">
              <span class="h-em">{h['emoji']}</span>
              <span class="h-pct" style="color:{hc};">{hp}%</span>
              <span class="h-bdg">
                <span style="background:{bbg};border:1px solid {bbd};color:{hc};
                  border-radius:20px;padding:2px 10px;font-size:0.71rem;font-weight:700;white-space:nowrap;">
                  {h['risk_level'].upper()}</span>
              </span>
              <span class="h-meta">
                🌧 {h['inputs']['rainfall']:.0f}mm &nbsp;·&nbsp;
                🌊 {h['inputs']['river_level']:.1f}m &nbsp;·&nbsp;
                💨 {h['inputs']['humidity']:.0f}%
              </span>
              <span class="h-ts">{h['timestamp']}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.button("🗑️  Clear History", key="btn_clear"):
            st.session_state.history = []
            st.rerun()