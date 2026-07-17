"""
Stockley — Stock Price Dashboard
Warm dark theme, grid-based card layout, using streamlit-lightweight-charts
for the price/volume chart.

Run with: streamlit run app.py
Requires the FastAPI backend running at http://localhost:8000
Install deps: pip install streamlit streamlit-lightweight-charts pandas requests
"""
import requests
import streamlit as st
import pandas as pd
from streamlit_lightweight_charts import renderLightweightCharts

API_BASE = "http://localhost:8000/api"

st.set_page_config(page_title="Stockley", page_icon="📈", layout="wide")

# ----------------------------------------------------------------------------
# THEME TOKENS — warm dark + amber/gold accent
# Keep these in sync with .streamlit/config.toml (see file alongside this one)
# ----------------------------------------------------------------------------
BG          = "#050505"
PANEL       = "rgba(20,20,20,0.85)"
PANEL_SOFT  = "rgba(12,12,12,0.75)"
BORDER      = "rgba(255,255,255,0.08)"

TEXT        = "#FFFFFF"
SUBTEXT     = "#9CA3AF"

ACCENT      = "#00E5FF"
ACCENT_SOFT = "rgba(0,229,255,0.15)"

GREEN       = "#00FF99"
GREEN_BG    = "rgba(0,255,153,0.12)"

RED         = "#FF4D6D"
RED_BG      = "rgba(255,77,109,0.12)"

GRID_LINE   = "rgba(255,255,255,0.04)"

st.markdown(
    f"""
    <style>
      .stApp {{
            background:
                radial-gradient(circle at top left, rgba(0,229,255,0.08), transparent 35%),
                radial-gradient(circle at bottom right, rgba(139,92,246,0.08), transparent 35%),
                #050505;
        }}

        #MainMenu {{visibility:hidden;}}
        footer {{visibility:hidden;}}
        header {{visibility:hidden;}}

        .stockley-title {{
            font-size:36px;
            font-weight:800;
            letter-spacing:1px;
            background: linear-gradient(90deg,#00E5FF,#8B5CF6,#00E5FF);
            background-size:200% auto;
            -webkit-background-clip:text;
            -webkit-text-fill-color:transparent;
            animation: shine 4s linear infinite;
        }}

        @keyframes shine {{
            to {{ background-position:200% center; }}
        }}
        

      .stockley-header {{ display:flex; align-items:center; gap:14px; margin-bottom: 4px; }}
      .stockley-icon {{ font-size: 30px; }}
      .stockley-title {{ font-size: 28px; font-weight: 700; color: {TEXT}; line-height:1.2; }}
      .stockley-caption {{ font-size: 14px; color: {SUBTEXT}; margin-top: 2px; }}

      div[data-testid="stVerticalBlock"]:has(> div.stockley-card-marker) {{
          background: {PANEL};
          border: 1px solid {BORDER};
          border-radius: 14px;
          padding: 18px 20px 20px 20px;
      }}
      .stockley-card-marker {{ display:none; }}
      .stockley-card-label {{
          font-size: 13px; color: {SUBTEXT}; font-weight: 500;
          text-transform: none; margin-bottom: 10px;
      }}

      div[data-testid="stButtonGroup"] button {{
          border-radius: 999px !important;
          border: 1px solid {BORDER} !important;
          background: {PANEL_SOFT} !important;
          color: {TEXT} !important;
      }}
      div[data-testid="stButtonGroup"] button[kind="primary"],
      div[data-testid="stButtonGroup"] button[aria-checked="true"] {{
          background: {ACCENT_SOFT} !important;
          border-color: {ACCENT} !important;
          color: {ACCENT} !important;
      }}

      .stat-label {{ font-size: 13px; color: {SUBTEXT}; margin-bottom: 6px; }}
      .stat-value {{ font-size: 24px; font-weight: 700; color: {TEXT}; margin-bottom: 8px; }}
      .stat-badge {{
          display:inline-block; font-size: 12px; font-weight: 600;
          padding: 3px 10px; border-radius: 999px;
      }}
      .badge-green {{ background: {GREEN_BG}; color: {GREEN}; }}
      .badge-red   {{ background: {RED_BG};   color: {RED}; }}
      .badge-flat  {{ background: {ACCENT_SOFT}; color: {ACCENT}; }}
      .badge-predicted {{ background: {ACCENT_SOFT}; color: {ACCENT}; font-style: normal; }}

      div[data-testid="stExpander"] {{
          background: {PANEL}; border: 1px solid {BORDER}; border-radius: 12px;
      }}
      div[data-testid="stDataFrame"] {{ border-radius: 10px; overflow: hidden; }}
      div[data-testid="stCaptionContainer"] p {{ color: {SUBTEXT}; }}
      hr {{ border-color: {BORDER}; }}
    
      div[data-testid="stVerticalBlock"]:has(> div.stockley-card-marker) {{
          background: linear-gradient(
              135deg,
              rgba(255,255,255,0.06),
              rgba(255,255,255,0.02)
          );
          backdrop-filter: blur(28px);
          -webkit-backdrop-filter: blur(28px);

          border: 1px solid rgba(255,255,255,0.10);
          border-radius: 24px;

          box-shadow:
              0 8px 32px rgba(0,0,0,0.45),
              inset 0 1px 0 rgba(255,255,255,0.05);

          padding: 24px;
          transition: all 0.3s ease;
      }}

      div[data-testid="stVerticalBlock"]:has(> div.stockley-card-marker):hover {{
          transform: translateY(-6px);
          border-color: rgba(0,229,255,0.35);

          box-shadow:
              0 12px 40px rgba(0,0,0,0.60),
              0 0 25px rgba(0,229,255,0.15),
              0 0 60px rgba(139,92,246,0.08);
      }}

      .stat-value {{
          font-size: 34px !important;
          font-weight: 800 !important;
          color: #ffffff !important;
      }}

      .stat-label {{
          text-transform: uppercase;
          letter-spacing: 1.4px;
          font-size: 11px !important;
          color: #9CA3AF !important;
      }}

      .stat-badge {{
          border: 1px solid rgba(255,255,255,0.08);
          backdrop-filter: blur(12px);
      }}

      div[data-testid="stExpander"] {{
          background: rgba(255,255,255,0.03) !important;
          backdrop-filter: blur(24px);
          border-radius: 20px !important;
          border: 1px solid rgba(255,255,255,0.08) !important;
      }}

      ::-webkit-scrollbar {{
          width: 8px;
      }}

      ::-webkit-scrollbar-thumb {{
          background: rgba(0,229,255,0.5);
          border-radius: 20px;
      }}
    
</style>
    """,
    unsafe_allow_html=True,
)


def card_marker():
    st.markdown('<div class="stockley-card-marker"></div>', unsafe_allow_html=True)


def stat_card(label: str, value: str, pct=None):
    """Renders one grid cell: a bordered card with a label, big value, and optional badge."""
    badge = ""
    if pct is not None:
        cls = "badge-green" if pct > 0 else ("badge-red" if pct < 0 else "badge-flat")
        arrow = "↑" if pct > 0 else ("↓" if pct < 0 else "•")
        badge = f'<span class="stat-badge {cls}">{arrow} {abs(pct):.2f}%</span>'
    with st.container(border=True):
        card_marker()
        st.markdown(
            f"""
            <div class="stat-label">{label}</div>
            <div class="stat-value">{value}</div>
            {badge}
            """,
            unsafe_allow_html=True,
        )


# ----------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="stockley-header">
        <span class="stockley-icon">📈</span>
        <div>
            <div class="stockley-title">Stockley</div>
            <div class="stockley-caption">Real historical data, served from your FastAPI backend.</div>
        </div>
    </div>
    <div style="height:18px"></div>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# DATA FETCHING
# ----------------------------------------------------------------------------
@st.cache_data(ttl=300)
def fetch_stocks():
    resp = requests.get(f"{API_BASE}/stocks", timeout=10)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=300)
def fetch_stock_detail(symbol: str, days: int):
    resp = requests.get(f"{API_BASE}/stocks/{symbol}", params={"days": days}, timeout=10)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=300)
def fetch_prediction(symbol: str):
    """Returns None if no model is trained yet for this symbol, instead of crashing."""
    try:
        resp = requests.get(f"{API_BASE}/predict/{symbol}", timeout=10)
        if resp.status_code == 503:
            return None  # no trained model yet
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException:
        return None


try:
    stocks = fetch_stocks()
except requests.exceptions.ConnectionError:
    st.error(
        "Can't reach the backend at http://localhost:8000. "
        "Make sure `uvicorn app.main:app --reload` is running."
    )
    st.stop()
except requests.exceptions.HTTPError as e:
    st.error(f"Backend returned an error: {e}")
    st.stop()

if not stocks:
    st.warning("No stocks found in the database yet. Run the data fetcher first.")
    st.stop()

symbol_list = [s["symbol"] for s in stocks]
name_by_symbol = {s["symbol"]: s["name"] for s in stocks}

TIME_HORIZONS = {
    "1 Month": 21,
    "3 Months": 63,
    "6 Months": 126,
    "1 Year": 252,
    "5 Years": 1260,
    "10 Years": 2520,
}

# ----------------------------------------------------------------------------
# ROW 1 — controls card, full width
# ----------------------------------------------------------------------------
with st.container(border=True):
    card_marker()
    ctrl_left, ctrl_right = st.columns(2, gap="large")
    with ctrl_left:
        st.markdown('<div class="stockley-card-label">Stock ticker</div>', unsafe_allow_html=True)
        selected_symbol = st.pills(
            "Stock ticker", options=symbol_list, default=symbol_list[0],
            selection_mode="single", label_visibility="collapsed",
        )
    with ctrl_right:
        st.markdown('<div class="stockley-card-label">Time horizon</div>', unsafe_allow_html=True)
        horizon_label = st.pills(
            "Time horizon", options=list(TIME_HORIZONS.keys()), default="6 Months",
            selection_mode="single", label_visibility="collapsed",
        )

st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)

selected_symbol = selected_symbol or symbol_list[0]
horizon_label = horizon_label or "6 Months"
days = TIME_HORIZONS[horizon_label]

detail = fetch_stock_detail(selected_symbol, days)
prediction = fetch_prediction(selected_symbol)
prices = detail["prices"]

if not prices:
    st.warning("No historical price data for this stock yet.")
    st.stop()

df = pd.DataFrame(prices)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

last_price = detail.get("last_price")
pct_change = None
if len(df) >= 2:
    change = df["close"].iloc[-1] - df["close"].iloc[-2]
    pct_change = (change / df["close"].iloc[-2]) * 100

day_high = df["high"].iloc[-1]
day_low = df["low"].iloc[-1]
avg_volume = int(df["volume"].tail(10).mean())

# ----------------------------------------------------------------------------
# ROW 2 — 4-column grid of stat cards
# ----------------------------------------------------------------------------
c1, c2, c3, c4, c5 = st.columns(5, gap="medium")
with c1:
    stat_card("Last Price", f"₹{last_price:,.2f}" if last_price else "N/A", pct_change)
with c2:
    stat_card("Day Range", f"₹{day_low:,.2f} – ₹{day_high:,.2f}")
with c3:
    stat_card("Avg Volume (10d)", f"{avg_volume:,}")
with c4:
    stat_card("Sector", detail.get("sector") or "N/A")
with c5:
    if prediction:
        stat_card(
            "Predicted Next Close",
            f"₹{prediction['predicted_next_close']:,.2f}",
            prediction["predicted_change_pct"],
        )
    else:
        with st.container(border=True):
            card_marker()
            st.markdown(
                '<div class="stat-label">Predicted Next Close</div>'
                '<div class="stat-value" style="font-size:15px; color:'
                f'{SUBTEXT};">Model not trained yet</div>',
                unsafe_allow_html=True,
            )

st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# ROW 3 — chart card, full width
# ----------------------------------------------------------------------------
with st.container(border=True):
    card_marker()
    st.markdown(
        f'<div class="stockley-card-label">{selected_symbol} — {name_by_symbol.get(selected_symbol, "")} '
        f"· last {len(df)} trading days</div>",
        unsafe_allow_html=True,
    )

    candle_data = [
        {"time": row.date.strftime("%Y-%m-%d"), "open": row.open,
         "high": row.high, "low": row.low, "close": row.close}
        for row in df.itertuples()
    ]
    volume_data = [
        {"time": row.date.strftime("%Y-%m-%d"), "value": row.volume,
         "color": GREEN if row.close >= row.open else RED}
        for row in df.itertuples()
    ]

    chart_options = {
        "height": 650,
        "layout": {"background": {"type": "solid", "color": PANEL}, "textColor": SUBTEXT},
        "grid": {"vertLines": {"color": GRID_LINE}, "horzLines": {"color": GRID_LINE}},
        "rightPriceScale": {"borderColor": BORDER},
        "timeScale": {"borderColor": BORDER},
        "crosshair": {"mode": 0},
    }
    volume_options = {
        "height": 160,
        "layout": {"background": {"type": "solid", "color": PANEL}, "textColor": SUBTEXT},
        "grid": {"vertLines": {"color": GRID_LINE}, "horzLines": {"color": GRID_LINE}},
        "rightPriceScale": {"borderColor": BORDER},
        "timeScale": {"borderColor": BORDER, "visible": True},
    }

    candlestick_series = {
        "type": "Candlestick", "data": candle_data,
        "options": {
            "upColor": GREEN, "downColor": RED, "borderVisible": False,
            "wickUpColor": GREEN, "wickDownColor": RED,
        },
    }
    if prediction:
        candlestick_series["priceLines"] = [{
            "price": prediction["predicted_next_close"],
            "color": ACCENT,
            "lineWidth": 2,
            "lineStyle": 2,  # dashed
            "axisLabelVisible": True,
            "title": "Predicted",
        }]

    renderLightweightCharts(
        [
            {
                "chart": chart_options,
                "series": [candlestick_series],
            },
            {
                "chart": volume_options,
                "series": [{"type": "Histogram", "data": volume_data,
                            "options": {"priceFormat": {"type": "volume"}}}],
            },
        ],
        key=f"chart-{selected_symbol}-{days}",
    )

# ----------------------------------------------------------------------------
# RAW DATA
# ----------------------------------------------------------------------------
st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)
with st.expander("View raw data"):
    st.dataframe(df.sort_values("date", ascending=False), use_container_width=True)