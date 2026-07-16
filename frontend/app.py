"""
Stockley — Stock Price Dashboard
Dark, card-based UI (inspired by the "Stock peer analysis" reference layout),
using streamlit-lightweight-charts for the price/volume chart.

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
# THEME TOKENS — match these to .streamlit/config.toml (see file alongside this one)
# ----------------------------------------------------------------------------
BG          = "#0d1220"
PANEL       = "#141a2b"
PANEL_SOFT  = "#111726"
BORDER      = "rgba(255,255,255,0.08)"
TEXT        = "#e7e9ee"
SUBTEXT     = "#8b93a7"
ACCENT      = "#7c6cf6"
ACCENT_SOFT = "rgba(124,108,246,0.16)"
GREEN       = "#34d399"
GREEN_BG    = "rgba(52,211,153,0.14)"
RED         = "#f87171"
RED_BG      = "rgba(248,113,113,0.14)"
GRID_LINE   = "rgba(255,255,255,0.06)"

# ----------------------------------------------------------------------------
# CSS
# Cards are built with a small invisible "marker" div + a `:has()` CSS selector,
# since Streamlit's internal testids for bordered containers change across
# versions and aren't safe to hardcode. `:has()` needs a Chromium/Edge/Safari-
# based browser (2023+) — this covers the vast majority of desktop users.
# ----------------------------------------------------------------------------
st.markdown(
    f"""
    <style>
      .stApp {{ background-color: {BG}; }}

      /* page title header */
      .stockley-header {{ display:flex; align-items:center; gap:14px; margin-bottom: 4px; }}
      .stockley-icon {{ font-size: 30px; }}
      .stockley-title {{ font-size: 28px; font-weight: 700; color: {TEXT}; line-height:1.2; }}
      .stockley-caption {{ font-size: 14px; color: {SUBTEXT}; margin-top: 2px; }}

      /* card panels, activated via marker + :has() */
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

      /* pill / segmented button group (used for ticker + time horizon) */
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

      /* stat blocks (Last Price / Change / Sector) */
      .stat-label {{ font-size: 13px; color: {SUBTEXT}; margin-bottom: 6px; }}
      .stat-value {{ font-size: 26px; font-weight: 700; color: {TEXT}; margin-bottom: 8px; }}
      .stat-badge {{
          display:inline-block; font-size: 12px; font-weight: 600;
          padding: 3px 10px; border-radius: 999px;
      }}
      .badge-green {{ background: {GREEN_BG}; color: {GREEN}; }}
      .badge-red   {{ background: {RED_BG};   color: {RED}; }}
      .badge-flat  {{ background: {ACCENT_SOFT}; color: {ACCENT}; }}

      /* dataframe + expander */
      div[data-testid="stExpander"] {{
          background: {PANEL}; border: 1px solid {BORDER}; border-radius: 12px;
      }}
      div[data-testid="stDataFrame"] {{ border-radius: 10px; overflow: hidden; }}

      /* misc */
      div[data-testid="stCaptionContainer"] p {{ color: {SUBTEXT}; }}
      hr {{ border-color: {BORDER}; }}
    </style>
    """,
    unsafe_allow_html=True,
)


def card_marker():
    """Call as the first line inside a `with st.container(border=True):` block
    to make that container pick up the .stockley-card panel styling above."""
    st.markdown('<div class="stockley-card-marker"></div>', unsafe_allow_html=True)


def stat_badge_html(label: str, value: str, pct: float | None = None) -> str:
    badge = ""
    if pct is not None:
        cls = "badge-green" if pct > 0 else ("badge-red" if pct < 0 else "badge-flat")
        arrow = "↑" if pct > 0 else ("↓" if pct < 0 else "•")
        badge = f'<span class="stat-badge {cls}">{arrow} {abs(pct):.2f}%</span>'
    return f"""
    <div>
        <div class="stat-label">{label}</div>
        <div class="stat-value">{value}</div>
        {badge}
    </div>
    """


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
@st.cache_data(ttl=300)  # cache for 5 minutes so we don't hammer the API on every rerun
def fetch_stocks():
    resp = requests.get(f"{API_BASE}/stocks", timeout=10)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=300)
def fetch_stock_detail(symbol: str, days: int):
    resp = requests.get(f"{API_BASE}/stocks/{symbol}", params={"days": days}, timeout=10)
    resp.raise_for_status()
    return resp.json()


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
# LAYOUT — controls (left) + chart (right), like the reference dashboard
# ----------------------------------------------------------------------------
left, right = st.columns([1, 2.6], gap="medium")

with left:
    with st.container(border=True):
        card_marker()
        st.markdown('<div class="stockley-card-label">Stock ticker</div>', unsafe_allow_html=True)
        selected_symbol = st.pills(
            "Stock ticker",
            options=symbol_list,
            default=symbol_list[0],
            selection_mode="single",
            label_visibility="collapsed",
        )
        st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)
        st.markdown('<div class="stockley-card-label">Time horizon</div>', unsafe_allow_html=True)
        horizon_label = st.pills(
            "Time horizon",
            options=list(TIME_HORIZONS.keys()),
            default="6 Months",
            selection_mode="single",
            label_visibility="collapsed",
        )

    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)

    selected_symbol = selected_symbol or symbol_list[0]
    horizon_label = horizon_label or "6 Months"
    days = TIME_HORIZONS[horizon_label]

    detail = fetch_stock_detail(selected_symbol, days)
    prices = detail["prices"]

    if not prices:
        st.warning("No historical price data for this stock yet.")
        st.stop()

    df = pd.DataFrame(prices)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    last_price = detail.get("last_price")
    pct_change = None
    change = None
    if len(df) >= 2:
        change = df["close"].iloc[-1] - df["close"].iloc[-2]
        pct_change = (change / df["close"].iloc[-2]) * 100

    with st.container(border=True):
        card_marker()
        st.markdown(
            stat_badge_html(
                "Last price",
                f"₹{last_price:,.2f}" if last_price else "N/A",
                pct_change,
            ),
            unsafe_allow_html=True,
        )
        st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
        st.markdown(
            stat_badge_html("Sector", detail.get("sector") or "N/A"),
            unsafe_allow_html=True,
        )

with right:
    with st.container(border=True):
        card_marker()
        st.markdown(
            f'<div class="stockley-card-label">{selected_symbol} — {name_by_symbol.get(selected_symbol, "")} '
            f"· last {len(df)} trading days</div>",
            unsafe_allow_html=True,
        )

        candle_data = [
            {
                "time": row.date.strftime("%Y-%m-%d"),
                "open": row.open,
                "high": row.high,
                "low": row.low,
                "close": row.close,
            }
            for row in df.itertuples()
        ]
        volume_data = [
            {
                "time": row.date.strftime("%Y-%m-%d"),
                "value": row.volume,
                "color": GREEN if row.close >= row.open else RED,
            }
            for row in df.itertuples()
        ]

        chart_options = {
            "height": 400,
            "layout": {
                "background": {"type": "solid", "color": PANEL},
                "textColor": SUBTEXT,
            },
            "grid": {
                "vertLines": {"color": GRID_LINE},
                "horzLines": {"color": GRID_LINE},
            },
            "rightPriceScale": {"borderColor": BORDER},
            "timeScale": {"borderColor": BORDER},
            "crosshair": {"mode": 0},
        }
        volume_options = {
            "height": 120,
            "layout": {
                "background": {"type": "solid", "color": PANEL},
                "textColor": SUBTEXT,
            },
            "grid": {
                "vertLines": {"color": GRID_LINE},
                "horzLines": {"color": GRID_LINE},
            },
            "rightPriceScale": {"borderColor": BORDER},
            "timeScale": {"borderColor": BORDER, "visible": True},
        }

        renderLightweightCharts(
            [
                {
                    "chart": chart_options,
                    "series": [
                        {
                            "type": "Candlestick",
                            "data": candle_data,
                            "options": {
                                "upColor": GREEN,
                                "downColor": RED,
                                "borderVisible": False,
                                "wickUpColor": GREEN,
                                "wickDownColor": RED,
                            },
                        }
                    ],
                },
                {
                    "chart": volume_options,
                    "series": [
                        {
                            "type": "Histogram",
                            "data": volume_data,
                            "options": {"priceFormat": {"type": "volume"}},
                        }
                    ],
                },
            ],
            key=f"chart-{selected_symbol}-{days}",
        )

# ----------------------------------------------------------------------------
# RAW DATA
# ----------------------------------------------------------------------------
st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)
with st.expander("View raw data"):
    st.dataframe(
        df.sort_values("date", ascending=False),
        use_container_width=True
    )
    