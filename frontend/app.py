"""
Stockley — Stock Price Dashboard
Dark, glassmorphic theme with a cyan/violet accent, grid-based card layout,
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
# THEME TOKENS — dark + cyan/violet accent
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
ACCENT_2    = "#8B5CF6"  # secondary accent, used only in the title gradient

GREEN       = "#00FF99"
GREEN_BG    = "rgba(0,255,153,0.12)"

RED         = "#FF4D6D"
RED_BG      = "rgba(255,77,109,0.12)"

GRID_LINE   = "rgba(255,255,255,0.04)"

# Spacing scale — use these instead of ad-hoc pixel spacers so vertical
# rhythm stays consistent across sections.
SPACE_SM = 8
SPACE_MD = 16
SPACE_LG = 24


def vspace(px: int = SPACE_MD):
    st.markdown(f'<div style="height:{px}px"></div>', unsafe_allow_html=True)


st.markdown(
    f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@500;700&display=swap');

      .stApp {{
            background:
                radial-gradient(circle at top left, rgba(0,229,255,0.08), transparent 35%),
                radial-gradient(circle at bottom right, rgba(139,92,246,0.08), transparent 35%),
                {BG};
        }}

        #MainMenu {{visibility:hidden;}}
        footer {{visibility:hidden;}}
        header {{visibility:hidden;}}

        /* -- Header / page title -------------------------------------- */
        .stockley-header {{ display:flex; align-items:center; gap:14px; margin-bottom: 4px; }}
        .stockley-icon {{ font-size: 30px; }}
        .stockley-title {{
            font-size: 32px;
            font-weight: 800;
            letter-spacing: 0.5px;
            line-height: 1.2;
            background: linear-gradient(90deg, {ACCENT}, {ACCENT_2}, {ACCENT});
            background-size: 200% auto;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: shine 4s linear infinite;
        }}
        @keyframes shine {{
            to {{ background-position: 200% center; }}
        }}
        .stockley-caption {{ font-size: 14px; color: {SUBTEXT}; margin-top: 2px; }}

        /* -- Cards -------------------------------------------------------
           Single source of truth for the glass-card look. Previously this
           selector was defined twice with conflicting values (a flat panel,
           then a glass override) -- only the second ever applied, so the
           first was dead weight. Keeping one definition here. */
        div[data-testid="stVerticalBlock"]:has(> div.stockley-card-marker) {{
            background: linear-gradient(
                135deg,
                rgba(255,255,255,0.06),
                rgba(255,255,255,0.02)
            );
            backdrop-filter: blur(28px);
            -webkit-backdrop-filter: blur(28px);
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 20px;
            padding: 22px 22px 24px 22px;
            box-shadow:
                0 8px 32px rgba(0,0,0,0.45),
                inset 0 1px 0 rgba(255,255,255,0.05);
            transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
        }}
        div[data-testid="stVerticalBlock"]:has(> div.stockley-card-marker):hover {{
            transform: translateY(-4px);
            border-color: rgba(0,229,255,0.35);
            box-shadow:
                0 12px 40px rgba(0,0,0,0.60),
                0 0 25px rgba(0,229,255,0.15),
                0 0 60px rgba(139,92,246,0.08);
        }}
        .stockley-card-marker {{ display:none; }}
        .stockley-card-label {{
            font-size: 12px;
            font-weight: 600;
            letter-spacing: 1.2px;
            text-transform: uppercase;
            color: {SUBTEXT};
            margin-bottom: 12px;
        }}

        /* -- Pills / segmented controls ---------------------------------- */
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

        /* -- Stat cards ---------------------------------------------------
           Numeric values use a monospace face with tabular figures --
           reads like a trading terminal rather than generic dashboard
           sans, and keeps digits aligned as values change. */
        .stat-label {{
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 1.4px;
            text-transform: uppercase;
            color: {SUBTEXT};
            margin-bottom: 8px;
        }}
        .stat-value {{
            font-family: 'JetBrains Mono', ui-monospace, monospace;
            font-variant-numeric: tabular-nums;
            font-size: 30px;
            font-weight: 700;
            color: {TEXT};
            margin-bottom: 10px;
            line-height: 1.15;
        }}
        .quick-ticker {{
            font-family: 'JetBrains Mono', ui-monospace, monospace;
            font-size: 15px;
            font-weight: 700;
            letter-spacing: 0.5px;
            color: {ACCENT};
            margin-bottom: 4px;
        }}
        div[data-testid="column"] button {{
            border-radius: 12px !important;
            border: 1px solid {BORDER} !important;
            background: {PANEL_SOFT} !important;
            color: {SUBTEXT} !important;
            font-size: 12px !important;
        }}
        div[data-testid="column"] button:hover {{
            border-color: {ACCENT} !important;
            color: {ACCENT} !important;
        }}
        .stat-badge {{
            display: inline-block;
            font-family: 'JetBrains Mono', ui-monospace, monospace;
            font-variant-numeric: tabular-nums;
            font-size: 12px;
            font-weight: 600;
            padding: 3px 10px;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.08);
            backdrop-filter: blur(12px);
        }}
        .badge-green {{ background: {GREEN_BG}; color: {GREEN}; }}
        .badge-red   {{ background: {RED_BG};   color: {RED}; }}
        .badge-flat  {{ background: {ACCENT_SOFT}; color: {ACCENT}; }}
        .badge-predicted {{ background: {ACCENT_SOFT}; color: {ACCENT}; font-style: normal; }}

        /* -- Misc ----------------------------------------------------- */
        div[data-testid="stExpander"] {{
            background: rgba(255,255,255,0.03) !important;
            backdrop-filter: blur(24px);
            border-radius: 18px !important;
            border: 1px solid rgba(255,255,255,0.08) !important;
        }}
        div[data-testid="stDataFrame"] {{ border-radius: 10px; overflow: hidden; }}
        div[data-testid="stCaptionContainer"] p {{ color: {SUBTEXT}; }}
        hr {{ border-color: {BORDER}; }}

        ::-webkit-scrollbar {{ width: 8px; }}
        ::-webkit-scrollbar-thumb {{
            background: rgba(0,229,255,0.5);
            border-radius: 20px;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


def format_price(symbol: str, value) -> str:
    """India-listed symbols (.NS / .BSE) are quoted in ₹; everything else here is USD.
    (Note: the main stat cards further down always show ₹ regardless of symbol --
    a pre-existing bug, left as-is since it wasn't part of this change. Happy to
    fix that too if you want.)"""
    if value is None:
        return "N/A"
    currency = "₹" if symbol.endswith((".NS", ".BSE")) else "$"
    return f"{currency}{value:,.2f}"


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
    """,
    unsafe_allow_html=True,
)
vspace(SPACE_LG)


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


@st.cache_data(ttl=300)
def fetch_trending():
    """Top gainers/losers across all tracked stocks, from GET /api/market/trending."""
    try:
        resp = requests.get(f"{API_BASE}/market/trending", timeout=10)
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

QUICK_ACCESS = [
    ("AAPL", "Apple"),
    ("NVDA", "NVIDIA"),
    ("MSFT", "Microsoft"),
    ("GOOGL", "Alphabet"),
    ("AMD", "AMD"),
]


@st.cache_data(ttl=300)
def fetch_quick_stats(symbol: str):
    """Last price + 1-day % change for a quick-access tile, from a small
    5-day window rather than the full detail fetch used by the main chart."""
    try:
        d = fetch_stock_detail(symbol, 5)
    except requests.exceptions.RequestException:
        return None
    day_prices = d.get("prices", [])
    last_price = d.get("last_price") or (day_prices[-1]["close"] if day_prices else None)
    pct = None
    if len(day_prices) >= 2:
        pct = (day_prices[-1]["close"] - day_prices[-2]["close"]) / day_prices[-2]["close"] * 100
    return {"last_price": last_price, "pct_change": pct}


# ----------------------------------------------------------------------------
# ROW 0 — quick-access grid, top 5 global stocks
# ----------------------------------------------------------------------------
with st.container(border=True):
    card_marker()
    st.markdown('<div class="stockley-card-label">Top Global Stocks</div>', unsafe_allow_html=True)
    quick_cols = st.columns(5, gap="medium")
    for col, (sym, display_name) in zip(quick_cols, QUICK_ACCESS):
        with col:
            if sym not in symbol_list:
                st.markdown(
                    f'<div class="quick-ticker">{sym}</div>'
                    f'<div class="stat-value" style="font-size:14px;">Not tracked yet</div>',
                    unsafe_allow_html=True,
                )
                continue

            stats = fetch_quick_stats(sym)
            price_str = format_price(sym, stats["last_price"] if stats else None)
            pct = stats.get("pct_change") if stats else None

            badge = ""
            if pct is not None:
                cls = "badge-green" if pct > 0 else ("badge-red" if pct < 0 else "badge-flat")
                arrow = "↑" if pct > 0 else ("↓" if pct < 0 else "•")
                badge = f'<span class="stat-badge {cls}">{arrow} {abs(pct):.2f}%</span>'

            st.markdown(
                f"""
                <div class="stat-label">{display_name}</div>
                <div class="quick-ticker">{sym}</div>
                <div class="stat-value" style="font-size:20px;">{price_str}</div>
                {badge}
                """,
                unsafe_allow_html=True,
            )
            vspace(SPACE_SM)
            if st.button("View chart", key=f"quick_{sym}", use_container_width=True):
                st.session_state["ticker_pills"] = sym
                st.rerun()

vspace(SPACE_MD)

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
            selection_mode="single", label_visibility="collapsed", key="ticker_pills",
        )
    with ctrl_right:
        st.markdown('<div class="stockley-card-label">Time horizon</div>', unsafe_allow_html=True)
        horizon_label = st.pills(
            "Time horizon", options=list(TIME_HORIZONS.keys()), default="6 Months",
            selection_mode="single", label_visibility="collapsed",
        )

vspace(SPACE_MD)

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
# ROW 2 — 5-column grid of stat cards
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
                '<div class="stat-value" style="font-size:15px; font-family:inherit; color:'
                f'{SUBTEXT};">Model not trained yet</div>',
                unsafe_allow_html=True,
            )

vspace(SPACE_MD)

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

vspace(SPACE_MD)

# ----------------------------------------------------------------------------
# ROW — Peer comparison, all selected stocks normalized to % change so they're
# comparable on one chart regardless of price scale (₹2,000 stock vs $200
# stock). Uses the same time horizon already picked above.
# ----------------------------------------------------------------------------
with st.container(border=True):
    card_marker()
    st.markdown(
        '<div class="stockley-card-label">Peer Comparison — Normalized % Change</div>',
        unsafe_allow_html=True,
    )

    default_peers = [sym for sym, _ in QUICK_ACCESS if sym in symbol_list][:5]
    compare_symbols = st.multiselect(
        "Stocks to compare", options=symbol_list, default=default_peers,
        label_visibility="collapsed",
    )

    if len(compare_symbols) < 2:
        st.caption("Pick at least 2 stocks to compare.")
    else:
        normalized_series = {}
        for sym in compare_symbols:
            try:
                peer_detail = fetch_stock_detail(sym, days)
            except requests.exceptions.RequestException:
                continue
            peer_prices = peer_detail.get("prices", [])
            if len(peer_prices) < 2:
                continue
            peer_df = pd.DataFrame(peer_prices)
            peer_df["date"] = pd.to_datetime(peer_df["date"])
            peer_df = peer_df.sort_values("date")
            base_close = peer_df["close"].iloc[0]
            normalized_series[sym] = pd.Series(
                (peer_df["close"].values / base_close - 1) * 100,
                index=peer_df["date"].values,
            )

        if normalized_series:
            compare_df = pd.DataFrame(normalized_series)
            st.line_chart(compare_df, height=380)
            st.caption(
                f"Rebased to 0% at the start of the selected {horizon_label.lower()} window."
            )
        else:
            st.caption("No data available for the selected stocks.")

vspace(SPACE_MD)

# ----------------------------------------------------------------------------
# ROW — Market Movers, top gainers/losers from GET /api/market/trending
# ----------------------------------------------------------------------------
with st.container(border=True):
    card_marker()
    st.markdown(
        '<div class="stockley-card-label">Market Movers — Today\'s Top Gainers & Losers</div>',
        unsafe_allow_html=True,
    )

    trending = fetch_trending()

    if not trending:
        st.caption("Trending data unavailable — check the backend is running and /api/market/trending is up.")
    else:
        gainers = trending.get("gainers", [])
        losers = trending.get("losers", [])

        mv_left, mv_right = st.columns(2, gap="large")

        with mv_left:
            st.markdown(
                f'<div class="stat-label" style="color:{GREEN};">Top Gainers</div>',
                unsafe_allow_html=True,
            )
            if not gainers:
                st.caption("No gainers to show.")
            for item in gainers[:5]:
                sym = item["symbol"]
                price_str = format_price(sym, item.get("price"))
                pct = item.get("pct_change", 0.0)
                st.markdown(
                    f"""
                    <div style="display:flex; justify-content:space-between; align-items:center;
                                padding:10px 0; border-bottom:1px solid {BORDER};">
                        <div>
                            <div class="quick-ticker">{sym}</div>
                            <div class="stockley-caption">{item.get("name", "")}</div>
                        </div>
                        <div style="text-align:right;">
                            <div class="stat-value" style="font-size:16px; margin-bottom:2px;">{price_str}</div>
                            <span class="stat-badge badge-green">↑ {abs(pct):.2f}%</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with mv_right:
            st.markdown(
                f'<div class="stat-label" style="color:{RED};">Top Losers</div>',
                unsafe_allow_html=True,
            )
            if not losers:
                st.caption("No losers to show.")
            for item in losers[:5]:
                sym = item["symbol"]
                price_str = format_price(sym, item.get("price"))
                pct = item.get("pct_change", 0.0)
                st.markdown(
                    f"""
                    <div style="display:flex; justify-content:space-between; align-items:center;
                                padding:10px 0; border-bottom:1px solid {BORDER};">
                        <div>
                            <div class="quick-ticker">{sym}</div>
                            <div class="stockley-caption">{item.get("name", "")}</div>
                        </div>
                        <div style="text-align:right;">
                            <div class="stat-value" style="font-size:16px; margin-bottom:2px;">{price_str}</div>
                            <span class="stat-badge badge-red">↓ {abs(pct):.2f}%</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

vspace(SPACE_MD)

# ----------------------------------------------------------------------------
# RAW DATA
# ----------------------------------------------------------------------------
with st.expander("View raw data"):
    st.dataframe(df.sort_values("date", ascending=False), use_container_width=True)