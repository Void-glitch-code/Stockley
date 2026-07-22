# Stockley — Remaining Plan

Where things stand and what's left, in order. ~3 months of runway.

---

## Order of operations

1. **Week 5-6 backend features** (next up, starting order agreed: auth → trending/comparison/sector → watchlist)
2. **Advanced features** (forecasting, performance stats, TradingView links, education layer)
3. **Frontend revamp** — full rebuild in React + Tailwind, Binance-inspired look (targeted for end of August — not before)
4. **Deployment** (Docker + live hosting) — last

Rationale for this order: UI is getting rebuilt from scratch later anyway, so there's no point polishing the current Streamlit UI beyond "functional." Backend substance and the ML story are the parts that survive the eventual frontend rewrite and matter most for interviews — build those first.

---

## 1. Week 5-6 backend features

**Note going in:** this is the most generic part of the whole build — every CRUD tutorial has auth + watchlist. Worth keeping lean here rather than over-investing, so there's more time for the parts that actually differentiate the project (see section 2).

### 1a. Auth (start here)
- `User` model (SQLAlchemy) — needs your current `models.py`/`database.py`/`main.py` to build against correctly
- Register / login endpoints, JWT issuing + verification
- Password hashing (bcrypt via passlib)
- `get_current_user` dependency, used to protect watchlist endpoints later

### 1b. Comparison / trending / sector endpoints (second)
- `GET /api/stocks/{symbol}/comparison?compare_with=...`
- `GET /api/market/trending` — top gainers/losers
- `GET /api/stocks/sector/{sector}`
- `GET /api/stocks?sector=...&min_price=...&max_price=...` filtering

### 1c. Watchlist (last)
- `POST /api/watchlist`, `GET /api/watchlist`, `DELETE /api/watchlist/{symbol}`
- Depends on auth being in place first

---

## 2. Advanced features — the differentiating layer

This is where most of the remaining 3 months should go. Framing: **Stockley isn't a trading platform — it's for someone new to the stock market who wants to understand what's going on and what their options are.** Every feature below should serve that framing, not "predict prices better."

### Time series forecasting (extend existing work)
- Extend `train.py`'s regression target beyond next-day to 5-day / 20-day horizons
- Show as a fan chart: predicted path + widening confidence band, not a single point
- Wider horizon = more visible uncertainty = itself an educational moment about forecast confidence

### "How is the stock performing" panel
- Derived stats per stock: 1W / 1M / 3M / 6M / YTD / 1Y return, rolling volatility, position within 52-week range
- All computable from data already stored — no new data sources needed

### TradingView integration
- Quick: "Open in TradingView" link per symbol (`tradingview.com/symbols/NSE-{SYMBOL}/`, adjusted per exchange)
- Better: embed TradingView's free Symbol Overview widget (iframe/script, no API key) for a live mini-chart

### Education layer (the mission-critical piece, given the "beginner" framing)
- **Glossary** — candlestick, volume, P/E, market cap, volatility, dividend yield — plain-language, linked contextually from wherever those terms appear
- **"How to actually start investing in India" primer** — demat/trading accounts, KYC, stocks vs. mutual funds vs. index funds vs. SIPs, why diversification matters
- **Narrative tie-back to the ML results**: the walk-forward backtest found no reliable single-stock edge — that's not a flaw, it's evidence for *why* diversification beats stock-picking. This connects the technical work to the educational mission and is a genuinely strong thread to walk an interviewer through.
- **Lightweight risk-profile quiz** (3-4 questions) pointing to *categories* of options (index funds, blue-chip stocks, etc.), not specific stock picks — clearly labeled as educational, not financial advice

---

## 3. Data cleanup (fold in whenever convenient, before final deploy)

- Currently 10 Indian companies are listed on **both** `.BSE` and `.NS` (e.g. `TCS.BSE` and `TCS.NS`) — this double-counts rather than adding real coverage, since large-cap NSE/BSE prices track each other closely via arbitrage.
- Fix: one exchange per company, spread across both — e.g. `TCS.NS`, `RELIANCE.BSE`, `INFY.NS`, `HDFCBANK.BSE`, etc. — giving 20 genuinely distinct Indian companies instead of 10 duplicated ones.
- Requires: deciding which 10 of the current 20 rows to keep, deleting the other 10 (and their `historical_prices`) from the DB, retraining those models.

---

## 4. Frontend revamp (end of August)

- Full rebuild in React + TailwindCSS, Binance-inspired visual direction
- Not started until the above is in good shape — no point polishing Streamlit further given this is coming

---

## 5. Deployment (last)

- Dockerize FastAPI backend + frontend
- Deploy free-tier (Render/Railway for API, Vercel for React frontend once built)
- This is typically one of the first things an interviewer checks for in a portfolio project — don't let it slip to "someday"

---

## Already done (for reference)

- Full data pipeline: Alpha Vantage + manual CSV import (NSE + global), handling Indian comma-grouped numbers and mixed date formats
- 30 stocks tracked, PostgreSQL-backed
- Regression model (`train.py`) + direction classifier (`train_direction.py`), both with **leak-free CV-based model selection** (fixed a real test-set leakage bug mid-project)
- Walk-forward backtest (`backtest.py`) — the most rigorous evaluation in the project, pooled across up to 5 folds per stock
- Honest headline finding: no consistent, exploitable signal in daily OHLCV history for next-day price or direction across 30 stocks — a real negative result, properly earned
- Streamlit dashboard: candlestick + volume chart, prediction overlay, top-5-global quick-access grid, normalized peer-comparison chart
- Strong README documenting methodology, the leakage fix, and results honestly

**Deliberate deviations from the original plan, with reasons:**
- Dropped yfinance (rate-limit instability) in favor of Alpha Vantage + manual CSVs
- Dropped Prophet (poor fit for stock data, black-box confidence intervals) in favor of Ridge/RandomForest/GradientBoosting with explainable, CV-validated selection
