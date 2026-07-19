# Stockley

A full-stack stock price dashboard and next-day price predictor for 30 stocks across US and Indian (NSE/BSE) markets — FastAPI backend, PostgreSQL, scikit-learn models, and a Streamlit frontend.

> **On the predictions:** this project deliberately benchmarks every model against a naive "tomorrow = today" baseline, and reports the result honestly even when the model loses. See [Results & methodology](#results--methodology) for why that matters more than the accuracy number itself.

---

## What it does

- Tracks daily OHLCV price history for 30 stocks: 10 NSE-listed and 10 BSE-listed Indian stocks, plus 10 global stocks (US-listed).
- Trains a next-day closing price model per stock (Ridge / Random Forest / Gradient Boosting, auto-selected via cross-validation).
- Serves everything through a FastAPI backend, with a Streamlit dashboard for browsing price history, viewing predictions, and comparing stocks against each other on a normalized chart.

## Tech stack

| Layer | Tools |
|---|---|
| Backend API | FastAPI, SQLAlchemy |
| Database | PostgreSQL |
| ML | scikit-learn (Ridge, RandomForestRegressor, GradientBoostingRegressor), pandas, joblib |
| Frontend | Streamlit, streamlit-lightweight-charts |
| Data sources | Alpha Vantage (free tier: 10 BSE + 5 global tickers) and manually downloaded NSE/global CSVs (nseindia.com) for the rest |

## Project structure

```
backend/
  app/
    ml/
      features.py       # feature engineering (lags, rolling stats, momentum)
      train.py           # trains + evaluates one model per stock
      saved_models/      # one .joblib file per stock
    utils/
      import_nse_csv.py     # one-time import for manually downloaded NSE CSVs
      import_manual_csv.py  # one-time import for manually downloaded global CSVs
  data_import/
    global/              # manually downloaded global stock CSVs
    nse/                 # manually downloaded NSE stock CSVs
  init_db.py
frontend/
  app.py                 # Streamlit dashboard
```

## Data pipeline

Data comes from two sources per stock, since Alpha Vantage's free tier only covers 10 BSE + 5 global tickers:

1. **Alpha Vantage API** — auto-fetched for the tickers the free tier supports.
2. **Manual CSV import** — the remaining NSE and global tickers are downloaded by hand from nseindia.com / respective sources and imported via `import_nse_csv.py` / `import_manual_csv.py`. These handle Indian-style comma-grouped numbers (`1,83,02,021`) and mixed date formats.

NSE and BSE listings for the same company (e.g. `RELIANCE.NS` vs `RELIANCE.BSE`) are tracked as separate records rather than merged, since the two exchanges can report slightly different prices.

## Results & methodology

Each stock gets its own model, evaluated with a **time-based train/test split** (never shuffled — shuffling time series data leaks the future into training). The target is next-day *return*, not absolute price, since tree-based models can't extrapolate past the price range they were trained on.

**Every model is benchmarked against a naive baseline** ("tomorrow's close = today's close"). This matters because daily stock returns are close to a random walk — that's a foundational result in finance, not a flaw in this codebase. A model that doesn't beat the naive baseline on a given stock isn't necessarily broken; it may just mean there's no exploitable signal in OHLCV history alone for that stock over that period.

Current results: **the model beats the naive baseline on roughly 5–10 of 30 stocks**, depending on the run, with average MAPE close to (and sometimes slightly worse than) the baseline's. That's the expected range for this task, not a bug.

**Model selection is done via time-series cross-validation on the training set only** — an earlier version of this pipeline picked the "best" of three candidate models by lowest error *on the test set itself*, which is a subtle form of leakage: it tunes model choice to the exact data later reported as the result, inflating the "beats baseline" count with noise (test sets here are only 16–45 rows). Fixing that dropped the reported win count, which is the correct direction — the earlier number was optimistic, not the current one.

## Setup

```bash
git clone <repo-url>
cd Stockley

# backend
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
cp .env.example .env         # fill in your DB connection string + Alpha Vantage key
python init_db.py
python -m app.utils.import_nse_csv
python -m app.utils.import_manual_csv
python -m app.ml.train
uvicorn app.main:app --reload

# frontend (separate terminal)
cd frontend
pip install streamlit streamlit-lightweight-charts pandas requests
streamlit run app.py
```

The frontend expects the backend running at `http://localhost:8000`.

## Known limitations

- Two NSE tickers (`HINDUNILVR.NS`, `ITC.NS`) currently have shorter price history than the rest, since their source CSVs covered a narrower date range — models for these are trained on less data than the others.
- Free-tier Alpha Vantage access caps automated fetching at 10 BSE + 5 global tickers; the rest rely on manual CSV re-download to stay current.
- Predictions are next-day close only. Given how close daily returns are to a random walk, a longer prediction horizon (e.g. 5-day) or a directional (up/down) classification target would likely be a more honest and more learnable problem — worth exploring next.

## License

MIT
