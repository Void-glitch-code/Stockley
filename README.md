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
      features.py           # feature engineering (lags, rolling stats, momentum)
      train.py               # trains + evaluates one regression model per stock
      train_direction.py     # trains + evaluates one up/down direction classifier per stock
      backtest.py             # walk-forward backtest (multiple folds) for both, run after training
      saved_models/           # one .joblib file per stock (regression)
      saved_models_direction/ # one .joblib file per stock (direction)
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

**Every model is benchmarked against a naive baseline** — "tomorrow's close = today's close" for the regression model, and the **majority class in the training data** (not a 50/50 coin flip) for the direction classifier. This matters because daily stock returns are close to a random walk — that's a foundational result in finance, not a flaw in this codebase. A model that doesn't beat the naive baseline on a given stock isn't necessarily broken; it may just mean there's no exploitable signal in OHLCV history alone for that stock over that period.

**Headline result — walk-forward backtest** (`backtest.py`), the most trustworthy number in this project since it's averaged across up to 5 sequential train/test windows per stock rather than one:

| | Stocks beating baseline | Avg metric (model vs. baseline) |
|---|---|---|
| Regression (next-day close) | 6/30 | MAE 12.03 vs. 11.92 |
| Direction (up/down) | 11/30 | Accuracy 0.507 vs. 0.517 |

Read those two rows together, not separately: direction "wins" on more individual stocks, but its *average* accuracy is still slightly below baseline — meaning the wins on some stocks are being offset by losses on others, not a genuine net edge. Fold-to-fold spread (MAE std ≈2.69, accuracy std ≈0.064) is large enough that several of the individual "beats baseline" results aren't reliable from one window to the next.

**Conclusion: across 30 stocks, two target formulations, and a proper walk-forward evaluation, there's no consistent, exploitable signal in daily OHLCV history for next-day price or direction.** That's not a failed project — a rigorous negative result, arrived at honestly, is the actual point here. Predicting daily stock movement from price history alone is close to the hardest version of this problem you could pick, and the evidence here is consistent with markets pricing that information in efficiently.

**Two methodology notes worth calling out on their own:**
- **Model selection is done via time-series cross-validation on the training set only.** An earlier version of this pipeline picked the "best" of several candidates by lowest error *on the test set itself* — a subtle form of leakage that tunes model choice to the exact data later reported as the result, inflating "beats baseline" counts with noise (test sets here are as small as 16–45 rows). Fixing that lowered the reported win count, which was the correct direction: the earlier number was optimistic, not this one.
- **The walk-forward backtest exists because even the leak-free single-split evaluation is still just one noisy sample per stock.** Running the same evaluation across multiple sequential windows and reporting the spread, not just the mean, is what actually distinguishes "this stock has a real edge" from "this stock got lucky once."

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
python -m app.ml.train              # regression models
python -m app.ml.train_direction    # direction classifiers
python -m app.ml.backtest           # walk-forward evaluation of both (optional, slower)
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
- Both next-day close (regression) and next-day direction (classification) are implemented; neither shows a consistent edge over baseline in the walk-forward backtest (see Results above). A longer prediction horizon (e.g. 5-day) hasn't been tried yet and might behave differently, since it's a somewhat different question than next-day movement.

## License

MIT
