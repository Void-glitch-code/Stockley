# Stockley — Stock Price Dashboard & Predictor

A stock price tracking and next-day prediction platform for Indian equities (BSE-listed), built as a final-year project.

**Status:** In progress — Week 3 of an 8-week build. Core pipeline (data → DB → API → ML → dashboard) is working end to end.

---

## What's actually built (as of now)

This deviates from the original 12-week plan in several places, based on real constraints hit during development (documented below under "Key Decisions & Why"). This README reflects the **current, real** implementation.

### Tech Stack

**Backend**
- FastAPI
- PostgreSQL + SQLAlchemy (ORM)
- Alpha Vantage API (data source — not yfinance, see below)
- scikit-learn (Ridge Regression, Random Forest, Gradient Boosting)
- joblib (model persistence)

**Frontend**
- Streamlit (not React — see below)
- streamlit-lightweight-charts for candlestick/volume charts

**Data**
- 10 BSE-listed stocks: Reliance, TCS, HDFC Bank, Infosys, ICICI Bank, Hindustan Unilever, SBI, Bharti Airtel, ITC, Kotak Mahindra Bank
- ~100 trading days of history per stock (free-tier API limit — see below)

---

## Features working right now

- ✅ Historical OHLCV data fetched from Alpha Vantage, stored in PostgreSQL
- ✅ `GET /api/stocks` — list all tracked stocks
- ✅ `GET /api/stocks/{symbol}` — stock detail + historical prices
- ✅ `GET /api/stocks/{symbol}/chart` — chart-ready OHLCV series
- ✅ `GET /api/predict/{symbol}` — next-day closing price prediction
- ✅ Streamlit dashboard: ticker + time horizon selection, candlestick + volume charts, stat cards (last price, day range, avg volume, sector)
- ✅ ML pipeline: feature engineering (lag features, rolling averages, momentum), time-based train/test split (no data leakage), naive-baseline comparison, multi-model selection (Ridge/RandomForest/GradientBoosting picked per stock)

## Not built yet (originally planned, deprioritized for timeline)

- ❌ React frontend (using Streamlit instead — much faster to build, still fully functional)
- ❌ Docker / cloud deployment
- ❌ Authentication, user accounts, watchlist
- ❌ LSTM / Prophet models (using scikit-learn instead — see below)
- ❌ Multi-day (7/14/30-day) forecasting — currently next-day only

---

## Key Decisions & Why

**yfinance → Alpha Vantage.** yfinance repeatedly hit `429 Too Many Requests` and broken-cookie errors due to Yahoo Finance tightening scraping restrictions. Switched to Alpha Vantage's official REST API for reliability. Trade-off: free tier only allows `outputsize=compact` (~100 days of history), not the full multi-year history originally planned.

**Prophet → scikit-learn.** Prophet's `cmdstanpy`/Stan backend was unreliable to install across Python versions. Replaced with scikit-learn regression models, which are simpler to reason about and sufficient for a next-day prediction task at this data scale.

**Predicting returns, not absolute price.** Initial models trained directly on next-day closing price performed *worse* than a naive "tomorrow = today" baseline on every single stock. Root cause: tree-based models (Random Forest, Gradient Boosting) cannot extrapolate beyond the range of values seen during training — since stock prices trend over time, the test period's prices fell outside the training range, and predictions got stuck near the training boundary. Fixed by training on next-day **return** (% change) instead of absolute price, then reconstructing price as `today_close × (1 + predicted_return)`. This is standard practice in quantitative finance and meaningfully improved results.

**Model selection per stock.** Rather than committing to one algorithm, `train.py` trains Ridge, Random Forest, and Gradient Boosting for each stock and keeps whichever performs best on a held-out, time-ordered test set (never randomly shuffled, to avoid leaking future data into training).

**React → Streamlit.** Given the compressed timeline, Streamlit delivers a fully interactive, good-looking dashboard in a fraction of the development time a React + TailwindCSS build would take, while still demonstrating real frontend/UX decisions (custom theming, grid layout, live API integration).

---

## Honest Model Performance Notes

With ~100 days of data per stock and highly liquid, large-cap Indian equities, next-day price movement is close to a random walk (consistent with weak-form market efficiency). After fixing the extrapolation issue above:

- Average model MAPE: ~1.2–1.3%
- Average naive baseline MAPE: ~1.1%
- Model beats the naive baseline on roughly 2/10 stocks

This is reported honestly rather than cherry-picked, and is a legitimate, defensible finding for a final-year project write-up: rigorous methodology (leak-free time-series validation, baseline comparison, multi-model testing) matters more than an artificially "impressive" accuracy number.

---

## Project Structure

```
stockley/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI entrypoint
│   │   ├── database.py          # SQLAlchemy engine/session
│   │   ├── models.py            # Stock, HistoricalPrice tables
│   │   ├── schemas.py           # Pydantic response models
│   │   ├── routes/
│   │   │   ├── stocks.py        # /api/stocks endpoints
│   │   │   └── predict.py       # /api/predict endpoint
│   │   ├── ml/
│   │   │   ├── features.py      # Feature engineering
│   │   │   ├── train.py         # Model training + evaluation
│   │   │   └── saved_models/    # Trained model files (.joblib)
│   │   └── utils/
│   │       └── data_fetcher.py  # Alpha Vantage data pipeline
│   ├── init_db.py               # One-time table creation
│   ├── requirements.txt
│   └── .env.example
└── frontend_streamlit/
    ├── app.py                   # Streamlit dashboard
    └── .streamlit/config.toml   # Theme config
```

---

## Setup

**1. Backend**
```bash
cd backend
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # fill in your Postgres + Alpha Vantage credentials
python init_db.py             # create tables
python -m app.utils.data_fetcher   # populate historical prices
python -m app.ml.train             # train prediction models
uvicorn app.main:app --reload      # start API at localhost:8000
```

**2. Frontend**
```bash
cd frontend_streamlit
streamlit run app.py          # opens at localhost:8501
```

API docs available at `http://localhost:8000/docs` (Swagger UI).

---

## Next Steps

- Surface predictions directly in the Streamlit dashboard
- Set up a recurring data refresh (daily fetch → retrain) as history accumulates past the current ~100-day window
- Multi-day forecasting (currently next-day only)
- Deployment (Railway for backend, Streamlit Community Cloud for frontend)
