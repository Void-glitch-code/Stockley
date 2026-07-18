"""
Fetches historical OHLCV data from Alpha Vantage and stores it in the
database. Free tier limits: 5 requests/minute, 25 requests/day, and only
outputsize=compact (last ~100 trading days) -- "full" history is now a
premium-only feature. 100 days is enough to train a baseline model on.

Usage: python -m app.utils.data_fetcher
"""
import os
import time
import requests
from datetime import datetime
from sqlalchemy.orm import Session
from dotenv import load_dotenv

from app.database import SessionLocal
from app.models import Stock, HistoricalPrice

load_dotenv()

API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
BASE_URL = "https://www.alphavantage.co/query"

# Alpha Vantage uses ".BSE" suffix for Indian stocks (not ".NS")
STOCK_SYMBOLS = {
    # --- Indian equities (BSE) -- 10 calls ---
    "RELIANCE.BSE": ("Reliance Industries", "Energy"),
    "TCS.BSE": ("Tata Consultancy Services", "IT"),
    "HDFCBANK.BSE": ("HDFC Bank", "Finance"),
    "INFY.BSE": ("Infosys", "IT"),
    "ICICIBANK.BSE": ("ICICI Bank", "Finance"),
    "HINDUNILVR.BSE": ("Hindustan Unilever", "FMCG"),
    "SBIN.BSE": ("State Bank of India", "Finance"),
    "BHARTIARTL.BSE": ("Bharti Airtel", "Telecom"),
    "ITC.BSE": ("ITC Limited", "FMCG"),
    "KOTAKBANK.BSE": ("Kotak Mahindra Bank", "Finance"),

    # --- Global mega-caps via Alpha Vantage -- 5 calls ---
    # Plain tickers -- Alpha Vantage supports US-listed stocks natively, no suffix needed.
    # Total: 10 + 5 = 15 calls/day, well under the 25/day free-tier cap.
    # Apple, NVIDIA, Meta, Intel, AMD are handled separately via manual CSV
    # backfill instead (see import_manual_csv.py) -- swapped groups from the
    # original plan since those 5 got downloaded manually by mistake, and
    # swapping which group is "API" vs "manual" doesn't change anything
    # (5+5 either way), so no re-downloading was needed.
    "TSLA": ("Tesla", "Consumer Discretionary"),
    "ORCL": ("Oracle", "Technology"),
    "MSFT": ("Microsoft", "Technology"),
    "JPM": ("JPMorgan Chase", "Finance"),
    "GOOGL": ("Alphabet", "Technology"),
}


def get_or_create_stock(db: Session, symbol: str, name: str, sector: str) -> Stock:
    stock = db.query(Stock).filter(Stock.symbol == symbol).first()
    if stock is None:
        stock = Stock(symbol=symbol, name=name, sector=sector)
        db.add(stock)
        db.commit()
        db.refresh(stock)
    return stock


def fetch_daily_series(symbol: str) -> dict:
    """Calls Alpha Vantage TIME_SERIES_DAILY, returns the raw time series dict."""
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "outputsize": "compact",  # free tier: last 100 data points ("full" is now premium-only)
        "apikey": API_KEY,
    }
    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if "Error Message" in data:
        raise ValueError(f"Alpha Vantage error: {data['Error Message']}")
    if "Note" in data:
        raise RuntimeError(f"Rate limited: {data['Note']}")
    if "Information" in data:
        raise RuntimeError(f"Rate limited: {data['Information']}")

    series = data.get("Time Series (Daily)")
    if not series:
        raise ValueError(f"No 'Time Series (Daily)' in response for {symbol}: {data}")
    return series


def fetch_and_store(symbol: str, name: str, sector: str, db: Session):
    print(f"Fetching {symbol}...")
    stock = get_or_create_stock(db, symbol, name, sector)

    series = fetch_daily_series(symbol)

    existing_dates = {
        row.date for row in
        db.query(HistoricalPrice.date).filter(HistoricalPrice.stock_id == stock.id).all()
    }

    new_rows = []
    for date_str, values in series.items():
        row_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        if row_date in existing_dates:
            continue
        new_rows.append(HistoricalPrice(
            stock_id=stock.id,
            date=row_date,
            open=float(values["1. open"]),
            high=float(values["2. high"]),
            low=float(values["3. low"]),
            close=float(values["4. close"]),
            volume=int(values["5. volume"]),
        ))

    if new_rows:
        latest_date_str = max(series.keys())
        last_close = float(series[latest_date_str]["4. close"])

        db.bulk_save_objects(new_rows)
        stock.last_price = last_close
        db.commit()
        print(f"  Inserted {len(new_rows)} new rows. Last close: {last_close:.2f}")
    else:
        print("  No new rows to insert (already up to date).")


def main():
    if not API_KEY:
        raise SystemExit(
            "ALPHA_VANTAGE_API_KEY not set. Add it to your .env file. "
            "Get a free key at https://www.alphavantage.co/support/#api-key"
        )

    db = SessionLocal()
    try:
        for i, (symbol, (name, sector)) in enumerate(STOCK_SYMBOLS.items()):
            try:
                fetch_and_store(symbol, name, sector, db)
            except Exception as e:
                print(f"  Error fetching {symbol}: {e}")

            # Free tier: 5 requests/minute -> wait ~13s between calls to be safe
            if i < len(STOCK_SYMBOLS) - 1:
                time.sleep(13)
    finally:
        db.close()
    print("Done.")


if __name__ == "__main__":
    main()