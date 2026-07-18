"""
One-time import for historical price data downloaded manually (via Google
Sheets GOOGLEFINANCE or Nasdaq.com's Historical Data export), for the 5
global stocks NOT covered by the Alpha Vantage fetcher: Apple, NVIDIA,
Meta, Intel, AMD.

These are handled separately (rather than adding them to data_fetcher.py)
to stay within Alpha Vantage's 25-calls/day free tier limit -- see
STOCK_SYMBOLS in data_fetcher.py for the 15 stocks that DO use the API.

Usage:
    Place CSVs in backend/data_import/global/, named exactly like the keys
    below (e.g. data_import/global/TSLA.csv), then run:
        python -m app.utils.import_manual_csv

Expected CSV columns (case-insensitive, order doesn't matter):
    Date, Open, High, Low, Close, Volume
This matches both Nasdaq.com's export and a GOOGLEFINANCE() sheet exported
to CSV. If your columns are named differently (e.g. "Close/Last" from
Nasdaq, or extra columns like Adj Close), the column-mapping section below
handles the common variants -- add more if your source differs.
"""
import os
import pandas as pd
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models import Stock, HistoricalPrice

IMPORT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data_import", "global"
)

# symbol -> (filename without extension, display name, sector)
MANUAL_STOCKS = {
    "AAPL": ("AAPL", "Apple", "Technology"),
    "NVDA": ("NVDA", "NVIDIA", "Technology"),
    "META": ("META", "Meta Platforms", "Technology"),
    "INTC": ("INTC", "Intel", "Technology"),
    "AMD": ("AMD", "Advanced Micro Devices", "Technology"),
}

# Maps common column name variants (lowercased) to our standard names.
# Nasdaq.com uses "Close/Last" instead of "Close" and often includes a
# leading "$" in price columns -- handled below during cleaning.
COLUMN_ALIASES = {
    "date": "date",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "close/last": "close",
    "adj close": "close",  # only used if plain 'close' isn't present
    "volume": "volume",
}


def load_and_clean_csv(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df.columns = [c.strip().lower() for c in df.columns]

    rename_map = {}
    for col in df.columns:
        if col in COLUMN_ALIASES:
            rename_map[col] = COLUMN_ALIASES[col]
    df = df.rename(columns=rename_map)

    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV {filepath} is missing columns {missing} after cleaning. "
            f"Found columns: {list(df.columns)}. Update COLUMN_ALIASES if your "
            f"source uses different names."
        )
    df = df[required]

    # Strip $ signs and commas from price columns (common in Nasdaq.com exports).
    # Applied unconditionally (not gated on dtype) since pandas' string dtype
    # detection varies across versions/backends and isn't reliable to branch on.
    for col in ["open", "high", "low", "close"]:
        df[col] = (
            df[col].astype(str).str.replace(r"[$,]", "", regex=True)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["volume"] = df["volume"].astype(str).str.replace(",", "", regex=False)
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)

    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values("date").reset_index(drop=True)
    return df


def get_or_create_stock(db: Session, symbol: str, name: str, sector: str) -> Stock:
    stock = db.query(Stock).filter(Stock.symbol == symbol).first()
    if stock is None:
        stock = Stock(symbol=symbol, name=name, sector=sector)
        db.add(stock)
        db.commit()
        db.refresh(stock)
    return stock


def import_one_stock(symbol: str, filename: str, name: str, sector: str, db: Session):
    filepath = os.path.join(IMPORT_DIR, f"{filename}.csv")
    if not os.path.exists(filepath):
        print(f"  Skipping {symbol}: file not found at {filepath}")
        return

    print(f"Importing {symbol} from {filepath}...")
    df = load_and_clean_csv(filepath)

    stock = get_or_create_stock(db, symbol, name, sector)

    existing_dates = {
        row.date for row in
        db.query(HistoricalPrice.date).filter(HistoricalPrice.stock_id == stock.id).all()
    }

    new_rows = [
        HistoricalPrice(
            stock_id=stock.id, date=row.date, open=row.open,
            high=row.high, low=row.low, close=row.close, volume=row.volume,
        )
        for row in df.itertuples()
        if row.date not in existing_dates
    ]

    if new_rows:
        db.bulk_save_objects(new_rows)
        stock.last_price = float(df["close"].iloc[-1])
        db.commit()
        print(f"  Inserted {len(new_rows)} rows. Last close: {stock.last_price:.2f}")
    else:
        print("  No new rows to insert (already up to date).")


def main():
    os.makedirs(IMPORT_DIR, exist_ok=True)
    db = SessionLocal()
    try:
        for symbol, (filename, name, sector) in MANUAL_STOCKS.items():
            import_one_stock(symbol, filename, name, sector, db)
    finally:
        db.close()
    print("Done.")


if __name__ == "__main__":
    main()