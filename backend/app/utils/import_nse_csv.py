"""
One-time import for historical price data downloaded manually from NSE
India's official website (nseindia.com), for the 10 Indian stocks.

NSE's export format: DATE, SERIES, OPEN, HIGH, LOW, PREV. CLC, LTP, CLOSE,
VWAP, 52W H, 52W L, VOLUME, VALUE, NO. OF TRADES
-- we only need DATE, OPEN, HIGH, LOW, CLOSE, VOLUME; the rest are ignored.

Numbers use Indian-style comma grouping (e.g. "1,83,02,021") and dates are
in DD-Mon-YY format (e.g. "17-Jul-26") -- both handled below.

These are imported as their OWN separate stock records (e.g. "RELIANCE.NS"),
distinct from the existing "RELIANCE.BSE" records populated by the Alpha
Vantage fetcher -- NSE and BSE are different exchanges and can have
slightly different prices, so they're tracked as separate listings rather
than merged into one series.

Usage:
    Place CSVs in backend/data_import/nse/, named to match the plain
    ticker, e.g. data_import/nse/RELIANCE.csv.
    Then run: python -m app.utils.import_nse_csv
"""
import os
import pandas as pd
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models import Stock, HistoricalPrice

IMPORT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data_import", "nse"
)

# NSE download filename (without .csv) -> (new .NS symbol, display name, sector)
NSE_STOCKS = {
    "RELIANCE_data": ("RELIANCE.NS", "Reliance Industries", "Energy"),
    "TCS_data": ("TCS.NS", "Tata Consultancy Services", "IT"),
    "HDFCBANK_data": ("HDFCBANK.NS", "HDFC Bank", "Finance"),
    "INFY_data": ("INFY.NS", "Infosys", "IT"),
    "ICICIBANK_data": ("ICICIBANK.NS", "ICICI Bank", "Finance"),
    "HINDU_data": ("HINDUNILVR.NS", "Hindustan Unilever", "FMCG"),
    "SBIN_data": ("SBIN.NS", "State Bank of India", "Finance"),
    "AIRTEL_data": ("BHARTIARTL.NS", "Bharti Airtel", "Telecom"),
    "ITC_data": ("ITC.NS", "ITC Limited", "FMCG"),
    "KOTAK_data": ("KOTAKBANK.NS", "Kotak Mahindra Bank", "Finance"),
}


def load_and_clean_nse_csv(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df.columns = [c.strip().upper() for c in df.columns]

    required_source_cols = ["DATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]
    missing = [c for c in required_source_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV {filepath} is missing expected columns {missing}. "
            f"Found columns: {list(df.columns)}."
        )

    df = df[required_source_cols].copy()
    df.columns = ["date", "open", "high", "low", "close", "volume"]

    # Strip Indian-style comma grouping from numeric columns (e.g. "1,83,02,021")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(str).str.replace(",", "", regex=False)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["volume"] = df["volume"].fillna(0).astype(int)

    # NSE dates are typically DD-Mon-YYYY (e.g. "17-Jul-2026"), but some
    # export tools use a 2-digit year instead -- try both rather than
    # assuming one and failing on the other.
    try:
        df["date"] = pd.to_datetime(df["date"], format="%d-%b-%Y").dt.date
    except ValueError:
        df["date"] = pd.to_datetime(df["date"], format="%d-%b-%y").dt.date

    df = df.dropna(subset=["open", "high", "low", "close"])
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


def import_one_stock(nse_filename: str, ns_symbol: str, name: str, sector: str, db: Session):
    filepath = os.path.join(IMPORT_DIR, f"{nse_filename}.csv")
    if not os.path.exists(filepath):
        print(f"  Skipping {ns_symbol}: file not found at {filepath}")
        return

    print(f"Importing {ns_symbol} from {filepath}...")
    df = load_and_clean_nse_csv(filepath)

    stock = get_or_create_stock(db, ns_symbol, name, sector)

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
        # Only update last_price if this import actually reaches more recent
        # data than what's already stored -- don't let an older backfill
        # overwrite a more current price.
        latest_new_date = df["date"].max()
        latest_existing_date = max(existing_dates) if existing_dates else None
        if latest_existing_date is None or latest_new_date > latest_existing_date:
            stock.last_price = float(df.loc[df["date"] == latest_new_date, "close"].iloc[0])
        db.commit()
        print(f"  Inserted {len(new_rows)} new rows (backfill).")
    else:
        print("  No new rows to insert (already up to date).")


def main():
    os.makedirs(IMPORT_DIR, exist_ok=True)
    db = SessionLocal()
    try:
        for nse_filename, (ns_symbol, name, sector) in NSE_STOCKS.items():
            import_one_stock(nse_filename, ns_symbol, name, sector, db)
    finally:
        db.close()
    print("Done.")


if __name__ == "__main__":
    main()