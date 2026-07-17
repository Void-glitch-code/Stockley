"""
Trains a next-day closing price predictor for each stock in the database.

Uses a time-based train/test split (never randomly shuffled -- shuffling
time series data leaks future information into training) and evaluates
with MAE and RMSE. Saves one trained model per stock to app/ml/saved_models/.

Usage: python -m app.ml.train
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from app.database import SessionLocal
from app.models import Stock, HistoricalPrice
from app.ml.features import build_features

MODELS_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
TEST_SIZE_FRACTION = 0.2  # last 20% of days held out for testing
MIN_ROWS_REQUIRED = 30    # skip stocks that don't have enough data yet


def load_price_history(db, stock_id: int) -> pd.DataFrame:
    rows = (
        db.query(HistoricalPrice)
        .filter(HistoricalPrice.stock_id == stock_id)
        .order_by(HistoricalPrice.date)
        .all()
    )
    return pd.DataFrame([{
        "date": r.date, "open": r.open, "high": r.high,
        "low": r.low, "close": r.close, "volume": r.volume,
    } for r in rows])


def train_one_stock(symbol: str, raw_df: pd.DataFrame) -> dict | None:
    features_df, feature_cols = build_features(raw_df)

    if len(features_df) < MIN_ROWS_REQUIRED:
        print(f"  Skipping {symbol}: only {len(features_df)} usable rows "
              f"(need at least {MIN_ROWS_REQUIRED}).")
        return None

    # Time-based split: train on the earlier portion, test on the most recent
    # portion. This mimics the real situation of predicting the future from
    # the past, and avoids the data leakage a random shuffle would cause.
    split_idx = int(len(features_df) * (1 - TEST_SIZE_FRACTION))
    train_df = features_df.iloc[:split_idx]
    test_df = features_df.iloc[split_idx:]

    X_train, y_train = train_df[feature_cols], train_df["target"]
    X_test, y_test = test_df[feature_cols], test_df["target"]

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=2,
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mape = float(np.mean(np.abs((y_test.values - preds) / y_test.values)) * 100)

    # --- Naive baseline: "tomorrow's close = today's close" ---
    # If our model can't beat this trivial guess, it isn't actually learning
    # anything useful -- stock prices are highly autocorrelated day-to-day,
    # so this is a deceptively strong baseline to have to beat.
    naive_preds = test_df["close"].values
    naive_mae = mean_absolute_error(y_test, naive_preds)
    naive_rmse = np.sqrt(mean_squared_error(y_test, naive_preds))
    naive_mape = float(np.mean(np.abs((y_test.values - naive_preds) / y_test.values)) * 100)

    # Save the trained model + the feature column order (needed at prediction time)
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f"{symbol}.joblib")
    joblib.dump({"model": model, "feature_cols": feature_cols}, model_path)

    print(f"  {symbol}: train={len(train_df)} rows, test={len(test_df)} rows")
    print(f"    Model:    MAE={mae:.2f}  RMSE={rmse:.2f}  MAPE={mape:.2f}%")
    print(f"    Baseline: MAE={naive_mae:.2f}  RMSE={naive_rmse:.2f}  MAPE={naive_mape:.2f}%")
    beats_baseline = mae < naive_mae
    print(f"    -> Model {'BEATS' if beats_baseline else 'does NOT beat'} the naive baseline")

    return {"symbol": symbol, "mae": mae, "rmse": rmse, "mape": mape,
            "naive_mae": naive_mae, "naive_mape": naive_mape,
            "beats_baseline": beats_baseline,
            "train_rows": len(train_df), "test_rows": len(test_df)}


def main():
    db = SessionLocal()
    results = []
    try:
        stocks = db.query(Stock).order_by(Stock.symbol).all()
        print(f"Training models for {len(stocks)} stocks...\n")

        for stock in stocks:
            print(f"Processing {stock.symbol}...")
            raw_df = load_price_history(db, stock.id)
            if raw_df.empty:
                print(f"  Skipping {stock.symbol}: no price history in DB.")
                continue
            result = train_one_stock(stock.symbol, raw_df)
            if result:
                results.append(result)
    finally:
        db.close()

    print("\n--- Summary ---")
    if results:
        summary_df = pd.DataFrame(results)
        print(summary_df.to_string(index=False))
        print(f"\nAverage MAPE (model):    {summary_df['mape'].mean():.2f}%")
        print(f"Average MAPE (baseline): {summary_df['naive_mape'].mean():.2f}%")
        n_beats = summary_df["beats_baseline"].sum()
        print(f"Model beats naive baseline on {n_beats}/{len(summary_df)} stocks")
    else:
        print("No models were trained -- check you have enough price history per stock.")


if __name__ == "__main__":
    main()