"""
Walk-forward backtesting utilities for regression and direction models.

Usage (from repo root):
  python -m app.ml.walk_forward regression    # regression walk-forward
  python -m app.ml.walk_forward direction     # direction (classification) walk-forward

This module runs multiple sequential train/test windows (walk‑forward),
aggregates metrics across windows per stock, and prints a concise summary.
It reuses the existing feature builder and model-selection helpers defined in
train.py and train_direction.py to keep behavior consistent with training.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import argparse
from typing import List

from app.database import SessionLocal
from app.models import Stock, HistoricalPrice
from app.ml.features import build_features

# Import helpers from existing training modules to avoid duplicating logic
from app.ml import train as reg_train
from app.ml import train_direction as dir_train


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


def walk_forward_regression_for_stock(symbol: str, raw_df: pd.DataFrame,
                                      train_window: int | None = None,
                                      test_window: int | None = None,
                                      step: int | None = None) -> dict | None:
    features_df, feature_cols = build_features(raw_df)
    n = len(features_df)
    if n < reg_train.MIN_ROWS_REQUIRED:
        print(f"  Skipping {symbol}: only {n} usable rows (need at least {reg_train.MIN_ROWS_REQUIRED}).")
        return None

    # sensible defaults based on available history
    if test_window is None:
        test_window = max(1, int(n * reg_train.TEST_SIZE_FRACTION))
    if train_window is None:
        train_window = max(reg_train.MIN_ROWS_REQUIRED, n - test_window - test_window)
    if step is None:
        step = test_window

    if train_window + test_window > n:
        print(f"  Skipping {symbol}: train_window+test_window ({train_window}+{test_window}) > available rows ({n}).")
        return None

    metrics = []
    baseline_metrics = []

    # Slide the window forward in steps
    for start in range(0, n - train_window - test_window + 1, step):
        train_df = features_df.iloc[start:start + train_window]
        test_df = features_df.iloc[start + train_window:start + train_window + test_window]

        # Skip tiny or unusable splits
        if len(train_df) < reg_train.MIN_ROWS_REQUIRED or len(test_df) < 1:
            continue

        X_train, y_train = train_df[feature_cols], train_df["target_return"]
        X_test = test_df[feature_cols]
        y_test_price = test_df["target"]

        # Baseline: tomorrow's close = today's close
        naive_preds = test_df["close"].values
        naive_mae = float(reg_train.mean_absolute_error(y_test_price, naive_preds)) if hasattr(reg_train, 'mean_absolute_error') else float(np.mean(np.abs(y_test_price.values - naive_preds)))
        naive_rmse = float(np.sqrt(reg_train.mean_squared_error(y_test_price, naive_preds))) if hasattr(reg_train, 'mean_squared_error') else float(np.sqrt(np.mean((y_test_price.values - naive_preds) ** 2)))
        naive_mape = float(np.mean(np.abs((y_test_price.values - naive_preds) / y_test_price.values)) * 100)

        # select best model + alpha via CV on current training window
        try:
            best_name, best_alpha, _ = reg_train.select_best_model_and_alpha(train_df, feature_cols)
        except Exception as e:
            print(f"    Skipping fold start={start}: model selection failed: {e}")
            continue

        model = reg_train.CANDIDATE_MODELS[best_name]()
        model.fit(X_train, y_train)
        predicted_returns = model.predict(X_test)
        shrunk_returns = best_alpha * predicted_returns
        predicted_prices = test_df["close"].values * (1 + shrunk_returns)

        mae = float(np.mean(np.abs(y_test_price.values - predicted_prices)))
        rmse = float(np.sqrt(np.mean((y_test_price.values - predicted_prices) ** 2)))
        mape = float(np.mean(np.abs((y_test_price.values - predicted_prices) / y_test_price.values)) * 100)

        metrics.append({"mae": mae, "rmse": rmse, "mape": mape})
        baseline_metrics.append({"mae": naive_mae, "rmse": naive_rmse, "mape": naive_mape})

    if not metrics:
        print(f"  No valid walk-forward windows for {symbol} (try smaller train/test windows).")
        return None

    metrics_df = pd.DataFrame(metrics)
    baseline_df = pd.DataFrame(baseline_metrics)

    summary = {
        "symbol": symbol,
        "n_windows": len(metrics_df),
        "mae_mean": float(metrics_df["mae"].mean()),
        "mae_std": float(metrics_df["mae"].std()),
        "rmse_mean": float(metrics_df["rmse"].mean()),
        "rmse_std": float(metrics_df["rmse"].std()),
        "mape_mean": float(metrics_df["mape"].mean()),
        "mape_std": float(metrics_df["mape"].std()),
        "baseline_mae_mean": float(baseline_df["mae"].mean()),
        "baseline_mape_mean": float(baseline_df["mape"].mean()),
        "beats_baseline_fraction": float((metrics_df["mae"] < baseline_df["mae"]).mean()),
    }
    return summary


def walk_forward_regression(args: argparse.Namespace):
    db = SessionLocal()
    results: List[dict] = []
    try:
        stocks = db.query(Stock).order_by(Stock.symbol).all()
        print(f"Running regression walk-forward backtest for {len(stocks)} stocks...\n")
        for stock in stocks:
            print(f"Processing {stock.symbol}...")
            raw_df = load_price_history(db, stock.id)
            if raw_df.empty:
                print(f"  Skipping {stock.symbol}: no price history in DB.")
                continue
            res = walk_forward_regression_for_stock(stock.symbol, raw_df,
                                                   train_window=args.train_window,
                                                   test_window=args.test_window,
                                                   step=args.step)
            if res:
                results.append(res)
    finally:
        db.close()

    if not results:
        print("No stocks produced walk-forward results.")
        return

    df = pd.DataFrame(results)
    print("\n--- Walk‑forward regression summary per stock ---")
    print(df.to_string(index=False))
    print(f"\nAverage MAE (model): {df['mae_mean'].mean():.3f} (+/- {df['mae_std'].mean():.3f})")
    print(f"Average MAPE (model): {df['mape_mean'].mean():.3f} (+/- {df['mape_std'].mean():.3f})")
    print(f"Average baseline MAPE: {df['baseline_mape_mean'].mean():.3f}")
    print(f"Model beats baseline on average {df['beats_baseline_fraction'].mean():.3f} of windows per stock")


# -------------------- Direction (classification) --------------------

def walk_forward_direction_for_stock(symbol: str, raw_df: pd.DataFrame,
                                     train_window: int | None = None,
                                     test_window: int | None = None,
                                     step: int | None = None) -> dict | None:
    features_df, feature_cols = build_features(raw_df)
    n = len(features_df)
    if n < dir_train.MIN_ROWS_REQUIRED:
        print(f"  Skipping {symbol}: only {n} usable rows (need at least {dir_train.MIN_ROWS_REQUIRED}).")
        return None

    if test_window is None:
        test_window = max(1, int(n * dir_train.TEST_SIZE_FRACTION))
    if train_window is None:
        train_window = max(dir_train.MIN_ROWS_REQUIRED, n - test_window - test_window)
    if step is None:
        step = test_window

    if train_window + test_window > n:
        print(f"  Skipping {symbol}: train_window+test_window ({train_window}+{test_window}) > available rows ({n}).")
        return None

    # prepare direction column
    df = features_df.copy()
    df["direction"] = (df["target_return"] > 0).astype(int)

    metrics = []
    baseline_metrics = []

    for start in range(0, n - train_window - test_window + 1, step):
        train_df = df.iloc[start:start + train_window]
        test_df = df.iloc[start + train_window:start + train_window + test_window]

        if len(train_df) < dir_train.MIN_ROWS_REQUIRED or len(test_df) < 1:
            continue

        X_train, y_train = train_df[feature_cols], train_df["direction"]
        X_test, y_test = test_df[feature_cols], test_df["direction"]

        if y_train.nunique() < 2:
            # can't train classifier on single-class training window
            continue

        # baseline: majority class in training portion
        majority = int(y_train.mode()[0])
        baseline_preds = np.full(len(y_test), majority)
        baseline_acc = float((baseline_preds == y_test.values).mean())

        try:
            best_name, _ = dir_train.select_best_classifier(X_train, y_train)
        except Exception as e:
            print(f"    Skipping fold start={start}: classifier selection failed: {e}")
            continue

        model = dir_train.CANDIDATE_MODELS[best_name]()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = float((preds == y_test.values).mean())
        # for simplicity omit precision/recall aggregation here; can be added if needed

        metrics.append({"accuracy": acc})
        baseline_metrics.append({"accuracy": baseline_acc})

    if not metrics:
        print(f"  No valid walk-forward windows for {symbol} (try smaller train/test windows).")
        return None

    mdf = pd.DataFrame(metrics)
    bdf = pd.DataFrame(baseline_metrics)

    summary = {
        "symbol": symbol,
        "n_windows": len(mdf),
        "accuracy_mean": float(mdf["accuracy"].mean()),
        "accuracy_std": float(mdf["accuracy"].std()),
        "baseline_accuracy_mean": float(bdf["accuracy"].mean()),
        "beats_baseline_fraction": float((mdf["accuracy"] > bdf["accuracy"]).mean()),
    }
    return summary


def walk_forward_direction(args: argparse.Namespace):
    db = SessionLocal()
    results: List[dict] = []
    try:
        stocks = db.query(Stock).order_by(Stock.symbol).all()
        print(f"Running direction walk-forward backtest for {len(stocks)} stocks...\n")
        for stock in stocks:
            print(f"Processing {stock.symbol}...")
            raw_df = load_price_history(db, stock.id)
            if raw_df.empty:
                print(f"  Skipping {stock.symbol}: no price history in DB.")
                continue
            res = walk_forward_direction_for_stock(stock.symbol, raw_df,
                                                  train_window=args.train_window,
                                                  test_window=args.test_window,
                                                  step=args.step)
            if res:
                results.append(res)
    finally:
        db.close()

    if not results:
        print("No stocks produced walk-forward results.")
        return

    df = pd.DataFrame(results)
    print("\n--- Walk‑forward direction summary per stock ---")
    print(df.to_string(index=False))
    print(f"\nAverage accuracy (model): {df['accuracy_mean'].mean():.3f} (+/- {df['accuracy_std'].mean():.3f})")
    print(f"Average baseline accuracy: {df['baseline_accuracy_mean'].mean():.3f}")
    print(f"Model beats baseline on average {df['beats_baseline_fraction'].mean():.3f} of windows per stock")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward backtesting for regression and direction models")
    sub = parser.add_subparsers(dest="cmd", required=True)

    reg = sub.add_parser("regression", help="Run walk-forward for regression models")
    reg.add_argument("--train-window", type=int, default=None, help="Training window size in rows (default: automatic)")
    reg.add_argument("--test-window", type=int, default=None, help="Test window size in rows (default: automatic)")
    reg.add_argument("--step", type=int, default=None, help="Step size to advance the window (default: test-window)")

    dirp = sub.add_parser("direction", help="Run walk-forward for direction classifiers")
    dirp.add_argument("--train-window", type=int, default=None, help="Training window size in rows (default: automatic)")
    dirp.add_argument("--test-window", type=int, default=None, help="Test window size in rows (default: automatic)")
    dirp.add_argument("--step", type=int, default=None, help="Step size to advance the window (default: test-window)")

    return parser.parse_args()


def main():
    args = parse_args()
    if args.cmd == "regression":
        walk_forward_regression(args)
    elif args.cmd == "direction":
        walk_forward_direction(args)


if __name__ == "__main__":
    main()
