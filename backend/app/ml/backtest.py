"""
Walk-forward backtest for both the price-regression model (train.py) and
the direction classifier (train_direction.py).

WHY: a single train/test split (what train.py and train_direction.py use
day-to-day) gives one noisy estimate per stock -- test sets are as small
as 16-45 rows, so a stock's "beats baseline" result can easily flip on
sampling luck alone. A walk-forward backtest instead uses several
sequential windows (via TimeSeriesSplit on the FULL dataset, not just the
final 80%): fold 1 trains on the earliest data and tests on the next
chunk, fold 2 trains on everything up to that point and tests on the
chunk after that, and so on. This gives multiple independent estimates
per stock, so "does this actually help" is based on evidence pooled
across folds/stocks rather than one lucky/unlucky split.

We also report the fold-to-fold spread (std) per stock, not just the
pooled average -- a stock that beats baseline with wildly different
error from fold to fold is a different, weaker finding than one that
beats it consistently, and the pooled mean alone can't tell those apart.

Reuses the exact same model-selection logic as train.py and
train_direction.py (imported directly) so results are apples-to-apples
with what's already being reported, just averaged over more windows.

Usage: python -m app.ml.backtest
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.model_selection import TimeSeriesSplit

from app.database import SessionLocal
from app.models import Stock, HistoricalPrice
from app.ml.features import build_features
from app.ml.train import (
    CANDIDATE_MODELS as REGRESSION_MODELS,
    select_best_model_and_alpha,
    MIN_ROWS_REQUIRED,
)
from app.ml.train_direction import (
    CANDIDATE_MODELS as DIRECTION_MODELS,
    select_best_classifier,
)

MAX_OUTER_FOLDS = 5
MIN_ROWS_PER_OUTER_FOLD = 25  # each outer fold's test window needs to be meaningfully sized


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


def backtest_regression(features_df: pd.DataFrame, feature_cols: list, n_splits: int) -> dict | None:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_records = []

    for train_idx, test_idx in tscv.split(features_df):
        train_df = features_df.iloc[train_idx]
        test_df = features_df.iloc[test_idx]
        if len(train_df) < MIN_ROWS_REQUIRED or len(test_df) == 0:
            continue

        X_train, y_train = train_df[feature_cols], train_df["target_return"]
        X_test = test_df[feature_cols]
        y_test_price = test_df["target"]

        naive_preds = test_df["close"].values
        naive_mae = mean_absolute_error(y_test_price, naive_preds)

        best_name, best_alpha, _ = select_best_model_and_alpha(train_df, feature_cols)
        model = REGRESSION_MODELS[best_name]()
        model.fit(X_train, y_train)
        predicted_returns = model.predict(X_test)
        predicted_prices = test_df["close"].values * (1 + best_alpha * predicted_returns)
        mae = mean_absolute_error(y_test_price, predicted_prices)

        fold_records.append({
            "test_rows": len(test_df), "mae": mae, "naive_mae": naive_mae,
            "beats_baseline": mae < naive_mae, "alpha": best_alpha,
        })

    if not fold_records:
        return None

    df = pd.DataFrame(fold_records)
    # Weight by fold size so bigger test windows count proportionally more
    pooled_mae = float(np.average(df["mae"], weights=df["test_rows"]))
    pooled_naive_mae = float(np.average(df["naive_mae"], weights=df["test_rows"]))
    return {
        "n_folds": len(df),
        "pooled_mae": pooled_mae,
        "pooled_naive_mae": pooled_naive_mae,
        "pooled_beats_baseline": pooled_mae < pooled_naive_mae,
        "folds_beating_baseline": int(df["beats_baseline"].sum()),
        # Fold-to-fold spread -- a stock that beats baseline with high
        # variance across folds is a weaker, less trustworthy result than
        # one that beats it consistently, even if the pooled mean is the same.
        "mae_std": float(df["mae"].std()) if len(df) > 1 else 0.0,
    }


def backtest_direction(features_df: pd.DataFrame, feature_cols: list, n_splits: int) -> dict | None:
    features_df = features_df.copy()
    features_df["direction"] = (features_df["target_return"] > 0).astype(int)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_records = []

    for train_idx, test_idx in tscv.split(features_df):
        train_df = features_df.iloc[train_idx]
        test_df = features_df.iloc[test_idx]
        if len(train_df) < MIN_ROWS_REQUIRED or len(test_df) == 0:
            continue

        X_train, y_train = train_df[feature_cols], train_df["direction"]
        X_test, y_test = test_df[feature_cols], test_df["direction"]
        if y_train.nunique() < 2:
            continue

        majority_class = int(y_train.mode()[0])
        baseline_preds = np.full(len(y_test), majority_class)
        baseline_accuracy = accuracy_score(y_test, baseline_preds)

        best_name, _ = select_best_classifier(X_train, y_train)
        model = DIRECTION_MODELS[best_name]()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)

        fold_records.append({
            "test_rows": len(test_df), "accuracy": accuracy,
            "baseline_accuracy": baseline_accuracy, "beats_baseline": accuracy > baseline_accuracy,
        })

    if not fold_records:
        return None

    df = pd.DataFrame(fold_records)
    pooled_accuracy = float(np.average(df["accuracy"], weights=df["test_rows"]))
    pooled_baseline_accuracy = float(np.average(df["baseline_accuracy"], weights=df["test_rows"]))
    return {
        "n_folds": len(df),
        "pooled_accuracy": pooled_accuracy,
        "pooled_baseline_accuracy": pooled_baseline_accuracy,
        "pooled_beats_baseline": pooled_accuracy > pooled_baseline_accuracy,
        "folds_beating_baseline": int(df["beats_baseline"].sum()),
        "accuracy_std": float(df["accuracy"].std()) if len(df) > 1 else 0.0,
    }


def main():
    db = SessionLocal()
    regression_results = []
    direction_results = []
    try:
        stocks = db.query(Stock).order_by(Stock.symbol).all()
        print(f"Walk-forward backtesting {len(stocks)} stocks...\n")

        for stock in stocks:
            raw_df = load_price_history(db, stock.id)
            if raw_df.empty:
                continue
            features_df, feature_cols = build_features(raw_df)
            if len(features_df) < MIN_ROWS_REQUIRED:
                print(f"{stock.symbol}: skipped (only {len(features_df)} usable rows)")
                continue

            n_splits = min(MAX_OUTER_FOLDS, max(2, len(features_df) // MIN_ROWS_PER_OUTER_FOLD))

            reg_result = backtest_regression(features_df, feature_cols, n_splits)
            dir_result = backtest_direction(features_df, feature_cols, n_splits)

            print(f"{stock.symbol}: ({n_splits} folds)")
            if reg_result:
                r = reg_result
                verdict = "BEATS" if r["pooled_beats_baseline"] else "does NOT beat"
                print(f"  Regression: pooled MAE={r['pooled_mae']:.2f} (+/- {r['mae_std']:.2f}) "
                      f"vs baseline={r['pooled_naive_mae']:.2f} "
                      f"({r['folds_beating_baseline']}/{r['n_folds']} folds beat baseline) -> {verdict}")
                regression_results.append({"symbol": stock.symbol, **r})
            else:
                print("  Regression: not enough data for any fold")

            if dir_result:
                d = dir_result
                verdict = "BEATS" if d["pooled_beats_baseline"] else "does NOT beat"
                print(f"  Direction:  pooled acc={d['pooled_accuracy']:.3f} (+/- {d['accuracy_std']:.3f}) "
                      f"vs baseline={d['pooled_baseline_accuracy']:.3f} "
                      f"({d['folds_beating_baseline']}/{d['n_folds']} folds beat baseline) -> {verdict}")
                direction_results.append({"symbol": stock.symbol, **d})
            else:
                print("  Direction:  not enough data for any fold")
            print()
    finally:
        db.close()

    print("--- Overall Summary (pooled across all stocks, weighted by fold size) ---\n")

    if regression_results:
        reg_df = pd.DataFrame(regression_results)
        overall_mae = reg_df["pooled_mae"].mean()
        overall_naive_mae = reg_df["pooled_naive_mae"].mean()
        avg_fold_std = reg_df["mae_std"].mean()
        n_stocks_beating = int(reg_df["pooled_beats_baseline"].sum())
        print(f"Regression: {n_stocks_beating}/{len(reg_df)} stocks beat baseline on pooled walk-forward MAE")
        print(f"  Avg pooled MAE across stocks: model={overall_mae:.2f}  baseline={overall_naive_mae:.2f}")
        print(f"  Avg fold-to-fold MAE std across stocks: {avg_fold_std:.2f} "
              f"(higher = less consistent across folds, treat that stock's result with more caution)")

    if direction_results:
        dir_df = pd.DataFrame(direction_results)
        overall_acc = dir_df["pooled_accuracy"].mean()
        overall_baseline_acc = dir_df["pooled_baseline_accuracy"].mean()
        avg_fold_std = dir_df["accuracy_std"].mean()
        n_stocks_beating = int(dir_df["pooled_beats_baseline"].sum())
        print(f"\nDirection: {n_stocks_beating}/{len(dir_df)} stocks beat baseline on pooled walk-forward accuracy")
        print(f"  Avg pooled accuracy across stocks: model={overall_acc:.3f}  baseline={overall_baseline_acc:.3f}")
        print(f"  Avg fold-to-fold accuracy std across stocks: {avg_fold_std:.3f}")


if __name__ == "__main__":
    main()