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
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from app.database import SessionLocal
from app.models import Stock, HistoricalPrice
from app.ml.features import build_features

MODELS_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
TEST_SIZE_FRACTION = 0.2  # last 20% of days held out for testing
MIN_ROWS_REQUIRED = 30    # skip stocks that don't have enough data yet
MAX_CV_SPLITS = 4         # cap on time-series CV folds for model selection
MIN_ROWS_PER_FOLD = 12    # need at least this many rows per CV fold to be meaningful

# Candidate models to try per stock. With very little training data (as few
# as ~60-80 rows here), a simpler, more regularized model like Ridge often
# generalizes better than a complex Random Forest, which can overfit noise
# even when trained on returns instead of raw prices. We try a few and keep
# whichever actually performs best -- via cross-validation on the TRAINING
# data only, never the test set (see select_best_model below).
CANDIDATE_MODELS = {
    "ridge": lambda: Ridge(alpha=5.0),
    "random_forest": lambda: RandomForestRegressor(
        n_estimators=200, max_depth=4, min_samples_leaf=3, random_state=42,
    ),
    "gradient_boosting": lambda: GradientBoostingRegressor(
        n_estimators=100, max_depth=2, learning_rate=0.05, random_state=42,
    ),
}


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


def select_best_model(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[str, dict]:
    """
    Picks the best candidate model using time-series cross-validation
    WITHIN the training set only. The held-out test set is never touched
    here -- this is the fix for the previous version, which picked
    "best model" by lowest MAE *on the test set*, which is a form of
    leakage: it silently tunes model choice to the exact data you later
    report performance on, inflating "beats baseline" results with noise
    (especially damaging here since test sets are only 16-45 rows).

    Returns (best_model_name, {model_name: avg_cv_mae}) for logging.
    """
    n_rows = len(X_train)
    n_splits = min(MAX_CV_SPLITS, max(2, n_rows // MIN_ROWS_PER_FOLD))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    cv_scores = {name: [] for name in CANDIDATE_MODELS}
    for fold_train_idx, fold_val_idx in tscv.split(X_train):
        X_fold_train, X_fold_val = X_train.iloc[fold_train_idx], X_train.iloc[fold_val_idx]
        y_fold_train, y_fold_val = y_train.iloc[fold_train_idx], y_train.iloc[fold_val_idx]

        for name, make_model in CANDIDATE_MODELS.items():
            model = make_model()
            model.fit(X_fold_train, y_fold_train)
            preds = model.predict(X_fold_val)
            cv_scores[name].append(mean_absolute_error(y_fold_val, preds))

    avg_scores = {name: float(np.mean(scores)) for name, scores in cv_scores.items()}
    best_name = min(avg_scores, key=avg_scores.get)
    return best_name, avg_scores


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

    X_train, y_train = train_df[feature_cols], train_df["target_return"]
    X_test = test_df[feature_cols]
    y_test_price = test_df["target"]  # actual next-day close, for evaluation in ₹ terms

    # --- Naive baseline: "tomorrow's close = today's close" ---
    naive_preds = test_df["close"].values
    naive_mae = mean_absolute_error(y_test_price, naive_preds)
    naive_rmse = np.sqrt(mean_squared_error(y_test_price, naive_preds))
    naive_mape = float(np.mean(np.abs((y_test_price.values - naive_preds) / y_test_price.values)) * 100)

    # --- Pick the best model via CV on the training set only ---
    best_name, cv_scores = select_best_model(X_train, y_train)

    # --- Fit the chosen model on the FULL training set, evaluate ONCE on test ---
    model = CANDIDATE_MODELS[best_name]()
    model.fit(X_train, y_train)

    predicted_returns = model.predict(X_test)
    predicted_prices = test_df["close"].values * (1 + predicted_returns)

    mae = mean_absolute_error(y_test_price, predicted_prices)
    rmse = np.sqrt(mean_squared_error(y_test_price, predicted_prices))
    mape = float(np.mean(np.abs((y_test_price.values - predicted_prices) / y_test_price.values)) * 100)

    # Save the trained model + the feature column order (needed at prediction time)
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f"{symbol}.joblib")
    joblib.dump({
        "model": model,
        "feature_cols": feature_cols,
        "model_name": best_name,
        "cv_scores": cv_scores,
        "trained_at": datetime.utcnow().isoformat(),
        "last_data_date": str(features_df["date"].max()),
    }, model_path)

    cv_summary = ", ".join(f"{name}={score:.4f}" for name, score in sorted(cv_scores.items(), key=lambda kv: kv[1]))
    print(f"  {symbol}: train={len(train_df)} rows, test={len(test_df)} rows | "
          f"best model (by CV): {best_name}")
    print(f"    CV return-MAE by model: {cv_summary}")
    print(f"    Model:    MAE={mae:.2f}  RMSE={rmse:.2f}  MAPE={mape:.2f}%")
    print(f"    Baseline: MAE={naive_mae:.2f}  RMSE={naive_rmse:.2f}  MAPE={naive_mape:.2f}%")
    beats_baseline = mae < naive_mae
    print(f"    -> Model {'BEATS' if beats_baseline else 'does NOT beat'} the naive baseline")

    return {"symbol": symbol, "model_name": best_name, "mae": mae, "rmse": rmse, "mape": mape,
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