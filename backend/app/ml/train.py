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

# Shrinkage factors to blend the model's predicted return toward the naive
# baseline (which is equivalent to a predicted return of exactly 0). alpha=1.0
# means "trust the model fully"; alpha=0.0 means "ignore the model, use the
# baseline"; values in between shrink the model's prediction toward zero.
# This is a standard variance-reduction technique for noisy regression
# targets -- if the model's return predictions are directionally OK but
# noisy, shrinking them toward zero often reduces error even though it never
# "sees" a genuinely different pattern.
SHRINKAGE_ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

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


def select_best_model_and_alpha(train_df: pd.DataFrame, feature_cols: list) -> tuple[str, float, dict]:
    """
    Picks the best (model, shrinkage_alpha) combination using time-series
    cross-validation WITHIN the training set only -- the held-out test set
    is never touched here. Evaluation is done in price-space (not
    return-space) since that's what we actually care about and report.

    Returns (best_model_name, best_alpha, {(model_name, alpha): avg_cv_mae})
    for logging.
    """
    n_rows = len(train_df)
    n_splits = min(MAX_CV_SPLITS, max(2, n_rows // MIN_ROWS_PER_FOLD))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    cv_scores = {(name, a): [] for name in CANDIDATE_MODELS for a in SHRINKAGE_ALPHAS}

    for fold_train_idx, fold_val_idx in tscv.split(train_df):
        fold_train = train_df.iloc[fold_train_idx]
        fold_val = train_df.iloc[fold_val_idx]
        X_fold_train, y_fold_train = fold_train[feature_cols], fold_train["target_return"]
        X_fold_val = fold_val[feature_cols]
        y_fold_val_price = fold_val["target"]
        val_close = fold_val["close"].values

        for name, make_model in CANDIDATE_MODELS.items():
            model = make_model()
            model.fit(X_fold_train, y_fold_train)
            predicted_returns = model.predict(X_fold_val)

            for alpha in SHRINKAGE_ALPHAS:
                shrunk_returns = alpha * predicted_returns
                predicted_prices = val_close * (1 + shrunk_returns)
                mae = mean_absolute_error(y_fold_val_price, predicted_prices)
                cv_scores[(name, alpha)].append(mae)

    avg_scores = {key: float(np.mean(scores)) for key, scores in cv_scores.items()}
    best_name, best_alpha = min(avg_scores, key=avg_scores.get)
    return best_name, best_alpha, avg_scores


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

    # --- Pick the best (model, shrinkage_alpha) combo via CV on training data only ---
    best_name, best_alpha, cv_scores = select_best_model_and_alpha(train_df, feature_cols)

    # --- Fit the chosen model on the FULL training set, evaluate ONCE on test ---
    model = CANDIDATE_MODELS[best_name]()
    model.fit(X_train, y_train)

    predicted_returns = model.predict(X_test)
    shrunk_returns = best_alpha * predicted_returns  # blend toward baseline (alpha=0 = pure baseline)
    predicted_prices = test_df["close"].values * (1 + shrunk_returns)

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
        "shrinkage_alpha": best_alpha,
        "cv_scores": {f"{name}_alpha{a}": score for (name, a), score in cv_scores.items()},
        "trained_at": datetime.utcnow().isoformat(),
        "last_data_date": str(features_df["date"].max()),
    }, model_path)

    best_combo_score = cv_scores[(best_name, best_alpha)]
    print(f"  {symbol}: train={len(train_df)} rows, test={len(test_df)} rows | "
          f"best (by CV): {best_name}, shrinkage_alpha={best_alpha} (cv_mae={best_combo_score:.2f})")
    print(f"    Model:    MAE={mae:.2f}  RMSE={rmse:.2f}  MAPE={mape:.2f}%")
    print(f"    Baseline: MAE={naive_mae:.2f}  RMSE={naive_rmse:.2f}  MAPE={naive_mape:.2f}%")
    beats_baseline = mae < naive_mae
    print(f"    -> Model {'BEATS' if beats_baseline else 'does NOT beat'} the naive baseline")

    return {"symbol": symbol, "model_name": best_name, "shrinkage_alpha": best_alpha,
            "mae": mae, "rmse": rmse, "mape": mape,
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