"""
Trains a next-day price DIRECTION classifier (up/down) for each stock,
complementary to the regression model in train.py.

Why direction instead of exact price:
Predicting the exact next-day close is hard to beat a naive baseline on for
liquid stocks (see train.py's findings) -- "no change" is a deceptively
strong guess when day-to-day price movement is close to a random walk.
Direction is a different, often more tractable question: even a modest
edge over chance is a genuinely interesting result, and there's no
equivalent "trivially strong" baseline the way there is for exact price.

Baseline used here: the MAJORITY CLASS in the training data (e.g. if a
stock went up on 54% of training days, "always predict up" gets 54%
accuracy) -- NOT a naive 50/50 coin flip. Most stocks drift upward more
often than not over multi-year windows, so a 50/50 baseline would be too
easy to beat and not a fair test.

Usage: python -m app.ml.train_direction
"""
import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from app.database import SessionLocal
from app.models import Stock, HistoricalPrice
from app.ml.features import build_features

MODELS_DIR = os.path.join(os.path.dirname(__file__), "saved_models_direction")
TEST_SIZE_FRACTION = 0.2
MIN_ROWS_REQUIRED = 30
MAX_CV_SPLITS = 4
MIN_ROWS_PER_FOLD = 12

CANDIDATE_MODELS = {
    # Scaled since raw features mix vastly different magnitudes (e.g. price
    # levels in the hundreds/thousands vs. returns as small decimals) --
    # logistic regression's solver struggles to converge without this.
    "logistic_regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=1.0)),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=200, max_depth=4, min_samples_leaf=3, random_state=42,
    ),
    "gradient_boosting": lambda: GradientBoostingClassifier(
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


def select_best_classifier(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[str, dict]:
    """Picks the best classifier via time-series CV on training data only,
    scored by accuracy. Mirrors train.py's leak-free CV approach."""
    n_rows = len(X_train)
    n_splits = min(MAX_CV_SPLITS, max(2, n_rows // MIN_ROWS_PER_FOLD))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    cv_scores = {name: [] for name in CANDIDATE_MODELS}
    for fold_train_idx, fold_val_idx in tscv.split(X_train):
        X_fold_train, X_fold_val = X_train.iloc[fold_train_idx], X_train.iloc[fold_val_idx]
        y_fold_train, y_fold_val = y_train.iloc[fold_train_idx], y_train.iloc[fold_val_idx]

        # Skip a fold if the training portion has only one class -- can't fit a classifier
        if y_fold_train.nunique() < 2:
            continue

        for name, make_model in CANDIDATE_MODELS.items():
            model = make_model()
            model.fit(X_fold_train, y_fold_train)
            preds = model.predict(X_fold_val)
            cv_scores[name].append(accuracy_score(y_fold_val, preds))

    avg_scores = {name: float(np.mean(scores)) if scores else 0.0 for name, scores in cv_scores.items()}
    best_name = max(avg_scores, key=avg_scores.get)  # higher accuracy is better
    return best_name, avg_scores


def train_one_stock(symbol: str, raw_df: pd.DataFrame) -> dict | None:
    features_df, feature_cols = build_features(raw_df)

    if len(features_df) < MIN_ROWS_REQUIRED:
        print(f"  Skipping {symbol}: only {len(features_df)} usable rows "
              f"(need at least {MIN_ROWS_REQUIRED}).")
        return None

    # Binary target: did the price go UP the next day? (1 = up, 0 = down/flat)
    features_df = features_df.copy()
    features_df["direction"] = (features_df["target_return"] > 0).astype(int)

    split_idx = int(len(features_df) * (1 - TEST_SIZE_FRACTION))
    train_df = features_df.iloc[:split_idx]
    test_df = features_df.iloc[split_idx:]

    X_train, y_train = train_df[feature_cols], train_df["direction"]
    X_test, y_test = test_df[feature_cols], test_df["direction"]

    if y_train.nunique() < 2:
        print(f"  Skipping {symbol}: training data has only one class (all up or all down).")
        return None

    # --- Baseline: majority class in the TRAINING data (not a 50/50 guess) ---
    majority_class = int(y_train.mode()[0])
    majority_class_freq = float((y_train == majority_class).mean())
    baseline_preds = np.full(len(y_test), majority_class)
    baseline_accuracy = accuracy_score(y_test, baseline_preds)

    # --- Pick best classifier via CV on training data only ---
    best_name, cv_scores = select_best_classifier(X_train, y_train)

    model = CANDIDATE_MODELS[best_name]()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    accuracy = accuracy_score(y_test, preds)
    # zero_division=0 avoids a warning/crash if the model never predicts one of the classes
    precision = precision_score(y_test, preds, zero_division=0)
    recall = recall_score(y_test, preds, zero_division=0)

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f"{symbol}.joblib")
    joblib.dump({
        "model": model,
        "feature_cols": feature_cols,
        "model_name": best_name,
        "cv_scores": cv_scores,
        "majority_class": majority_class,
        "trained_at": datetime.utcnow().isoformat(),
        "last_data_date": str(features_df["date"].max()),
    }, model_path)

    cv_summary = ", ".join(f"{name}={score:.3f}" for name, score in sorted(cv_scores.items(), key=lambda kv: -kv[1]))
    print(f"  {symbol}: train={len(train_df)} rows, test={len(test_df)} rows | best (by CV): {best_name}")
    print(f"    CV accuracy by model: {cv_summary}")
    print(f"    Model:    accuracy={accuracy:.3f}  precision={precision:.3f}  recall={recall:.3f}")
    print(f"    Baseline (majority class, train freq={majority_class_freq:.3f}): accuracy={baseline_accuracy:.3f}")
    beats_baseline = accuracy > baseline_accuracy
    print(f"    -> Model {'BEATS' if beats_baseline else 'does NOT beat'} the majority-class baseline")

    return {
        "symbol": symbol, "model_name": best_name,
        "accuracy": accuracy, "precision": precision, "recall": recall,
        "baseline_accuracy": baseline_accuracy, "beats_baseline": beats_baseline,
        "train_rows": len(train_df), "test_rows": len(test_df),
    }


def main():
    db = SessionLocal()
    results = []
    try:
        stocks = db.query(Stock).order_by(Stock.symbol).all()
        print(f"Training direction classifiers for {len(stocks)} stocks...\n")

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
        print(f"\nAverage accuracy (model):    {summary_df['accuracy'].mean():.3f}")
        print(f"Average accuracy (baseline): {summary_df['baseline_accuracy'].mean():.3f}")
        n_beats = summary_df["beats_baseline"].sum()
        print(f"Model beats majority-class baseline on {n_beats}/{len(summary_df)} stocks")
    else:
        print("No models were trained -- check you have enough price history per stock.")


if __name__ == "__main__":
    main()