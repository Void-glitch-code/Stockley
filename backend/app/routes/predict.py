import os
import joblib
import pandas as pd
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Stock, HistoricalPrice
from app.ml.features import build_latest_features

router = APIRouter(prefix="/api/predict", tags=["predict"])

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ml", "saved_models")

# If the model's training data is older than this many days vs. today's date,
# flag the prediction as potentially stale (model hasn't seen recent prices).
STALENESS_THRESHOLD_DAYS = 5


@router.get("/{symbol}")
def predict_next_close(symbol: str, db: Session = Depends(get_db)):
    """
    Predicts tomorrow's closing price for a stock, using its saved model.

    Errors:
      404 - stock doesn't exist in the database
      503 - no trained model yet, not enough price history, or the saved
            model file is unreadable/incompatible (e.g. feature set changed
            since it was trained -- just retrain in that case)
    """
    symbol = symbol.upper()
    stock = db.query(Stock).filter(Stock.symbol == symbol).first()
    if stock is None:
        raise HTTPException(status_code=404, detail=f"Stock '{symbol}' not found")

    model_path = os.path.join(MODELS_DIR, f"{symbol}.joblib")
    if not os.path.exists(model_path):
        raise HTTPException(
            status_code=503,
            detail=f"No trained model available for '{symbol}' yet. Run `python -m app.ml.train` first.",
        )

    try:
        bundle = joblib.load(model_path)
        model, feature_cols = bundle["model"], bundle["feature_cols"]
    except Exception:
        raise HTTPException(
            status_code=503,
            detail=f"Saved model for '{symbol}' could not be loaded (it may be corrupted or "
                   f"out of date). Re-run `python -m app.ml.train` to regenerate it.",
        )

    rows = (
        db.query(HistoricalPrice)
        .filter(HistoricalPrice.stock_id == stock.id)
        .order_by(HistoricalPrice.date)
        .all()
    )
    if not rows:
        raise HTTPException(
            status_code=503,
            detail=f"No price history in the database for '{symbol}' yet. "
                   f"Run `python -m app.utils.data_fetcher` first.",
        )

    raw_df = pd.DataFrame([{
        "date": r.date, "open": r.open, "high": r.high,
        "low": r.low, "close": r.close, "volume": r.volume,
    } for r in rows])

    latest_features = build_latest_features(raw_df)
    if latest_features is None:
        raise HTTPException(
            status_code=503,
            detail=f"Not enough price history yet for '{symbol}' to compute a prediction "
                   f"(need at least ~21 trading days for rolling features).",
        )

    try:
        X = latest_features[feature_cols].to_frame().T
        raw_predicted_return = float(model.predict(X)[0])
        # Apply the same shrinkage-toward-baseline blend that was validated
        # via CV during training -- serving the raw model output here would
        # be inconsistent with what train.py actually evaluated and reported.
        shrinkage_alpha = bundle.get("shrinkage_alpha", 1.0)  # default 1.0 for older model files
        predicted_return = shrinkage_alpha * raw_predicted_return
    except (KeyError, ValueError):
        raise HTTPException(
            status_code=503,
            detail=f"Model for '{symbol}' expects different features than what's currently "
                   f"available (likely trained on an older version of the feature pipeline). "
                   f"Re-run `python -m app.ml.train` to regenerate it.",
        )

    today_close = float(latest_features["close"])
    predicted_price = today_close * (1 + predicted_return)

    # --- Staleness check: is this model trained on reasonably recent data? ---
    is_stale = False
    last_data_date = bundle.get("last_data_date")
    if last_data_date:
        try:
            trained_on_date = datetime.strptime(last_data_date, "%Y-%m-%d").date()
            latest_available_date = latest_features["date"]
            if hasattr(latest_available_date, "date"):
                latest_available_date = latest_available_date.date()
            gap_days = (latest_available_date - trained_on_date).days
            is_stale = gap_days > STALENESS_THRESHOLD_DAYS
        except (ValueError, TypeError):
            pass  # if metadata is missing/malformed (older model file), skip the check

    response = {
        "symbol": symbol,
        "as_of_date": str(latest_features["date"]),
        "last_close": round(today_close, 2),
        "predicted_next_close": round(predicted_price, 2),
        "predicted_change_pct": round(predicted_return * 100, 2),
        "model_type": bundle.get("model_name", "unknown"),
        "shrinkage_alpha": shrinkage_alpha,
        "model_trained_at": bundle.get("trained_at"),
        "model_is_stale": is_stale,
    }
    if is_stale:
        response["stale_warning"] = (
            "This model was trained on older data than what's now available. "
            "Consider re-running `python -m app.ml.train` to refresh it."
        )
    return response