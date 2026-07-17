import os
import joblib
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Stock, HistoricalPrice
from app.ml.features import build_latest_features

router = APIRouter(prefix="/api/predict", tags=["predict"])

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ml", "saved_models")


@router.get("/{symbol}")
def predict_next_close(symbol: str, db: Session = Depends(get_db)):
    """
    Predicts tomorrow's closing price for a stock, using its saved model.
    Returns 404 if the stock doesn't exist, 503 if no trained model is
    available yet (run `python -m app.ml.train` first).
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

    bundle = joblib.load(model_path)
    model, feature_cols = bundle["model"], bundle["feature_cols"]

    rows = (
        db.query(HistoricalPrice)
        .filter(HistoricalPrice.stock_id == stock.id)
        .order_by(HistoricalPrice.date)
        .all()
    )
    raw_df = pd.DataFrame([{
        "date": r.date, "open": r.open, "high": r.high,
        "low": r.low, "close": r.close, "volume": r.volume,
    } for r in rows])

    latest_features = build_latest_features(raw_df)
    if latest_features is None:
        raise HTTPException(
            status_code=503,
            detail=f"Not enough price history yet for '{symbol}' to compute a prediction.",
        )

    X = latest_features[feature_cols].to_frame().T
    predicted_return = float(model.predict(X)[0])
    today_close = float(latest_features["close"])
    predicted_price = today_close * (1 + predicted_return)

    return {
        "symbol": symbol,
        "as_of_date": str(latest_features["date"]),
        "last_close": round(today_close, 2),
        "predicted_next_close": round(predicted_price, 2),
        "predicted_change_pct": round(predicted_return * 100, 2),
    }