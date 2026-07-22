from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.database import get_db
from app.models import Stock, HistoricalPrice
from app.schemas import StockOut, StockDetailOut, HistoricalPriceOut
from statistics import mean, stdev
from app.schemas import ComparisonStock, ComparisonStats, ComparisonPoint, StockComparisonOut

router = APIRouter(prefix="/api/stocks", tags=["stocks"])


@router.get("", response_model=list[StockOut])
def list_stocks(db: Session = Depends(get_db)):
    """Returns all stocks tracked by Stockley."""
    return db.query(Stock).order_by(Stock.symbol).all()


@router.get("/{symbol}", response_model=StockDetailOut)
def get_stock(symbol: str, days: int = 100, db: Session = Depends(get_db)):
    """Returns stock info plus the last N days of historical prices (default 100)."""
    stock = db.query(Stock).filter(Stock.symbol == symbol.upper()).first()
    if stock is None:
        raise HTTPException(status_code=404, detail=f"Stock '{symbol}' not found")

    prices = (
        db.query(HistoricalPrice)
        .filter(HistoricalPrice.stock_id == stock.id)
        .order_by(desc(HistoricalPrice.date))
        .limit(days)
        .all()
    )
    prices.reverse()  # oldest -> newest, easier for charting

    return StockDetailOut(
        id=stock.id,
        symbol=stock.symbol,
        name=stock.name,
        sector=stock.sector,
        last_price=stock.last_price,
        prices=[HistoricalPriceOut.model_validate(p) for p in prices],
    )


@router.get("/{symbol}/chart", response_model=list[HistoricalPriceOut])
def get_stock_chart(symbol: str, days: int = 100, db: Session = Depends(get_db)):
    """Returns just the OHLCV series for charting (no stock metadata wrapper)."""
    stock = db.query(Stock).filter(Stock.symbol == symbol.upper()).first()
    if stock is None:
        raise HTTPException(status_code=404, detail=f"Stock '{symbol}' not found")

    prices = (
        db.query(HistoricalPrice)
        .filter(HistoricalPrice.stock_id == stock.id)
        .order_by(desc(HistoricalPrice.date))
        .limit(days)
        .all()
    )
    prices.reverse()
    return prices

@router.get("/{symbol}/comparison", response_model=StockComparisonOut)
def compare_stocks(
    symbol: str,
    compare_with: str,
    db: Session = Depends(get_db),
):
    # Prevent comparing a stock with itself
    if symbol.upper() == compare_with.upper():
        raise HTTPException(
            status_code=400,
            detail="Cannot compare a stock with itself."
        )

    stock_a = (
        db.query(Stock)
        .filter(Stock.symbol == symbol.upper())
        .first()
    )

    if stock_a is None:
        raise HTTPException(
            status_code=404,
            detail=f"Stock '{symbol}' not found."
        )

    stock_b = (
        db.query(Stock)
        .filter(Stock.symbol == compare_with.upper())
        .first()
    )

    if stock_b is None:
        raise HTTPException(
            status_code=404,
            detail=f"Stock '{compare_with}' not found."
        )

    prices_a = (
        db.query(HistoricalPrice)
        .filter(HistoricalPrice.stock_id == stock_a.id)
        .order_by(HistoricalPrice.date)
        .all()
    )

    prices_b = (
        db.query(HistoricalPrice)
        .filter(HistoricalPrice.stock_id == stock_b.id)
        .order_by(HistoricalPrice.date)
        .all()
    )

    if not prices_a or not prices_b:
        raise HTTPException(
            status_code=404,
            detail="Historical price data missing."
        )

    series_a = {p.date: p.close for p in prices_a}
    series_b = {p.date: p.close for p in prices_b}

    common_dates = sorted(
        set(series_a.keys()) & set(series_b.keys())
    )

    if len(common_dates) < 2:
        raise HTTPException(
            status_code=400,
            detail="Not enough overlapping historical data."
        )

    closes_a = [series_a[d] for d in common_dates]
    closes_b = [series_b[d] for d in common_dates]

    base_a = closes_a[0]
    base_b = closes_b[0]

    normalized = []

    for d, a, b in zip(common_dates, closes_a, closes_b):
        normalized.append(
            ComparisonPoint(
                date=d,
                stock_a=(a / base_a) * 100,
                stock_b=(b / base_b) * 100,
            )
        )

    return_a = ((closes_a[-1] - closes_a[0]) / closes_a[0]) * 100
    return_b = ((closes_b[-1] - closes_b[0]) / closes_b[0]) * 100

    daily_returns_a = []

    for i in range(1, len(closes_a)):
        daily_returns_a.append(
            ((closes_a[i] - closes_a[i - 1]) / closes_a[i - 1]) * 100
        )

    daily_returns_b = []

    for i in range(1, len(closes_b)):
        daily_returns_b.append(
            ((closes_b[i] - closes_b[i - 1]) / closes_b[i - 1]) * 100
        )

    volatility_a = stdev(daily_returns_a) if len(daily_returns_a) > 1 else 0
    volatility_b = stdev(daily_returns_b) if len(daily_returns_b) > 1 else 0

    if return_a > return_b:
        winner = stock_a.symbol
    elif return_b > return_a:
        winner = stock_b.symbol
    else:
        winner = "Tie"

    return StockComparisonOut(
        stock_a=ComparisonStock(
            symbol=stock_a.symbol,
            name=stock_a.name,
            sector=stock_a.sector,
            return_pct=round(return_a, 2),
            volatility=round(volatility_a, 2),
        ),
        stock_b=ComparisonStock(
            symbol=stock_b.symbol,
            name=stock_b.name,
            sector=stock_b.sector,
            return_pct=round(return_b, 2),
            volatility=round(volatility_b, 2),
        ),
        comparison=ComparisonStats(
            winner=winner,
            normalized_series=normalized,
        ),
    )
    
