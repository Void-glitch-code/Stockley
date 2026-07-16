from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.database import get_db
from app.models import Stock, HistoricalPrice
from app.schemas import StockOut, StockDetailOut, HistoricalPriceOut

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
