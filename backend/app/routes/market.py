from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Stock, HistoricalPrice
from app.schemas import TrendingResponse, TrendingStock

router = APIRouter(
    prefix="/api/market",
    tags=["market"],
)


@router.get("/trending", response_model=TrendingResponse)
def get_trending(db: Session = Depends(get_db)):
    market = []

    stocks = db.query(Stock).all()

    for stock in stocks:

        prices = (
            db.query(HistoricalPrice)
            .filter(HistoricalPrice.stock_id == stock.id)
            .order_by(HistoricalPrice.date.desc())
            .limit(2)
            .all()
        )

        if len(prices) < 2:
            continue

        latest = prices[0]
        previous = prices[1]

        pct_change = (
            (latest.close - previous.close)
            / previous.close
        ) * 100

        market.append(
            TrendingStock(
                symbol=stock.symbol,
                name=stock.name,
                price=round(latest.close, 2),
                pct_change=round(pct_change, 2),
            )
        )

    gainers = sorted(
        market,
        key=lambda x: x.pct_change,
        reverse=True,
    )[:5]

    losers = sorted(
        market,
        key=lambda x: x.pct_change,
    )[:5]

    return TrendingResponse(
        gainers=gainers,
        losers=losers,
    )