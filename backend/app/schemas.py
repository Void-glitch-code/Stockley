from datetime import date
from pydantic import BaseModel, ConfigDict


class StockOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    symbol: str
    name: str | None
    sector: str | None
    last_price: float | None


class HistoricalPriceOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int


class StockDetailOut(StockOut):
    prices: list[HistoricalPriceOut] = []
