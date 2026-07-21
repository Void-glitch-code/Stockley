from datetime import date
from pydantic import BaseModel, ConfigDict
from pydantic import BaseModel, EmailStr

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

class UserCreate(BaseModel):
    email:EmailStr
    password:str

class UserOut(BaseModel):
    id:int
    email:EmailStr
    created_at:date

class Token(BaseModel):
    access_token:str
    token_type:str = "bearer"

