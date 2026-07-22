from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


from app.routes import stocks, predict, auth
from app.routes import stocks, predict, auth, market
...


app = FastAPI(
    title="Stockley API",
    description="Stock price prediction platform - backend API",
    version="0.1.0",
)

# Allow the React frontend (running on a different port) to call this API.
# Tighten allow_origins to your actual frontend URL before deploying.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(stocks.router)
app.include_router(predict.router)
app.include_router(auth.router)
app.include_router(market.router)

@app.get("/")
def root():
    return {"message": "Stockley API is running. Visit /docs for Swagger UI."}


@app.get("/health")
def health_check():
    return {"status": "ok"}