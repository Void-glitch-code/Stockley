# Stockley — Full-Fledged Stock Price Prediction App
## FastAPI + ML + Docker + React Dashboard
### Build Timeline: Now to Mid-August (12 weeks)

---

## Project Overview

**Stockley** is a production-ready stock price prediction platform that:
- Fetches real stock data (historical prices)
- Trains LSTM/Prophet models to predict next 7-30 day prices
- Serves predictions via FastAPI REST API
- Shows live predictions and confidence intervals on React dashboard
- Deployed with Docker for easy scaling
- Deployed live on Railway/Vercel

**Why this impresses off-campus interviewers:**
- Real ML (not just classification, but time series forecasting)
- Full-stack (backend + frontend + database + deployment)
- Production-ready (Docker, error handling, monitoring)
- Real data (actual stock prices, live predictions)
- Shows you can take an idea from concept to deployment

---

## Tech Stack

**Backend**
- FastAPI (async, modern, built for APIs)
- Python 3.10+
- scikit-learn (preprocessing)
- TensorFlow/Keras (LSTM model training)
- OR Prophet (easier, Facebook's time series library)
- pandas, numpy (data manipulation)
- yfinance (fetch stock data)
- PostgreSQL (store predictions, user data)
- SQLAlchemy (ORM)

**Frontend**
- React + TypeScript
- Recharts (stock charts, predictions)
- TailwindCSS (styling)
- Axios (API calls)

**DevOps**
- Docker (containerization)
- Docker Compose (local development)
- Railway/Vercel (deployment)
- GitHub (version control)

**Database**
- PostgreSQL (historical prices, predictions, users)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      STOCKLEY APP                            │
└─────────────────────────────────────────────────────────────┘

Frontend (React)
  ├─ Home: Search stocks, view predictions
  ├─ Dashboard: Historical chart + predicted prices
  ├─ Comparison: Compare 2-3 stocks
  └─ My Watchlist: Saved stocks, alerts

↓↓↓ (API Calls) ↓↓↓

Backend (FastAPI)
  ├─ GET  /api/stocks                  → list all stocks
  ├─ GET  /api/stocks/{symbol}         → stock info + historical data
  ├─ POST /api/predict                 → predict next N days
  ├─ GET  /api/predict/{symbol}        → get latest prediction
  ├─ GET  /api/stocks/{symbol}/chart   → candlestick data
  ├─ POST /api/watchlist               → save stock (auth required)
  ├─ GET  /api/watchlist               → user's saved stocks
  └─ Auth endpoints (register, login)

↓↓↓ (Data) ↓↓↓

Database (PostgreSQL)
  ├─ stocks table (id, symbol, name, sector, last_price)
  ├─ historical_prices (stock_id, date, open, high, low, close, volume)
  ├─ predictions (stock_id, prediction_date, predicted_prices, confidence)
  ├─ users (id, email, password, created_at)
  └─ watchlist (user_id, stock_id, added_at)

↓↓↓ (Data) ↓↓↓

ML Models (Trained)
  ├─ LSTM model for each stock (learns patterns from historical data)
  └─ Predictions updated daily at 4 PM IST (after market close)
```

---

## Week-by-Week Breakdown

### Week 1-2 (Late May): Foundation & Data Pipeline
**Goal: Fetch real data, store in DB, understand what we're predicting**

**Week 1**

**Day 1-2: Project Setup**
- Create FastAPI project structure
  ```
  stockley/
  ├── backend/
  │   ├── app/
  │   │   ├── main.py
  │   │   ├── config.py
  │   │   ├── models.py (SQLAlchemy models)
  │   │   ├── schemas.py (Pydantic schemas)
  │   │   ├── database.py
  │   │   ├── routes/
  │   │   │   ├── stocks.py
  │   │   │   ├── predictions.py
  │   │   │   └── auth.py
  │   │   ├── ml/
  │   │   │   ├── models.py (LSTM/Prophet)
  │   │   │   ├── train.py
  │   │   │   └── predict.py
  │   │   └── utils/
  │   │       ├── data_fetcher.py (yfinance)
  │   │       └── validators.py
  │   ├── requirements.txt
  │   ├── Dockerfile
  │   └── docker-compose.yml
  ├── frontend/
  │   └── (React app, created with Vite)
  └── README.md
  ```
- Setup PostgreSQL locally (Docker container recommended)
- Create `.env` with API keys (yfinance doesn't need keys)

**Day 3-5: Data Pipeline**
- Learn yfinance: fetch historical stock data
  ```python
  import yfinance as yf
  df = yf.download("RELIANCE.NS", start="2023-01-01", end="2024-01-01")
  # Returns OHLCV data (Open, High, Low, Close, Volume)
  ```
- Build data fetcher script that:
  - Downloads 2 years of historical data for 10-20 stocks (TCS, Reliance, HDFC, Infy, etc.)
  - Stores in PostgreSQL
  - Runs daily to update prices
- Create SQLAlchemy models:
  ```python
  class Stock(Base):
      id = Column(Integer, primary_key=True)
      symbol = Column(String, unique=True)
      name = Column(String)
      sector = Column(String)
      last_price = Column(Float)

  class HistoricalPrice(Base):
      id = Column(Integer, primary_key=True)
      stock_id = Column(Integer, ForeignKey("stock.id"))
      date = Column(Date)
      open = Column(Float)
      high = Column(Float)
      low = Column(Float)
      close = Column(Float)
      volume = Column(Integer)
  ```

**Week 2**

**Day 1-2: FastAPI Basics**
- Setup FastAPI app with routes
- First endpoint: `GET /api/stocks` → returns list of stocks
- Second endpoint: `GET /api/stocks/{symbol}` → returns stock info + last 100 days of prices
- Test in Swagger UI at `localhost:8000/docs`

**Day 3-5: Data Exploration**
- Load historical data into pandas
- Visualize (plot closing prices over time)
- Calculate technical indicators (SMA, EMA, RSI, MACD) — optional but good
- Understand data shape: 500+ days of data per stock
- Check for missing values, handle them
- Normalize data (prices vary widely, ML needs normalized input)

**Hands-on Deliverable:**
- PostgreSQL with 10 stocks and 2 years of daily data
- `/api/stocks` endpoint working
- `/api/stocks/RELIANCE/chart` returns last 100 days as JSON

---

### Week 3-4 (June 1-15): ML Model Training
**Goal: Train LSTM models that learn stock price patterns**

**Week 3**

**Day 1-2: LSTM Basics**
- Understand LSTM (Long Short-Term Memory) for time series
  - LSTM learns patterns from sequence of prices
  - Input: last N days of prices → Output: next day's price
  - Example: Feed last 60 days → predict day 61
- Alternative: Prophet (easier, Facebook's library, better for beginner)
  - Pros: Auto-detects seasonality, trends, less tuning
  - Cons: Less flexible than LSTM
- **Recommendation: Start with Prophet (2-3 days), if comfortable add LSTM (3-4 days)**

**Day 3: Prophet Model**
```python
from prophet import Prophet

# Prepare data in format Prophet expects
df_prophet = df[['date', 'close']].copy()
df_prophet.columns = ['ds', 'y']

# Train
model = Prophet(yearly_seasonality=True, daily_seasonality=False)
model.fit(df_prophet)

# Predict next 30 days
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# Get predictions with confidence intervals
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30)
```

**Day 4-5: LSTM Model** (if doing this)
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Prepare sequences: convert [p1, p2, ..., p60] → p61
def create_sequences(data, lookback=60):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(data[i])
    return np.array(X), np.array(y)

# Build model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(60, 1)),
    LSTM(32),
    Dense(16, activation='relu'),
    Dense(1)  # Output: predicted price
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50, validation_split=0.2)
```

**Week 4**

**Day 1-2: Train Models for All Stocks**
- For each stock, train a separate model
- Save models to disk (joblib for Prophet, HDF5 for LSTM)
- Track model performance: MAE, RMSE on test set
- Create `/models/` directory with saved models

**Day 3-5: Prediction Endpoint**
- Build `POST /api/predict` endpoint that:
  - Takes symbol and number of days to predict (e.g., 7, 14, 30)
  - Loads trained model
  - Generates predictions
  - Returns predictions with confidence intervals
  - Stores in database for history
  ```python
  @app.post("/api/predict")
  def predict(symbol: str, days: int = 7):
      # Load model
      model = load_model(f"models/{symbol}_prophet.pkl")
      
      # Predict
      future = model.make_future_dataframe(periods=days)
      forecast = model.predict(future)
      
      # Store in DB
      for idx, row in forecast.tail(days).iterrows():
          db.add(Prediction(
              stock_id=get_stock_id(symbol),
              date=row['ds'],
              predicted_close=row['yhat'],
              confidence_lower=row['yhat_lower'],
              confidence_upper=row['yhat_upper']
          ))
      db.commit()
      
      return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict()
  ```

**Hands-on Deliverable:**
- Trained models for 10 stocks saved
- `/api/predict?symbol=RELIANCE&days=7` returns predictions
- Predictions stored in database with confidence intervals

---

### Week 5-6 (June 15-30): Backend Complete + Database
**Goal: Full-featured FastAPI backend, ready for frontend**

**Week 5**

**Day 1-2: Authentication**
- User registration and login (JWT tokens)
- `POST /api/auth/register` → create account
- `POST /api/auth/login` → get token
- Protect endpoints with `@app.get(..., dependencies=[Depends(get_current_user)])`

**Day 3-5: Watchlist & User Features**
- `POST /api/watchlist` → add stock to user's watchlist
- `GET /api/watchlist` → get user's saved stocks
- `DELETE /api/watchlist/{symbol}` → remove from watchlist
- Database: `watchlist` table linking users to stocks

**Week 6**

**Day 1-2: Advanced Endpoints**
- `GET /api/stocks/{symbol}/comparison?compare_with=INFY` → compare 2 stocks side by side
- `GET /api/stocks/{symbol}/historical?days=90` → last N days of historical data
- `GET /api/stocks/sector/{sector}` → all stocks in a sector
- `GET /api/market/trending` → top gainers/losers
- Filtering: `/api/stocks?sector=IT&min_price=100&max_price=5000`

**Day 3-4: Error Handling & Validation**
- Custom exception handling
- Input validation (symbol exists, days > 0, etc.)
- Consistent error response format
  ```json
  {"error": "Stock not found", "status": 404}
  ```

**Day 5: Documentation & Testing**
- Swagger docs at `/docs`
- Test all endpoints with Postman
- Write docstrings for every endpoint

**Hands-on Deliverable:**
- Full-featured backend API
- 20+ endpoints working
- Authentication and user features
- Swagger documentation complete
- All endpoints tested

---

### Week 7-8 (July 1-15): Frontend (React Dashboard)
**Goal: Beautiful, interactive dashboard showing predictions**

**Week 7**

**Day 1: Setup React Project**
- Create with Vite: `npm create vite@latest stockley-frontend -- --template react`
- Install dependencies:
  ```
  npm install axios recharts tailwindcss react-router-dom zustand
  ```
- Folder structure:
  ```
  src/
  ├── components/
  │   ├── StockCard.jsx
  │   ├── PredictionChart.jsx
  │   ├── Navbar.jsx
  │   └── Watchlist.jsx
  ├── pages/
  │   ├── Home.jsx
  │   ├── Dashboard.jsx
  │   └── Login.jsx
  ├── api/
  │   └── client.js (axios instance)
  ├── store/
  │   └── authStore.js (Zustand for auth state)
  └── App.jsx
  ```

**Day 2-3: Home Page**
- Search bar: search stocks by symbol or name
- Popular stocks section
- Login/Register forms (modal)
- Navigation

**Day 4-5: Dashboard (Core)**
- URL: `/stock/RELIANCE`
- Show:
  - Current price (green if up, red if down)
  - Last 100 days historical chart (candlestick or line)
  - Next 7-day prediction as line overlay on chart
  - Confidence interval (shaded area)
  - Key stats (52-week high/low, market cap, P/E ratio from yfinance)
  - Add to watchlist button
- Use Recharts for beautiful charts

**Week 8**

**Day 1-2: Advanced Features**
- Comparison view: `/compare?stocks=RELIANCE,INFY,TCS`
  - Multiple stocks on same chart
  - Side-by-side comparison
- Watchlist page: `/watchlist`
  - User's saved stocks with latest predictions
  - Quick links to dashboard
  - Alerts (notify if price moves X%)
- Trending page: `/trending`
  - Top gainers/losers
  - Sector performance

**Day 3-4: Polish & Responsive Design**
- Mobile responsive (TailwindCSS)
- Dark mode toggle (nice to have)
- Loading states and error handling
- Smooth animations

**Day 5: Testing**
- Test all pages locally
- Verify API integration
- Fix bugs

**Hands-on Deliverable:**
- Full React dashboard deployed locally
- All major pages working
- Beautiful, professional UI
- Connected to backend API

---

### Week 9-10 (July 15-30): Docker & Deployment
**Goal: Everything containerized, deployed live**

**Week 9**

**Day 1-2: Docker for Backend**
- `Dockerfile` for FastAPI app
  ```dockerfile
  FROM python:3.10-slim
  WORKDIR /app
  COPY requirements.txt .
  RUN pip install -r requirements.txt
  COPY . .
  CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
  ```
- `docker-compose.yml` for local development (FastAPI + PostgreSQL)
  ```yaml
  version: '3.8'
  services:
    db:
      image: postgres:15
      environment:
        POSTGRES_DB: stockley
        POSTGRES_USER: stockley
        POSTGRES_PASSWORD: secret
      ports:
        - "5432:5432"
    
    api:
      build: ./backend
      ports:
        - "8000:8000"
      depends_on:
        - db
      environment:
        DATABASE_URL: postgresql://stockley:secret@db/stockley
  ```
- Test: `docker-compose up` → app runs in container

**Day 3-5: Deploy Backend to Railway**
- Connect GitHub repo to Railway
- Set environment variables (DATABASE_URL, SECRET_KEY, etc.)
- Railway auto-deploys on push
- Verify API works at `yourdomain.railway.app/docs`

**Week 10**

**Day 1-2: Docker for Frontend**
- `Dockerfile` for React app
  ```dockerfile
  FROM node:18 AS build
  WORKDIR /app
  COPY package*.json .
  RUN npm install
  COPY . .
  RUN npm run build
  
  FROM nginx:alpine
  COPY --from=build /app/dist /usr/share/nginx/html
  EXPOSE 80
  CMD ["nginx", "-g", "daemon off;"]
  ```
- Test locally

**Day 3: Deploy Frontend to Vercel**
- Connect GitHub repo to Vercel
- Set environment variables (REACT_APP_API_URL pointing to Railway backend)
- Auto-deploys on push
- Frontend live at `yourdomain.vercel.app`

**Day 4-5: Integration Testing**
- Frontend talks to live backend
- Test full user flow: search → view prediction → add to watchlist → login → view watchlist
- Fix CORS issues if any
- Performance optimization (lazy loading, caching)

**Hands-on Deliverable:**
- Backend live at `stockley-api.railway.app`
- Frontend live at `stockley-dashboard.vercel.app`
- Both connected and working end-to-end
- Docker images ready for scaling

---

### Week 11-12 (July 30 - August 15): Polish & Documentation
**Goal: Production-ready, impressive portfolio piece**

**Week 11**

**Day 1-2: Advanced Features (Nice-to-haves)**
- Daily model retraining (scheduled task): every day at 4 PM, retrain models with new data
  - Use APScheduler or Celery for background tasks
- Price alerts: notify user when stock hits target price
- Technical indicators on chart (SMA, EMA, RSI, MACD)
- Export prediction as PDF/CSV

**Day 3-4: Code Quality**
- Add logging (FastAPI logging, React error boundaries)
- Error handling (500 errors, timeout, API failures)
- Unit tests (at least 10-15 tests for backend)
- Performance: optimize slow queries, add caching

**Day 5: Security**
- Input validation (no SQL injection, XSS protection)
- HTTPS only (Vercel/Railway auto-provide)
- Rate limiting on API (prevent spam)
- Hide sensitive data (API keys, secrets in .env)

**Week 12**

**Day 1-2: Documentation**
- Professional `README.md`:
  ```markdown
  # Stockley — Stock Price Prediction Platform
  
  ## Features
  - Real-time stock data (20+ Indian stocks)
  - LSTM/Prophet predictions with confidence intervals
  - User watchlist and alerts
  - Beautiful React dashboard
  
  ## Tech Stack
  - Backend: FastAPI, PostgreSQL, LSTM/Prophet
  - Frontend: React, Recharts, TailwindCSS
  - Deployment: Docker, Railway, Vercel
  
  ## Setup
  1. Clone repo
  2. Docker: `docker-compose up`
  3. Frontend: `npm install && npm run dev`
  4. Open `localhost:3000`
  
  ## API Documentation
  Swagger docs: `http://localhost:8000/docs`
  
  ## Demo
  Live: https://stockley-dashboard.vercel.app
  ```
- API documentation (Swagger auto-generates)
- Architecture diagram
- Model performance metrics (MAE, RMSE for each stock)

**Day 3-4: Final Polish**
- Bug fixes
- UI refinements
- Performance optimization
- Screenshot/GIF for README (app in action)

**Day 5: Launch**
- Push final code to GitHub
- Share live link
- Tweet/LinkedIn post about it (optional but good for visibility)

**Hands-on Deliverable:**
- Production-ready Stockley deployed live
- GitHub repo with professional README
- Swagger documentation
- Clean code, proper error handling
- Ready to demo in interviews

---

## Key Technical Decisions

### LSTM vs Prophet?

**Prophet (Recommended for Timeline)**
- ✅ Easier to learn and implement
- ✅ Automatically handles seasonality
- ✅ Gives confidence intervals by default
- ✅ Fast (no GPU needed)
- ❌ Less flexibility, can't customize architecture

**LSTM (More Impressive)**
- ✅ Shows deep learning knowledge
- ✅ Customizable, can add features (volume, technical indicators)
- ✅ Better for complex patterns
- ❌ Harder to tune, needs GPU for training (slower on CPU)
- ❌ Takes longer to implement

**Recommendation: Start with Prophet (week 3-4), if comfortable add LSTM later.**

---

## Data Storage Strategy

```python
# Database Schema

Stock
├── id (Primary Key)
├── symbol (RELIANCE, TCS, INFY)
├── name (Full name)
├── sector (IT, Finance, Auto)
└── last_price

HistoricalPrice
├── id
├── stock_id (FK)
├── date
├── open, high, low, close, volume

Prediction
├── id
├── stock_id (FK)
├── prediction_date
├── predicted_close
├── confidence_lower
├── confidence_upper
├── created_at

User (if adding auth)
├── id
├── email
├── password_hash
└── created_at

Watchlist
├── user_id (FK)
└── stock_id (FK)
```

---

## Daily/Weekly Tasks (After Launch)

Once live, maintain with:
- **Daily at 4:15 PM**: Fetch new closing prices, retrain models
- **Weekly**: Check model performance, retrain if MAE > threshold
- **Monthly**: Update stock list, add new stocks if requested

---

## Deliverables by August 15

✅ **Backend**
- FastAPI API with 25+ endpoints
- LSTM/Prophet models trained for 15+ stocks
- PostgreSQL database with 2 years of historical data
- JWT authentication
- Deployed on Railway

✅ **Frontend**
- React dashboard with stock search, prediction charts, watchlist
- Responsive design (mobile + desktop)
- Deployed on Vercel
- Connected to live backend

✅ **DevOps**
- Docker containerization (backend + frontend)
- GitHub repo with clean code
- Professional README and API docs
- Live at `stockley-dashboard.vercel.app`

✅ **Portfolio Quality**
- Shows full-stack skills (ML + backend + frontend)
- Production-ready (error handling, validation, logging)
- Real data, real predictions, real users
- Impressive for off-campus interviews and hiring

---

## Interview Talking Points

When interviewers ask about Stockley:

1. **"Why time series forecasting?"**
   - _Time series is one of the most practical ML applications. Stock prices follow patterns — trends, seasonality, volatility. LSTM/Prophet learns these patterns to make predictions._

2. **"How do you handle model drift?"**
   - _Models are retrained daily with new data. If MAE increases, I flag it for manual review._

3. **"What's your prediction accuracy?"**
   - _RMSE around 2-3% for 7-day predictions. Confidence intervals account for uncertainty._

4. **"Why FastAPI over Django?"**
   - _FastAPI is async-first, built for modern APIs, automatic API documentation. Better for ML serving._

5. **"What would you improve?"**
   - _Add ensemble models (combining LSTM + Prophet + Random Forest). Real-time streaming data. Options/futures predictions._

---

## Success Metrics

By August 15:

| Metric | Target |
|--------|--------|
| Users can search 20+ stocks | ✅ |
| Predictions load in <1s | ✅ |
| Model RMSE < 3% | ✅ |
| API uptime > 99% | ✅ |
| Code test coverage > 60% | ✅ |
| GitHub stars (if shared) | 50+ ⭐ |

---

## GitHub Structure

```
stockley/
├── README.md (professional, with screenshots)
├── LICENSE
├── docker-compose.yml
├── .gitignore
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── models.py
│   │   ├── schemas.py
│   │   ├── database.py
│   │   ├── routes/
│   │   ├── ml/
│   │   └── utils/
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── tests/
│   └── models/ (saved .pkl files)
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── api/
│   │   └── App.jsx
│   ├── package.json
│   ├── Dockerfile
│   └── .env.example
└── docs/
    ├── ARCHITECTURE.md
    ├── API_DOCS.md
    └── SETUP.md
```

---

## Timeline Summary

| Period | Focus | Output |
|--------|-------|--------|
| Week 1-2 | Data pipeline, setup | DB with 2 yrs data |
| Week 3-4 | ML models | Trained Prophet/LSTM |
| Week 5-6 | Backend APIs | 25+ endpoints |
| Week 7-8 | React frontend | Dashboard live locally |
| Week 9-10 | Docker & deployment | Everything in cloud |
| Week 11-12 | Polish & docs | Production-ready |

---

## Final Advice

1. **Start simple** — Prophet first, LSTM later if time
2. **Deploy early** — Get something live by week 9, iterate from there
3. **Real data** — Use actual stock prices, real predictions
4. **Professional README** — Recruiters judge projects by documentation
5. **Show reasoning** — Be able to explain every technical choice
6. **Interview ready** — Practice explaining the project, architecture, trade-offs

This is a 12-week journey that'll take you from concept to a portfolio piece that absolutely impresses in off-campus interviews.

 🚀

---

## Resources

- FastAPI: https://fastapi.tiangolo.com
- Prophet: https://facebook.github.io/prophet
- TensorFlow/LSTM: https://www.tensorflow.org/guide/keras/rnn
- yfinance: https://github.com/ranaroussi/yfinance
- Recharts: https://recharts.org
- Railway: https://railway.app
- Vercel: https://vercel.com
