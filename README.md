# 📊 NEPSE AI Analytics — Complete Market Intelligence Platform 
> Machine Learning · NLP Sentiment · Anomaly Detection · Portfolio Optimization · Flask Web App


## 📌 About the Project

NEPSE AI Analytics is a machine learning powered platform that
provides stock prediction, sentiment analysis, anomaly detection,
and portfolio optimization for the Nepal Stock Exchange (NEPSE).

The system integrates financial data, news sentiment, and
technical indicators to deliver actionable market insights
through an interactive dashboard.

---

## 🏗️ Project Architecture

```
major_project/
├── app.py                          # Flask main application
├── auth.py                         # Session-based authentication
├── chatbot.py                      # AI market assistant
├── requirements.txt
├── nepse_analysis_notebook.ipynb   # Full EDA + ML notebook
│
├── data/
│   ├── nepse_stock_prices.csv      # 250K+ OHLCV records, 124 symbols
│   ├── nepse_financial_news_large.csv
│   ├── nepse_social_sentiment_large.csv
│   └── nepse_company_fundamentals_large.csv
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py              # Data pipeline + feature caching
│   ├── stock_prediction.py         # RF + GB + Ridge ensemble
│   ├── sentiment_analysis.py       # VADER NLP + domain lexicon
│   ├── anomaly_detection.py        # Isolation Forest + Z-score + IQR
│   └── portfolio_recommendation.py # Multi-factor scoring
│
├── templates/
│   ├── login.html                  # Dark finance login page
│   ├── dashboard.html              # Full interactive dashboard
│   └── loading.html
│
└── models/                         # Saved trained models (.pkl)
```

---


### 1. Install dependencies
```bash
pip install flask pandas numpy scikit-learn joblib matplotlib
# Optional: pip install nltk plotly  (fallback scorer used if unavailable)
```

### 2. Run the application
```bash
python app.py
```

### 3. Open browser
```
http://localhost:5000
```

### 4. Login credentials
| Username | Password  | Role    |
|----------|-----------|---------|
| admin    | nepse2024 | Admin   |
| analyst  | nepse2024 | Analyst |
| demo     | demo123   | Viewer  |

---

## 🎯 Features

### 1. User Authentication
- SHA-256 password hashing
- Session-based access control
- Brute-force protection (5 attempts → 5 min lockout)
- Role-based access (admin / analyst / viewer)

### 2. Data Pipeline (`src/data_loader.py`)
- Loads 249,759 stock records across 124 symbols
- Auto-engineers 15+ technical features (MA, RSI, MACD, Bollinger Bands)
- Thread-safe LRU caching for fast responses
- Handles missing values via forward-fill

### 3. Stock Prediction (`src/stock_prediction.py`)
- **Ensemble model**: Random Forest (45%) + Gradient Boosting (40%) + Ridge (15%)
- **25 features**: Lag prices, MAs, RSI, MACD, Bollinger, volatility, volume, cyclical time
- **Output**: 5-day forecast with 95% confidence intervals
- **Signals**: BUY / HOLD / SELL based on predicted direction
- Models cached to `models/` for fast reload

### 4. Sentiment Analysis (`src/sentiment_analysis.py`)
- VADER NLP with NEPSE domain lexicon booster
- Analyzes 1,500 news articles + 1,500 social media posts
- Engagement-weighted social sentiment scoring
- Sector-wise sentiment breakdown (Banking, Hydropower, Insurance, etc.)
- Blended market mood: 60% news + 40% social

### 5. Anomaly Detection (`src/anomaly_detection.py`)
- **Isolation Forest** on multi-dimensional feature space
- **Z-score** for daily return outliers (threshold: 2.5σ)
- **IQR method** for volume spikes (multiplier: 2.5×)
- Severity classification: CRITICAL / HIGH / MEDIUM
- Market-wide health report

### 6. Portfolio Recommendation (`src/portfolio_recommendation.py`)
- Multi-factor composite scoring:
  - **Technical** (MA crossover, RSI, momentum, volatility)
  - **Fundamental** (EPS rank, P/E rank, market cap)
  - **Sentiment** (news/social compound score)
- Risk profiles: Conservative / Moderate / Aggressive
- Score-weighted portfolio allocation
- Sector diversification analysis

### 7. AI Chatbot (`chatbot.py`)
- Intent detection with 12+ categories
- Live market data context injection
- Quick-reply chips for common queries
- Markdown-formatted rich responses

### 8. Web Dashboard (`app.py` + `templates/`)
- **Dark professional UI** with animated grid background
- **6 tabs**: Overview, Market Watch, Prediction, Sentiment, Anomaly, Portfolio
- **Live charts** via Chart.js (price, RSI, MACD, volume, sentiment trend)
- **Real-time API endpoints** for all analytics
- Data loaded in background thread at startup

---

## 📡 API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/status` | Check if data pipeline is ready |
| `GET /api/stock/<symbol>` | OHLCV + technical indicators (last 120 days) |
| `GET /api/predict/<symbol>` | 5-day ML prediction |
| `GET /api/sentiment` | Full sentiment analysis summary |
| `GET /api/anomalies/<symbol>` | Per-symbol anomaly scan |
| `GET /api/portfolio?risk=moderate` | Portfolio recommendations |
| `POST /api/chat` | Chatbot message handler |
| `GET /api/market_summary` | Market-level statistics |

---

## 📓 Jupyter Notebook

`nepse_analysis_notebook.ipynb` covers:
1. Environment setup with dark theme plotting
2. Full EDA: price histories, return distributions, correlation heatmap
3. Feature engineering walkthrough
4. Sentiment analysis with visualizations
5. Model training, evaluation, feature importance
6. Anomaly detection with annotated charts
7. Portfolio optimization across all risk profiles
8. Master 6-panel market dashboard chart

---

## 🛠️ Technology Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.10+, Flask 3.0 |
| ML | scikit-learn (RF, GBM, Ridge, IsolationForest) |
| NLP | VADER + custom NEPSE lexicon |
| Data | pandas, numpy |
| Frontend | HTML5, CSS3, Chart.js 4.4, vanilla JS |
| Serialization | joblib |
| Auth | Flask sessions + SHA-256 |

---

