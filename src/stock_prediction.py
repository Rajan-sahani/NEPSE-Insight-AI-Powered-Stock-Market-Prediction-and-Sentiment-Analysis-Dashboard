

import os
from logger import logging
import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)


# ─── Feature Engineering ────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer ML features from OHLCV data."""
    d = df.copy().sort_values("Date")
    c = d["close"]

    # Price-based features
    d["lag_1"] = c.shift(1)
    d["lag_2"] = c.shift(2)
    d["lag_3"] = c.shift(3)
    d["lag_5"] = c.shift(5)
    d["lag_10"] = c.shift(10)

    # Moving averages & deviations
    d["ma5"] = c.rolling(5).mean()
    d["ma10"] = c.rolling(10).mean()
    d["ma20"] = c.rolling(20).mean()
    d["ma50"] = c.rolling(50).mean()
    d["price_to_ma10"] = c / d["ma10"].replace(0, np.nan)
    d["price_to_ma20"] = c / d["ma20"].replace(0, np.nan)

    # Bollinger Bands
    d["bb_std"] = c.rolling(20).std()
    d["bb_upper"] = d["ma20"] + 2 * d["bb_std"]
    d["bb_lower"] = d["ma20"] - 2 * d["bb_std"]
    d["bb_width"] = (d["bb_upper"] - d["bb_lower"]) / d["ma20"].replace(0, np.nan)
    d["bb_position"] = (c - d["bb_lower"]) / (d["bb_upper"] - d["bb_lower"]).replace(0, np.nan)

    # Momentum & returns
    d["return_1d"] = c.pct_change(1)
    d["return_5d"] = c.pct_change(5)
    d["return_10d"] = c.pct_change(10)
    d["momentum_5"] = c - c.shift(5)

    # Volatility
    d["volatility_10"] = d["return_1d"].rolling(10).std()
    d["volatility_20"] = d["return_1d"].rolling(20).std()

    # RSI
    d["rsi"] = _compute_rsi(c, 14)

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    d["macd"] = ema12 - ema26
    d["macd_signal"] = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"] = d["macd"] - d["macd_signal"]

    # Volume features
    if "traded_quantity" in d.columns:
        d["vol_ma5"] = d["traded_quantity"].rolling(5).mean()
        d["vol_ratio"] = d["traded_quantity"] / d["vol_ma5"].replace(0, np.nan)
    else:
        d["vol_ratio"] = 1.0

    # High-Low range
    if "high" in d.columns and "low" in d.columns:
        d["hl_range"] = (d["high"] - d["low"]) / c.replace(0, np.nan)

    # Day of week (cyclical)
    d["dow_sin"] = np.sin(2 * np.pi * d["Date"].dt.dayofweek / 5)
    d["dow_cos"] = np.cos(2 * np.pi * d["Date"].dt.dayofweek / 5)

    # Target: next day close
    d["target"] = c.shift(-1)

    return d


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


FEATURE_COLS = [
    "lag_1", "lag_2", "lag_3", "lag_5", "lag_10",
    "ma5", "ma10", "ma20", "price_to_ma10", "price_to_ma20",
    "bb_width", "bb_position",
    "return_1d", "return_5d", "return_10d", "momentum_5",
    "volatility_10", "volatility_20",
    "rsi", "macd", "macd_signal", "macd_hist",
    "vol_ratio", "dow_sin", "dow_cos",
]


# ─── Model Training ──────────────────────────────────────────────────────────────

class NEPSEPredictor:
    """Ensemble stock price predictor with calibrated uncertainty."""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.rf = RandomForestRegressor(
            n_estimators=150, max_depth=10,
            min_samples_leaf=5, random_state=42, n_jobs=-1
        )
        self.gb = GradientBoostingRegressor(
            n_estimators=100, max_depth=4,
            learning_rate=0.05, random_state=42
        )
        self.ridge = Ridge(alpha=10.0)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_importance: Dict = {}
        self.metrics: Dict = {}

    def train(self, df: pd.DataFrame) -> Dict:
        feat_df = build_features(df)
        feat_df = feat_df[FEATURE_COLS + ["target"]].dropna()

        if len(feat_df) < 50:
            raise ValueError(f"Insufficient data for {self.symbol}: {len(feat_df)} rows")

        X = feat_df[FEATURE_COLS].values
        y = feat_df["target"].values

        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)

        self.rf.fit(X_train_s, y_train)
        self.gb.fit(X_train_s, y_train)
        self.ridge.fit(X_train_s, y_train)

        # Ensemble predictions (weighted average)
        pred_test = self._ensemble_predict(X_test_s)

        mae = mean_absolute_error(y_test, pred_test)
        rmse = np.sqrt(mean_squared_error(y_test, pred_test))
        r2 = r2_score(y_test, pred_test)
        mape = np.mean(np.abs((y_test - pred_test) / np.where(y_test != 0, y_test, 1))) * 100

        self.metrics = {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "R2": round(r2, 4), "MAPE": round(mape, 2)}
        self.feature_importance = dict(
            sorted(
                zip(FEATURE_COLS, self.rf.feature_importances_),
                key=lambda x: x[1], reverse=True
            )
        )
        self.is_trained = True
        logger.info(f"[{self.symbol}] Model trained | MAE={mae:.2f} | R2={r2:.4f}")

        # Save
        joblib.dump(self, os.path.join(MODELS_DIR, f"{self.symbol}_predictor.pkl"))
        return self.metrics

    def _ensemble_predict(self, X_scaled: np.ndarray) -> np.ndarray:
        p_rf = self.rf.predict(X_scaled)
        p_gb = self.gb.predict(X_scaled)
        p_ridge = self.ridge.predict(X_scaled)
        return 0.45 * p_rf + 0.40 * p_gb + 0.15 * p_ridge

    def predict_next_n_days(self, df: pd.DataFrame, n: int = 5) -> List[Dict]:
        """Predict next n trading days prices with uncertainty intervals."""
        if not self.is_trained:
            raise RuntimeError("Model not trained yet")

        feat_df = build_features(df)
        last_row = feat_df[FEATURE_COLS].dropna().iloc[-1:].values
        last_price = df["close"].iloc[-1]

        predictions = []
        current_df = df.copy()

        for day in range(1, n + 1):
            X_s = self.scaler.transform(last_row)

            # Point estimate
            pred = self._ensemble_predict(X_s)[0]

            # Uncertainty from RF individual trees
            tree_preds = np.array([t.predict(X_s)[0] for t in self.rf.estimators_])
            std = tree_preds.std()
            ci_low = pred - 1.96 * std
            ci_high = pred + 1.96 * std

            change_pct = (pred - last_price) / last_price * 100
            signal = "BUY" if change_pct > 1.5 else "SELL" if change_pct < -1.5 else "HOLD"

            predictions.append({
                "day": day,
                "predicted_price": round(float(pred), 2),
                "ci_low": round(float(ci_low), 2),
                "ci_high": round(float(ci_high), 2),
                "change_pct": round(float(change_pct), 2),
                "signal": signal,
            })

            # Simulate next row by shifting lags
            last_row[0][0] = pred  # lag_1 = current pred
            last_price = pred

        return predictions


# ─── Public API ─────────────────────────────────────────────────────────────────

def get_or_train_predictor(symbol: str, df: pd.DataFrame) -> "NEPSEPredictor":
    model_path = os.path.join(MODELS_DIR, f"{symbol}_predictor.pkl")
    if os.path.exists(model_path):
        try:
            predictor = joblib.load(model_path)
            logger.info(f"[{symbol}] Loaded cached model")
            return predictor
        except Exception:
            pass
    predictor = NEPSEPredictor(symbol)
    predictor.train(df)
    return predictor


def predict_stock(symbol: str, df: pd.DataFrame, horizon: int = 5) -> Dict:
    """Main prediction entry point - returns full prediction result."""
    try:
        predictor = get_or_train_predictor(symbol, df)
        if not predictor.is_trained:
            predictor.train(df)
        predictions = predictor.predict_next_n_days(df, n=horizon)
        last_price = float(df["close"].iloc[-1])
        target_price = predictions[-1]["predicted_price"]
        overall_change = (target_price - last_price) / last_price * 100

        # Build historical chart data (last 60 days)
        hist = df.tail(60)[["Date", "close", "ma_10", "ma_30"]].copy()
        hist["Date"] = hist["Date"].dt.strftime("%Y-%m-%d")

        return {
            "symbol": symbol,
            "last_price": round(last_price, 2),
            "predictions": predictions,
            "overall_change_pct": round(overall_change, 2),
            "overall_signal": "BUY" if overall_change > 2 else "SELL" if overall_change < -2 else "HOLD",
            "model_metrics": predictor.metrics,
            "feature_importance": dict(list(predictor.feature_importance.items())[:10]),
            "historical_data": hist.to_dict("records"),
        }
    except Exception as e:
        logger.error(f"Prediction failed for {symbol}: {e}")
        return {"symbol": symbol, "error": str(e), "predictions": []}
