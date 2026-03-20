

import os
import pandas as pd
import numpy as np
from logger import logging
from functools import lru_cache

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


# ─── Stock Prices ───────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def load_stock_prices() -> pd.DataFrame:
    """Load and clean NEPSE stock price history."""
    path = os.path.join(DATA_DIR, "nepse_stock_prices.csv")
    logger.info(f"Loading stock prices from {path}")
    df = pd.read_csv(path, parse_dates=["published_date"])
    df.rename(columns={"published_date": "Date"}, inplace=True)
    df.sort_values(["Symbol", "Date"], inplace=True)
    df.drop_duplicates(subset=["Symbol", "Date"], inplace=True)

    # Numeric coercion
    for col in ["open", "high", "low", "close", "traded_quantity", "traded_amount", "per_change"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Forward-fill missing prices within each symbol
    df[["open", "high", "low", "close"]] = (
        df.groupby("Symbol")[["open", "high", "low", "close"]].transform(lambda s: s.ffill())
    )
    df.dropna(subset=["close"], inplace=True)

    # Feature engineering
    df["daily_return"] = df.groupby("Symbol")["close"].pct_change()
    df["volatility_20d"] = df.groupby("Symbol")["daily_return"].transform(
        lambda x: x.rolling(20, min_periods=5).std()
    )
    df["ma_10"] = df.groupby("Symbol")["close"].transform(lambda x: x.rolling(10).mean())
    df["ma_30"] = df.groupby("Symbol")["close"].transform(lambda x: x.rolling(30).mean())
    df["ma_50"] = df.groupby("Symbol")["close"].transform(lambda x: x.rolling(50).mean())
    df["rsi"] = df.groupby("Symbol")["close"].transform(_compute_rsi)

    logger.info(f"Stock prices loaded: {len(df):,} rows, {df['Symbol'].nunique()} symbols")
    return df.reset_index(drop=True)


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ─── Financial News ──────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def load_financial_news() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "nepse_financial_news_large.csv")
    logger.info(f"Loading financial news from {path}")
    df = pd.read_csv(path, parse_dates=["Date"])
    df.dropna(subset=["Title", "Content"], inplace=True)
    df["text"] = df["Title"].fillna("") + " " + df["Content"].fillna("")
    df["text"] = df["text"].str.strip()
    df.sort_values("Date", inplace=True)
    logger.info(f"Financial news loaded: {len(df):,} articles")
    return df.reset_index(drop=True)


# ─── Social Sentiment ────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def load_social_sentiment() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "nepse_social_sentiment_large.csv")
    logger.info(f"Loading social sentiment from {path}")
    df = pd.read_csv(path, parse_dates=["Date"])
    df.rename(columns={"Text": "text"}, inplace=True)
    df.dropna(subset=["text"], inplace=True)
    for col in ["Likes", "Retweets"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    df["engagement"] = df["Likes"] + df["Retweets"]
    df.sort_values("Date", inplace=True)
    logger.info(f"Social sentiment loaded: {len(df):,} posts")
    return df.reset_index(drop=True)


# ─── Company Fundamentals ────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def load_fundamentals() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "nepse_company_fundamentals_large.csv")
    logger.info(f"Loading fundamentals from {path}")
    df = pd.read_csv(path)
    df.rename(columns={"Company": "Symbol"}, inplace=True)
    for col in ["MarketCap", "EPS", "PE_Ratio"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=["Symbol"], inplace=True)
    # Compute fundamental score (higher EPS, lower PE preferred)
    df["fundamental_score"] = (
        df["EPS"].rank(pct=True) * 0.5 +
        (1 - df["PE_Ratio"].rank(pct=True)) * 0.3 +
        df["MarketCap"].rank(pct=True) * 0.2
    )
    logger.info(f"Fundamentals loaded: {len(df)} companies")
    return df.reset_index(drop=True)


# ─── Market Summary ──────────────────────────────────────────────────────────────

def get_market_summary() -> dict:
    """Return high-level market statistics."""
    df = load_stock_prices()
    latest_date = df["Date"].max()
    recent = df[df["Date"] == latest_date]
    prev_date = df[df["Date"] < latest_date]["Date"].max()
    prev = df[df["Date"] == prev_date]

    advances = int((recent["per_change"] > 0).sum())
    declines = int((recent["per_change"] < 0).sum())
    unchanged = int((recent["per_change"] == 0).sum())

    total_turnover = float(recent["traded_amount"].sum())

    # 52-week high/low for each symbol
    one_year_ago = latest_date - pd.Timedelta(days=365)
    yearly = df[df["Date"] >= one_year_ago]
    summary_52w = yearly.groupby("Symbol").agg(
        high_52w=("high", "max"),
        low_52w=("low", "min")
    ).reset_index()

    # Top gainers/losers
    recent_merged = recent.merge(summary_52w, on="Symbol", how="left")
    top_gainers = recent_merged.nlargest(5, "per_change")[["Symbol", "close", "per_change"]].to_dict("records")
    top_losers = recent_merged.nsmallest(5, "per_change")[["Symbol", "close", "per_change"]].to_dict("records")

    return {
        "latest_date": str(latest_date.date()),
        "total_symbols": int(df["Symbol"].nunique()),
        "advances": advances,
        "declines": declines,
        "unchanged": unchanged,
        "total_turnover": round(total_turnover, 2),
        "top_gainers": top_gainers,
        "top_losers": top_losers,
    }


def get_symbol_list() -> list:
    df = load_stock_prices()
    return sorted(df["Symbol"].unique().tolist())


def get_stock_data(symbol: str, days: int = 180) -> pd.DataFrame:
    df = load_stock_prices()
    sym_df = df[df["Symbol"] == symbol].copy()
    if days:
        cutoff = sym_df["Date"].max() - pd.Timedelta(days=days)
        sym_df = sym_df[sym_df["Date"] >= cutoff]
    return sym_df


def invalidate_cache():
    """Clear all cached data (useful after data updates)."""
    load_stock_prices.cache_clear()
    load_financial_news.cache_clear()
    load_social_sentiment.cache_clear()
    load_fundamentals.cache_clear()
    logger.info("All caches cleared.")
