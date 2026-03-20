

import re
from logger import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# ─── NEPSE-Specific Lexicon Boosters ────────────────────────────────────────────
NEPSE_POSITIVE_TERMS = {
    "bullish", "rally", "surge", "growth", "profit", "dividend", "bonus",
    "attractive", "buy", "undervalued", "strong", "positive", "gain",
    "recovery", "uptrend", "breakout", "accumulate", "recommend", "outperform",
    "earnings", "revenue", "expansion", "opportunity", "upgrade", "support",
    "hydropower", "infrastructure", "ipo", "oversubscribed", "listing gain",
}

NEPSE_NEGATIVE_TERMS = {
    "bearish", "crash", "fall", "loss", "decline", "sell", "overvalued",
    "weak", "negative", "downtrend", "breakdown", "liquidate", "underperform",
    "disappointing", "concern", "risk", "volatile", "correction", "selloff",
    "fraud", "scam", "manipulation", "probe", "penalty", "default",
    "suspension", "delisted", "investigation", "fine", "collapse",
}

# Sector keyword mapping for news routing
SECTOR_KEYWORDS = {
    "Banking": ["bank", "nabil", "nica", "scb", "ebl", "sbi", "kbl", "mbl", "deposit", "loan", "npa", "interest"],
    "Hydropower": ["hydropower", "hydro", "mw", "electricity", "nea", "energy", "power", "watt"],
    "Insurance": ["insurance", "life", "nlicl", "nlic", "licn", "premium", "claim", "policy"],
    "Finance": ["finance", "microfinance", "mfi", "nabil", "capital", "investment", "fund"],
    "Telecom": ["telecom", "ntc", "ncell", "mobile", "network", "broadband", "data"],
    "Manufacturing": ["manufacturing", "cement", "sugar", "textile", "production", "factory"],
}


# ─── Lightweight Sentiment Scorer (no external dependencies needed at runtime) ──

def _vader_score(text: str) -> Dict[str, float]:
    """Rule-based sentiment scoring without requiring external VADER install."""
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        import nltk
        try:
            sia = SentimentIntensityAnalyzer()
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
            sia = SentimentIntensityAnalyzer()
        return sia.polarity_scores(text)
    except Exception:
        return _fallback_score(text)


def _fallback_score(text: str) -> Dict[str, float]:
    """Keyword-based fallback scorer."""
    text_lower = text.lower()
    words = set(re.findall(r"\b\w+\b", text_lower))
    pos_hits = len(words & NEPSE_POSITIVE_TERMS)
    neg_hits = len(words & NEPSE_NEGATIVE_TERMS)
    total = pos_hits + neg_hits + 1
    compound = (pos_hits - neg_hits) / total
    compound = max(-1.0, min(1.0, compound))
    return {
        "pos": pos_hits / total,
        "neg": neg_hits / total,
        "neu": max(0, 1 - (pos_hits + neg_hits) / total),
        "compound": compound,
    }


def score_text(text: str) -> Dict[str, float]:
    """Score a single text. Returns compound, pos, neg, neu, label."""
    if not isinstance(text, str) or len(text.strip()) < 3:
        return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0, "label": "Neutral"}

    scores = _vader_score(text)

    # Apply NEPSE domain booster
    text_lower = text.lower()
    words = set(re.findall(r"\b\w+\b", text_lower))
    boost = (len(words & NEPSE_POSITIVE_TERMS) - len(words & NEPSE_NEGATIVE_TERMS)) * 0.05
    compound = max(-1.0, min(1.0, scores["compound"] + boost))

    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    return {
        "compound": round(compound, 4),
        "pos": round(scores["pos"], 4),
        "neg": round(scores["neg"], 4),
        "neu": round(scores["neu"], 4),
        "label": label,
    }


def analyze_news_sentiment(news_df: pd.DataFrame) -> pd.DataFrame:
    """Batch analyze financial news DataFrame."""
    df = news_df.copy()
    scores = df["text"].apply(score_text)
    df["compound"] = scores.apply(lambda x: x["compound"])
    df["sentiment_pos"] = scores.apply(lambda x: x["pos"])
    df["sentiment_neg"] = scores.apply(lambda x: x["neg"])
    df["sentiment_neu"] = scores.apply(lambda x: x["neu"])
    df["sentiment_label"] = scores.apply(lambda x: x["label"])
    df["sector"] = df["text"].apply(_detect_sector)
    return df


def analyze_social_sentiment(social_df: pd.DataFrame) -> pd.DataFrame:
    """Batch analyze social media posts, weighted by engagement."""
    df = social_df.copy()
    scores = df["text"].apply(score_text)
    df["compound"] = scores.apply(lambda x: x["compound"])
    df["sentiment_label"] = scores.apply(lambda x: x["label"])
    # Engagement-weighted compound
    max_eng = df["engagement"].max() if df["engagement"].max() > 0 else 1
    df["weight"] = 1 + (df["engagement"] / max_eng)
    df["weighted_compound"] = df["compound"] * df["weight"]
    return df


def _detect_sector(text: str) -> str:
    text_lower = text.lower()
    for sector, keywords in SECTOR_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return sector
    return "General"


def get_sentiment_summary(news_df: pd.DataFrame, social_df: pd.DataFrame) -> Dict:
    """Compute overall market sentiment dashboard metrics."""
    news_analyzed = analyze_news_sentiment(news_df)
    social_analyzed = analyze_social_sentiment(social_df)

    def distribution(df, col="sentiment_label"):
        counts = df[col].value_counts()
        total = max(len(df), 1)
        return {
            "Positive": round(counts.get("Positive", 0) / total * 100, 1),
            "Negative": round(counts.get("Negative", 0) / total * 100, 1),
            "Neutral": round(counts.get("Neutral", 0) / total * 100, 1),
        }

    news_dist = distribution(news_analyzed)
    social_dist = distribution(social_analyzed)

    # Blended market mood
    news_score = news_analyzed["compound"].mean()
    social_score = social_analyzed["weighted_compound"].sum() / social_analyzed["weight"].sum()
    blended = round(0.6 * news_score + 0.4 * social_score, 4)

    if blended >= 0.05:
        market_mood = "Bullish"
        mood_color = "#22c55e"
    elif blended <= -0.05:
        market_mood = "Bearish"
        mood_color = "#ef4444"
    else:
        market_mood = "Neutral"
        mood_color = "#f59e0b"

    # Sector sentiment breakdown
    sector_sentiment = (
        news_analyzed.groupby("sector")["compound"]
        .mean()
        .round(4)
        .reset_index()
        .rename(columns={"compound": "avg_sentiment"})
        .sort_values("avg_sentiment", ascending=False)
        .to_dict("records")
    )

    # Recent 30-day trend
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=30)
    recent_news = news_analyzed[news_analyzed["Date"] >= cutoff] if not news_analyzed.empty else news_analyzed
    daily_news = (
        recent_news.groupby(recent_news["Date"].dt.date)["compound"]
        .mean().round(4).reset_index()
        .rename(columns={"Date": "date", "compound": "sentiment"})
    )

    return {
        "blended_score": blended,
        "market_mood": market_mood,
        "mood_color": mood_color,
        "news_distribution": news_dist,
        "social_distribution": social_dist,
        "sector_sentiment": sector_sentiment,
        "daily_sentiment_trend": daily_news.to_dict("records"),
        "total_news_analyzed": len(news_analyzed),
        "total_posts_analyzed": len(social_analyzed),
    }
