
from logger import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ─── Scoring Engine ──────────────────────────────────────────────────────────────

def compute_technical_score(stock_df: pd.DataFrame) -> float:
    """Score based on price momentum, RSI, and moving average crossovers."""
    d = stock_df.sort_values("Date")
    if len(d) < 30:
        return 0.5

    last = d.iloc[-1]
    score = 0.5  # baseline

    # MA trend: price above MA = bullish
    if "ma_10" in d.columns and pd.notna(last.get("ma_10")):
        if last["close"] > last["ma_10"]:
            score += 0.1
        else:
            score -= 0.1

    if "ma_30" in d.columns and pd.notna(last.get("ma_30")):
        if last["close"] > last["ma_30"]:
            score += 0.05
        else:
            score -= 0.05

    # Golden cross: MA10 > MA30
    if "ma_10" in d.columns and "ma_30" in d.columns:
        recent = d.tail(5)
        if (recent["ma_10"] > recent["ma_30"]).all():
            score += 0.1
        elif (recent["ma_10"] < recent["ma_30"]).all():
            score -= 0.1

    # RSI: oversold (buy opportunity) vs overbought
    if "rsi" in d.columns and pd.notna(last.get("rsi")):
        rsi = last["rsi"]
        if 40 <= rsi <= 60:
            score += 0.05  # healthy range
        elif rsi < 30:
            score += 0.15  # oversold - potential buy
        elif rsi > 70:
            score -= 0.10  # overbought

    # Recent 20-day return momentum
    if len(d) >= 20:
        ret_20 = (d["close"].iloc[-1] - d["close"].iloc[-20]) / d["close"].iloc[-20]
        score += np.clip(ret_20 * 2, -0.2, 0.2)

    # Volatility penalty (prefer stable stocks)
    if "volatility_20d" in d.columns and pd.notna(last.get("volatility_20d")):
        vol = last["volatility_20d"]
        score -= np.clip(vol * 5, 0, 0.15)

    return round(np.clip(score, 0, 1), 4)


def compute_sentiment_score(symbol: str, sentiment_data: Optional[pd.DataFrame]) -> float:
    """Derive sentiment score from news/social data for a symbol."""
    if sentiment_data is None or sentiment_data.empty:
        return 0.5
    # Try to find mentions of this symbol
    mask = sentiment_data["text"].str.contains(symbol, case=False, na=False)
    relevant = sentiment_data[mask]
    if relevant.empty:
        # Use general market sentiment
        return 0.5 + 0.1 * np.sign(sentiment_data["compound"].mean() if "compound" in sentiment_data.columns else 0)
    avg = relevant["compound"].mean() if "compound" in relevant.columns else 0
    return round(0.5 + avg * 0.5, 4)  # normalize to [0,1]


def build_portfolio_score(
    symbol: str,
    stock_df: pd.DataFrame,
    fundamentals_df: pd.DataFrame,
    sentiment_df: Optional[pd.DataFrame] = None,
    risk_profile: str = "moderate",
) -> Dict:
    """
    Compute composite investment score for a single symbol.

    Weights:
        Aggressive:  Tech 40%, Fundamental 30%, Sentiment 30%
        Moderate:    Tech 30%, Fundamental 40%, Sentiment 30%
        Conservative: Tech 20%, Fundamental 50%, Sentiment 30%
    """
    weights = {
        "aggressive":    {"tech": 0.40, "fund": 0.30, "sent": 0.30},
        "moderate":      {"tech": 0.30, "fund": 0.40, "sent": 0.30},
        "conservative":  {"tech": 0.20, "fund": 0.50, "sent": 0.30},
    }.get(risk_profile, {"tech": 0.30, "fund": 0.40, "sent": 0.30})

    # Technical score
    tech_score = compute_technical_score(stock_df)

    # Fundamental score
    fund_row = fundamentals_df[fundamentals_df["Symbol"] == symbol]
    fund_score = float(fund_row["fundamental_score"].iloc[0]) if not fund_row.empty else 0.5
    sector = str(fund_row["Sector"].iloc[0]) if not fund_row.empty else "Unknown"
    eps = float(fund_row["EPS"].iloc[0]) if not fund_row.empty else 0.0
    pe = float(fund_row["PE_Ratio"].iloc[0]) if not fund_row.empty else 0.0

    # Sentiment score
    sent_score = compute_sentiment_score(symbol, sentiment_df)

    # Composite
    composite = (
        weights["tech"] * tech_score +
        weights["fund"] * fund_score +
        weights["sent"] * sent_score
    )

    # Risk classification
    if composite >= 0.70:
        rating = "STRONG BUY"
        risk_level = "Low"
    elif composite >= 0.60:
        rating = "BUY"
        risk_level = "Low-Medium"
    elif composite >= 0.45:
        rating = "HOLD"
        risk_level = "Medium"
    elif composite >= 0.35:
        rating = "WATCH"
        risk_level = "Medium-High"
    else:
        rating = "AVOID"
        risk_level = "High"

    # Target price (simple mean-reversion estimate)
    last_price = float(stock_df["close"].iloc[-1])
    ma30 = stock_df.get("ma_30", pd.Series([last_price])).iloc[-1]
    ma30 = float(ma30) if pd.notna(ma30) else last_price
    target_price = round(0.6 * ma30 + 0.4 * last_price * (1 + (composite - 0.5) * 0.3), 2)

    return {
        "symbol": symbol,
        "sector": sector,
        "current_price": round(last_price, 2),
        "target_price": target_price,
        "upside_pct": round((target_price - last_price) / last_price * 100, 2),
        "composite_score": round(composite, 4),
        "technical_score": round(tech_score, 4),
        "fundamental_score": round(fund_score, 4),
        "sentiment_score": round(sent_score, 4),
        "eps": round(eps, 2),
        "pe_ratio": round(pe, 2),
        "rating": rating,
        "risk_level": risk_level,
    }


# ─── Portfolio Builder ───────────────────────────────────────────────────────────

def build_portfolio(
    stock_df: pd.DataFrame,
    fundamentals_df: pd.DataFrame,
    sentiment_df: Optional[pd.DataFrame] = None,
    risk_profile: str = "moderate",
    top_n: int = 10,
) -> Dict:
    """
    Run full portfolio recommendation across all symbols.
    Returns ranked stock list with allocation weights.
    """
    results = []
    symbols = stock_df["Symbol"].unique()

    for symbol in symbols:
        sym_df = stock_df[stock_df["Symbol"] == symbol].copy()
        if len(sym_df) < 30:
            continue
        try:
            score = build_portfolio_score(symbol, sym_df, fundamentals_df, sentiment_df, risk_profile)
            results.append(score)
        except Exception as e:
            logger.debug(f"Skipping {symbol}: {e}")

    if not results:
        return {"recommendations": [], "portfolio": [], "sector_distribution": {}}

    ranked = sorted(results, key=lambda x: x["composite_score"], reverse=True)

    # Top N picks
    top_picks = [r for r in ranked[:top_n] if r["rating"] not in ("AVOID",)]

    # Portfolio allocation (score-weighted)
    total_score = sum(r["composite_score"] for r in top_picks) or 1
    for r in top_picks:
        r["allocation_pct"] = round(r["composite_score"] / total_score * 100, 1)

    # Sector diversification
    sector_dist = {}
    for r in top_picks:
        sec = r["sector"]
        sector_dist[sec] = sector_dist.get(sec, 0) + r["allocation_pct"]

    # Portfolio level stats
    avg_upside = np.mean([r["upside_pct"] for r in top_picks]) if top_picks else 0
    buy_count = sum(1 for r in top_picks if "BUY" in r["rating"])

    return {
        "recommendations": ranked[:20],  # Full ranked list (top 20)
        "portfolio": top_picks,          # Curated portfolio with allocations
        "sector_distribution": sector_dist,
        "avg_expected_upside": round(avg_upside, 2),
        "buy_count": buy_count,
        "risk_profile": risk_profile,
        "total_symbols_analyzed": len(results),
    }
