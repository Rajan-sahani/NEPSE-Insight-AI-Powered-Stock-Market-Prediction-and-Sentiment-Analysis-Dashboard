

from logger import logging
import numpy as np
import pandas as pd
from typing import Dict, List
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ─── Detection Methods ───────────────────────────────────────────────────────────

def detect_zscore_anomalies(series: pd.Series, threshold: float = 2.5) -> pd.Series:
    """Z-score based anomaly flag."""
    mu = series.mean()
    sigma = series.std()
    if sigma == 0:
        return pd.Series(False, index=series.index)
    z = (series - mu).abs() / sigma
    return z > threshold


def detect_iqr_anomalies(series: pd.Series, multiplier: float = 2.5) -> pd.Series:
    """IQR-based outlier detection."""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - multiplier * IQR
    upper = Q3 + multiplier * IQR
    return (series < lower) | (series > upper)


def detect_isolation_forest_anomalies(
    df: pd.DataFrame,
    features: List[str],
    contamination: float = 0.03,
) -> pd.Series:
    """Isolation Forest on multi-dimensional feature space."""
    X = df[features].dropna()
    if len(X) < 30:
        return pd.Series(False, index=df.index)

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    iso = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    labels = iso.fit_predict(X_s)  # -1 = anomaly, 1 = normal
    scores = iso.score_samples(X_s)  # lower = more anomalous

    result = pd.Series(False, index=df.index)
    result.loc[X.index] = labels == -1

    score_series = pd.Series(np.nan, index=df.index)
    score_series.loc[X.index] = scores

    return result, score_series


# ─── Comprehensive Anomaly Analysis ─────────────────────────────────────────────

def analyze_anomalies(df: pd.DataFrame, symbol: str) -> Dict:
    """
    Full anomaly detection pipeline for a single stock.
    Returns anomaly records with severity and explanation.
    """
    d = df.copy().sort_values("Date")

    if len(d) < 30:
        return {"symbol": symbol, "anomalies": [], "total_count": 0, "anomaly_rate": 0.0}

    # ── 1. Price jump anomalies (daily return Z-score)
    d["daily_return"] = d["close"].pct_change()
    d["is_return_anomaly"] = detect_zscore_anomalies(d["daily_return"].dropna())

    # ── 2. Volume spike anomalies
    if "traded_quantity" in d.columns and d["traded_quantity"].sum() > 0:
        d["is_volume_anomaly"] = detect_iqr_anomalies(d["traded_quantity"].fillna(0))
    else:
        d["is_volume_anomaly"] = False

    # ── 3. Isolation Forest on combined features
    feature_cols = []
    if "daily_return" in d.columns:
        d["daily_return"] = d["daily_return"].fillna(0)
        feature_cols.append("daily_return")
    if "traded_quantity" in d.columns:
        d["traded_quantity"] = d["traded_quantity"].fillna(0)
        feature_cols.append("traded_quantity")
    if "hl_range" not in d.columns and "high" in d.columns and "low" in d.columns:
        d["hl_range"] = (d["high"] - d["low"]) / d["close"].replace(0, np.nan)
    if "hl_range" in d.columns:
        d["hl_range"] = d["hl_range"].fillna(0)
        feature_cols.append("hl_range")

    if len(feature_cols) >= 2:
        iso_flag, iso_score = detect_isolation_forest_anomalies(d, feature_cols)
        d["is_iso_anomaly"] = iso_flag
        d["iso_score"] = iso_score
    else:
        d["is_iso_anomaly"] = False
        d["iso_score"] = 0.0

    # ── 4. Combine flags
    d["is_anomaly"] = d["is_return_anomaly"] | d["is_volume_anomaly"] | d["is_iso_anomaly"]

    # ── 5. Classify severity
    def _severity(row):
        hits = sum([row.get("is_return_anomaly", False),
                    row.get("is_volume_anomaly", False),
                    row.get("is_iso_anomaly", False)])
        if hits >= 3 or (row.get("daily_return", 0) and abs(row["daily_return"]) > 0.15):
            return "CRITICAL"
        elif hits == 2 or (row.get("daily_return", 0) and abs(row["daily_return"]) > 0.08):
            return "HIGH"
        elif hits == 1:
            return "MEDIUM"
        return "LOW"

    def _explain(row):
        parts = []
        ret = row.get("daily_return", 0) or 0
        if row.get("is_return_anomaly"):
            direction = "spike ▲" if ret > 0 else "crash ▼"
            parts.append(f"Price {direction} {abs(ret)*100:.1f}%")
        if row.get("is_volume_anomaly"):
            parts.append("Unusual volume")
        if row.get("is_iso_anomaly"):
            parts.append("Multi-dimensional outlier")
        return "; ".join(parts) if parts else "Minor irregularity"

    anomaly_df = d[d["is_anomaly"]].copy()
    anomaly_df["severity"] = anomaly_df.apply(_severity, axis=1)
    anomaly_df["explanation"] = anomaly_df.apply(_explain, axis=1)

    records = []
    for _, row in anomaly_df.tail(50).iterrows():  # Last 50 anomalies
        records.append({
            "date": str(row["Date"].date()),
            "close": round(float(row["close"]), 2),
            "daily_return_pct": round(float(row.get("daily_return", 0) or 0) * 100, 2),
            "severity": row["severity"],
            "explanation": row["explanation"],
        })

    total = len(d)
    anomaly_count = int(d["is_anomaly"].sum())

    return {
        "symbol": symbol,
        "anomalies": sorted(records, key=lambda x: x["date"], reverse=True),
        "total_count": anomaly_count,
        "anomaly_rate": round(anomaly_count / total * 100, 2),
        "critical_count": sum(1 for r in records if r["severity"] == "CRITICAL"),
        "high_count": sum(1 for r in records if r["severity"] == "HIGH"),
    }


def get_market_anomaly_report(stock_df: pd.DataFrame) -> Dict:
    """
    Run anomaly detection across all symbols and return market-level summary.
    """
    results = []
    for symbol in stock_df["Symbol"].unique():
        sym_df = stock_df[stock_df["Symbol"] == symbol].copy()
        if len(sym_df) < 30:
            continue
        result = analyze_anomalies(sym_df, symbol)
        if result["total_count"] > 0:
            results.append({
                "symbol": symbol,
                "anomaly_count": result["total_count"],
                "anomaly_rate": result["anomaly_rate"],
                "critical_count": result["critical_count"],
                "latest_anomaly": result["anomalies"][0] if result["anomalies"] else None,
            })

    results_df = pd.DataFrame(results) if results else pd.DataFrame()
    high_risk_symbols = (
        results_df.nlargest(10, "critical_count")["symbol"].tolist()
        if not results_df.empty else []
    )

    return {
        "total_anomalous_symbols": len(results),
        "high_risk_symbols": high_risk_symbols,
        "anomaly_table": results if results else [],
        "market_health": "ALERT" if len(high_risk_symbols) > 5 else "WATCH" if len(high_risk_symbols) > 2 else "STABLE",
    }
