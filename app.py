
import os
from logger import logging
import json
import threading
from datetime import datetime
from flask import (
    Flask, render_template, request, redirect,
    url_for, flash, session, jsonify
)

# ── Local imports
from auth import login_user, set_session, clear_session, is_authenticated, get_current_user, login_required
from chatbot import NEPSEChatbot

# ── Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Flask app setup
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "nepse_ai_super_secret_2024_xK9mP")
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

# ── Global data cache (loaded once at startup)
_data_cache = {}
_data_lock = threading.Lock()
_chatbot = NEPSEChatbot()


def _load_all_data():
    """Load and preprocess all datasets at startup."""
    global _data_cache
    logger.info("Initializing NEPSE data pipeline...")

    from src.data_loader import (
        load_stock_prices, load_financial_news,
        load_social_sentiment, load_fundamentals, get_market_summary
    )
    from src.sentiment_analysis import (
        analyze_news_sentiment, analyze_social_sentiment, get_sentiment_summary
    )
    from src.anomaly_detection import get_market_anomaly_report
    from src.portfolio_recommendation import build_portfolio

    with _data_lock:
        stock_df = load_stock_prices()
        news_df = load_financial_news()
        social_df = load_social_sentiment()
        fund_df = load_fundamentals()

        logger.info("Running sentiment analysis...")
        news_analyzed = analyze_news_sentiment(news_df)
        social_analyzed = analyze_social_sentiment(social_df)
        sentiment_summary = get_sentiment_summary(news_df, social_df)

        logger.info("Building portfolio recommendations...")
        portfolio_result = build_portfolio(stock_df, fund_df, news_analyzed)

        logger.info("Computing market summary...")
        market_summary = get_market_summary()

        _data_cache = {
            "stock_df": stock_df,
            "news_df": news_analyzed,
            "social_df": social_analyzed,
            "fund_df": fund_df,
            "sentiment_summary": sentiment_summary,
            "portfolio_result": portfolio_result,
            "market_summary": market_summary,
            "loaded_at": datetime.now().isoformat(),
        }

        # Feed chatbot context
        _chatbot.load_context(
            stock_df=stock_df,
            sentiment_summary=sentiment_summary,
            portfolio_result=portfolio_result,
            market_summary=market_summary,
        )

    logger.info("Data pipeline complete.")


# ── Load data in background thread
_init_thread = threading.Thread(target=_load_all_data, daemon=True)
_init_thread.start()


def _cache_ready() -> bool:
    return bool(_data_cache)


# ─── Auth Routes ──────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    if is_authenticated():
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if is_authenticated():
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        ip = request.remote_addr or "0.0.0.0"
        result = login_user(username, password, ip)
        if result["success"]:
            set_session(result["user"])
            return redirect(url_for("dashboard"))
        else:
            flash(result["message"], "error")
    return render_template("login.html")


@app.route("/logout")
def logout():
    clear_session()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))


# ─── Main Dashboard ───────────────────────────────────────────────────────────────

@app.route("/dashboard")
@login_required
def dashboard():
    user = get_current_user()
    if not _cache_ready():
        return render_template("loading.html", user=user)

    d = _data_cache
    ms = d["market_summary"]
    ss = d["sentiment_summary"]
    portfolio = d["portfolio_result"]

    # Symbols for dropdown
    symbols = sorted(d["stock_df"]["Symbol"].unique().tolist())

    return render_template(
        "dashboard.html",
        user=user,
        market_summary=ms,
        sentiment_summary=ss,
        portfolio=portfolio,
        symbols=symbols,
        loaded_at=d["loaded_at"],
    )


# ─── API Endpoints ────────────────────────────────────────────────────────────────

@app.route("/api/status")
def api_status():
    return jsonify({"ready": _cache_ready(), "loaded_at": _data_cache.get("loaded_at", None)})


@app.route("/api/stock/<symbol>")
@login_required
def api_stock_data(symbol):
    if not _cache_ready():
        return jsonify({"error": "Data loading..."}), 503
    df = _data_cache["stock_df"]
    sym_df = df[df["Symbol"] == symbol].copy()
    if sym_df.empty:
        return jsonify({"error": f"Symbol {symbol} not found"}), 404

    # Last 120 days
    sym_df = sym_df.tail(120)
    result = {
        "symbol": symbol,
        "dates": sym_df["Date"].dt.strftime("%Y-%m-%d").tolist(),
        "open": sym_df["open"].round(2).tolist(),
        "high": sym_df["high"].round(2).tolist(),
        "low": sym_df["low"].round(2).tolist(),
        "close": sym_df["close"].round(2).tolist(),
        "volume": sym_df["traded_quantity"].fillna(0).astype(int).tolist(),
        "ma10": sym_df["ma_10"].round(2).tolist(),
        "ma30": sym_df["ma_30"].round(2).tolist(),
        "rsi": sym_df["rsi"].round(2).tolist(),
        "daily_return": sym_df["daily_return"].round(4).tolist(),
    }
    return jsonify(result)


@app.route("/api/predict/<symbol>")
@login_required
def api_predict(symbol):
    if not _cache_ready():
        return jsonify({"error": "Data loading..."}), 503
    from src.stock_prediction import predict_stock
    df = _data_cache["stock_df"]
    sym_df = df[df["Symbol"] == symbol].copy()
    if sym_df.empty:
        return jsonify({"error": f"Symbol {symbol} not found"}), 404
    result = predict_stock(symbol, sym_df, horizon=5)
    return jsonify(result)


@app.route("/api/sentiment")
@login_required
def api_sentiment():
    if not _cache_ready():
        return jsonify({"error": "Data loading..."}), 503
    ss = _data_cache["sentiment_summary"]
    # Serialize datetime objects in daily trend
    trend = []
    for item in ss.get("daily_sentiment_trend", []):
        trend.append({"date": str(item["date"]), "sentiment": item["sentiment"]})
    return jsonify({**ss, "daily_sentiment_trend": trend})


@app.route("/api/anomalies/<symbol>")
@login_required
def api_anomalies(symbol):
    if not _cache_ready():
        return jsonify({"error": "Data loading..."}), 503
    from src.anomaly_detection import analyze_anomalies
    df = _data_cache["stock_df"]
    sym_df = df[df["Symbol"] == symbol].copy()
    if sym_df.empty:
        return jsonify({"error": f"Symbol {symbol} not found"}), 404
    result = analyze_anomalies(sym_df, symbol)
    return jsonify(result)


@app.route("/api/anomalies/market/report")
@login_required
def api_market_anomalies():
    if not _cache_ready():
        return jsonify({"error": "Data loading..."}), 503
    from src.anomaly_detection import get_market_anomaly_report
    result = get_market_anomaly_report(_data_cache["stock_df"])
    return jsonify(result)


@app.route("/api/portfolio")
@login_required
def api_portfolio():
    if not _cache_ready():
        return jsonify({"error": "Data loading..."}), 503
    risk = request.args.get("risk", "moderate")
    from src.portfolio_recommendation import build_portfolio
    result = build_portfolio(
        _data_cache["stock_df"],
        _data_cache["fund_df"],
        _data_cache["news_df"],
        risk_profile=risk,
    )
    return jsonify(result)


@app.route("/api/chat", methods=["POST"])
@login_required
def api_chat():
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "No message provided"}), 400
    user_msg = data["message"].strip()
    if not user_msg:
        return jsonify({"error": "Empty message"}), 400
    response = _chatbot.respond(user_msg)
    return jsonify({"response": response, "intent": _chatbot.conversation_history[-2]["intent"] if len(_chatbot.conversation_history) >= 2 else "unknown"})


@app.route("/api/market_summary")
@login_required
def api_market_summary():
    if not _cache_ready():
        return jsonify({"error": "Data loading..."}), 503
    return jsonify(_data_cache["market_summary"])


# ─── Error Handlers ───────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return render_template("login.html"), 404


@app.errorhandler(500)
def server_error(e):
    logger.error(f"Server error: {e}")
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)


