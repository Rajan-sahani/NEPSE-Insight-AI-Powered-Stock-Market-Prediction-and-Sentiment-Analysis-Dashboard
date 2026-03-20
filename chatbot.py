
import re
from logger import logging
import random
from typing import Dict, List, Optional
import pandas as pd

logger = logging.getLogger(__name__)

# ─── Intent Patterns ─────────────────────────────────────────────────────────────
INTENTS = {
    "greeting":       [r"\b(hello|hi|hey|namaste|good\s*(morning|evening|afternoon))\b"],
    "farewell":       [r"\b(bye|goodbye|quit|exit|thanks|thank you)\b"],
    "help":           [r"\b(help|what can you do|commands|options|how to)\b"],
    "market_trend":   [r"\b(nepse|market|index|trend|today|overall)\b"],
    "best_stock":     [r"\b(best stock|top stock|buy|recommend|invest|which stock)\b"],
    "sector_query":   [r"\b(banking|hydropower|insurance|finance|telecom|manufacturing|sector)\b"],
    "stock_price":    [r"\b(price|close|current|how much|value)\b.*\b[A-Z]{2,5}\b", r"\b[A-Z]{3,5}\b.*\b(price|close|current)\b"],
    "prediction":     [r"\b(predict|forecast|future|next|tomorrow|5 day|next week)\b"],
    "sentiment":      [r"\b(sentiment|mood|feeling|news|social|positive|negative|bullish|bearish)\b"],
    "portfolio":      [r"\b(portfolio|diversify|allocation|risk|aggressive|conservative|moderate)\b"],
    "anomaly":        [r"\b(anomaly|unusual|spike|crash|alert|irregular|suspicious)\b"],
    "pe_ratio":       [r"\b(pe ratio|p/e|valuation|cheap|expensive|overvalued|undervalued)\b"],
    "volume":         [r"\b(volume|turnover|traded|liquidity)\b"],
}

# ─── Response Templates ──────────────────────────────────────────────────────────
RESPONSES = {
    "greeting": [
        "Namaste! 🙏 I'm your NEPSE AI assistant. Ask me about stock prices, market trends, predictions, or portfolio recommendations!",
        "Hello! Welcome to NEPSE AI Analytics. How can I help you navigate Nepal's stock market today?",
    ],
    "farewell": [
        "Goodbye! Happy investing! 📈",
        "Thank you for using NEPSE AI Analytics. See you next trading session!",
    ],
    "help": """**What I can help you with:**
• 📊 **Market Trends** — "What's the NEPSE trend today?"
• 🏆 **Best Stocks** — "Which stocks should I buy?"
• 🏦 **Sector Analysis** — "How is the banking sector performing?"
• 🔮 **Predictions** — "Predict NABIL stock price"
• 💬 **Sentiment** — "What is market sentiment?"
• 💼 **Portfolio** — "Recommend a conservative portfolio"
• ⚠️ **Anomalies** — "Any anomalies detected?"
• 📉 **Valuations** — "Which stocks are undervalued?"
""",
    "unknown": [
        "I'm not sure about that. Try asking about stock prices, market trends, or portfolio recommendations!",
        "Hmm, I didn't catch that. You can ask me about NEPSE trends, stock predictions, or sector analysis.",
    ],
}


# ─── Core Bot Logic ──────────────────────────────────────────────────────────────

class NEPSEChatbot:
    def __init__(self):
        self.stock_df: Optional[pd.DataFrame] = None
        self.sentiment_summary: Optional[Dict] = None
        self.portfolio_result: Optional[Dict] = None
        self.market_summary: Optional[Dict] = None
        self.conversation_history: List[Dict] = []

    def load_context(self, stock_df=None, sentiment_summary=None, portfolio_result=None, market_summary=None):
        """Load live data context for dynamic responses."""
        if stock_df is not None:
            self.stock_df = stock_df
        if sentiment_summary is not None:
            self.sentiment_summary = sentiment_summary
        if portfolio_result is not None:
            self.portfolio_result = portfolio_result
        if market_summary is not None:
            self.market_summary = market_summary

    def detect_intent(self, text: str) -> str:
        text_lower = text.lower()
        for intent, patterns in INTENTS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    return intent
        return "unknown"

    def extract_symbol(self, text: str) -> Optional[str]:
        """Extract stock symbol (2-5 uppercase letters) from text."""
        matches = re.findall(r"\b([A-Z]{2,5})\b", text.upper())
        if self.stock_df is not None:
            valid_symbols = set(self.stock_df["Symbol"].unique())
            for m in matches:
                if m in valid_symbols:
                    return m
        return matches[0] if matches else None

    def respond(self, user_input: str) -> str:
        intent = self.detect_intent(user_input)
        symbol = self.extract_symbol(user_input)
        self.conversation_history.append({"role": "user", "text": user_input, "intent": intent})

        response = self._generate_response(intent, user_input, symbol)
        self.conversation_history.append({"role": "bot", "text": response})
        return response

    def _generate_response(self, intent: str, text: str, symbol: Optional[str]) -> str:
        if intent == "greeting":
            return random.choice(RESPONSES["greeting"])

        if intent == "farewell":
            return random.choice(RESPONSES["farewell"])

        if intent == "help":
            return RESPONSES["help"]

        if intent == "market_trend":
            return self._market_trend_response()

        if intent == "best_stock":
            return self._best_stock_response()

        if intent == "sector_query":
            return self._sector_response(text)

        if intent == "stock_price" and symbol:
            return self._stock_price_response(symbol)

        if intent == "prediction":
            return self._prediction_response(symbol, text)

        if intent == "sentiment":
            return self._sentiment_response()

        if intent == "portfolio":
            return self._portfolio_response(text)

        if intent == "anomaly":
            return self._anomaly_response()

        if intent == "pe_ratio":
            return self._valuation_response()

        if intent == "volume":
            return self._volume_response()

        return random.choice(RESPONSES["unknown"])

    # ── Contextual Response Builders ─────────────────────────────────────────────

    def _market_trend_response(self) -> str:
        if self.market_summary:
            ms = self.market_summary
            mood = self.sentiment_summary.get("market_mood", "Neutral") if self.sentiment_summary else "Neutral"
            adv = ms.get("advances", "N/A")
            dec = ms.get("declines", "N/A")
            turnover = ms.get("total_turnover", 0)
            gainers = ms.get("top_gainers", [])
            top_gainer = f"{gainers[0]['Symbol']} (+{gainers[0]['per_change']:.1f}%)" if gainers else "N/A"
            return (
                f"**NEPSE Market Summary** (as of {ms.get('latest_date', 'latest')})\n\n"
                f"📈 **Advances:** {adv} | 📉 **Declines:** {dec}\n"
                f"💰 **Turnover:** NPR {turnover:,.0f}\n"
                f"🌡️ **Market Mood:** {mood}\n"
                f"🏆 **Top Gainer:** {top_gainer}\n\n"
                f"The market is showing a **{mood.lower()}** sentiment with "
                f"{'more advancers than decliners' if adv > dec else 'more decliners than advancers'}."
            )
        return "NEPSE market data shows mixed signals. Check the dashboard for real-time charts and trends."

    def _best_stock_response(self) -> str:
        if self.portfolio_result:
            portfolio = self.portfolio_result.get("portfolio", [])[:5]
            if portfolio:
                lines = ["**🏆 Top Investment Picks (Moderate Risk):**\n"]
                for i, s in enumerate(portfolio, 1):
                    lines.append(f"{i}. **{s['symbol']}** ({s['sector']}) — Score: {s['composite_score']:.2f} | Rating: {s['rating']} | Target: NPR {s['target_price']}")
                lines.append("\n_Based on fundamentals, technicals, and market sentiment._")
                return "\n".join(lines)
        return (
            "Based on NEPSE analysis, focus on fundamentally strong stocks with good EPS and low P/E ratios. "
            "Banking and hydropower sectors typically offer stability. Use the Portfolio tab for personalized recommendations."
        )

    def _sector_response(self, text: str) -> str:
        text_lower = text.lower()
        sector_map = {
            "banking": ("Banking", "🏦", "Banks drive NEPSE with strong deposit growth and NPA management. Look for EPS > 40."),
            "hydropower": ("Hydropower", "⚡", "Hydropower is Nepal's future. Projects commissioning in coming quarters = price catalyst."),
            "insurance": ("Insurance", "🛡️", "Insurance sector benefits from growing premium income and low claim ratios."),
            "finance": ("Finance", "💹", "Finance companies offer high dividends but carry higher NPA risk."),
            "telecom": ("Telecom", "📡", "NTC dominates with stable dividends. Watch for 5G rollout developments."),
            "manufacturing": ("Manufacturing", "🏭", "Cement and sugar stocks tied to infrastructure boom. Seasonal demand patterns."),
        }
        for key, (sector, icon, desc) in sector_map.items():
            if key in text_lower:
                sentiment_info = ""
                if self.sentiment_summary:
                    for ss in self.sentiment_summary.get("sector_sentiment", []):
                        if ss["sector"] == sector:
                            sent = ss["avg_sentiment"]
                            sentiment_info = f"\n📊 Current sentiment: {'Positive' if sent > 0.05 else 'Negative' if sent < -0.05 else 'Neutral'} ({sent:+.3f})"
                return f"**{icon} {sector} Sector Analysis**\n\n{desc}{sentiment_info}"
        return "Which sector would you like to know about? Banking, Hydropower, Insurance, Finance, Telecom, or Manufacturing?"

    def _stock_price_response(self, symbol: str) -> str:
        if self.stock_df is not None:
            sym_df = self.stock_df[self.stock_df["Symbol"] == symbol]
            if not sym_df.empty:
                last = sym_df.sort_values("Date").iloc[-1]
                price = last["close"]
                change = last.get("per_change", 0) or 0
                rsi = last.get("rsi", None)
                arrow = "▲" if change > 0 else "▼" if change < 0 else "—"
                rsi_note = f" | RSI: {rsi:.1f}" if pd.notna(rsi) else ""
                signal = "Overbought ⚠️" if rsi and rsi > 70 else "Oversold 💡" if rsi and rsi < 30 else "Normal"
                return (
                    f"**{symbol} Stock Info**\n\n"
                    f"💰 **Current Price:** NPR {price:,.2f}\n"
                    f"📊 **Daily Change:** {arrow} {abs(change):.2f}%\n"
                    f"📈 **RSI (14):** {rsi:.1f} — {signal}{rsi_note}\n"
                    f"📅 **Date:** {str(last['Date'].date())}"
                )
        return f"I couldn't find price data for **{symbol}**. Please verify the symbol and try again."

    def _prediction_response(self, symbol: Optional[str], text: str) -> str:
        if symbol:
            return (
                f"To get a 5-day ML prediction for **{symbol}**, use the **Stock Prediction** tab on the dashboard. "
                f"Our ensemble model (Random Forest + Gradient Boosting) will predict the next 5 trading days with confidence intervals."
            )
        return (
            "Our prediction engine uses Random Forest + Gradient Boosting ensemble models trained on historical OHLCV data. "
            "Select a stock in the **Prediction** tab for 5-day forecasts with confidence bands."
        )

    def _sentiment_response(self) -> str:
        if self.sentiment_summary:
            ss = self.sentiment_summary
            mood = ss.get("market_mood", "Neutral")
            score = ss.get("blended_score", 0)
            news_pos = ss.get("news_distribution", {}).get("Positive", 0)
            news_neg = ss.get("news_distribution", {}).get("Negative", 0)
            return (
                f"**💬 NEPSE Market Sentiment Analysis**\n\n"
                f"🌡️ **Market Mood:** {mood} (score: {score:+.3f})\n"
                f"📰 **News Sentiment:** {news_pos:.0f}% Positive | {news_neg:.0f}% Negative\n\n"
                f"Sentiment is {'cautiously optimistic' if mood == 'Bullish' else 'showing concern' if mood == 'Bearish' else 'mixed'}. "
                f"Check the Sentiment tab for sector-wise breakdown and daily trends."
            )
        return "Sentiment analysis combines financial news and social media signals. Currently showing mixed market mood. Check the Sentiment tab for details."

    def _portfolio_response(self, text: str) -> str:
        text_lower = text.lower()
        if "conservative" in text_lower:
            return "**Conservative Portfolio:** Focus on banking (EPS>40), established hydropower, insurance. Target 8-12% annual return with low volatility. Use 60% large-cap, 40% stable dividend stocks."
        elif "aggressive" in text_lower:
            return "**Aggressive Portfolio:** Micro-cap hydropower, newly-listed stocks, high-beta banking. Target 25-40% return potential with higher volatility. Limit to 20-30% of total capital."
        else:
            return "**Moderate Portfolio (Recommended):** 40% Banking, 30% Hydropower, 15% Insurance, 15% Others. Rebalance quarterly. Use the Portfolio tab for current AI-ranked recommendations."

    def _anomaly_response(self) -> str:
        return (
            "**⚠️ Anomaly Detection:**\n\n"
            "Our Isolation Forest + Z-score engine monitors all NEPSE stocks for:\n"
            "• Unusual price spikes or crashes (>8% daily move)\n"
            "• Abnormal trading volume surges\n"
            "• Multi-dimensional statistical outliers\n\n"
            "Check the **Anomaly Detection** tab for real-time alerts and severity classifications."
        )

    def _valuation_response(self) -> str:
        if self.portfolio_result:
            recs = self.portfolio_result.get("recommendations", [])
            low_pe = sorted([r for r in recs if r.get("pe_ratio", 0) > 0], key=lambda x: x["pe_ratio"])[:3]
            if low_pe:
                lines = ["**💡 Potentially Undervalued Stocks (Low P/E):**\n"]
                for s in low_pe:
                    lines.append(f"• **{s['symbol']}** — P/E: {s['pe_ratio']:.1f} | EPS: {s['eps']:.2f}")
                return "\n".join(lines)
        return "Generally, NEPSE banking stocks with P/E < 12 are considered undervalued. Hydropower development-stage stocks may trade at premium P/E due to growth expectations."

    def _volume_response(self) -> str:
        if self.market_summary:
            turnover = self.market_summary.get("total_turnover", 0)
            return f"**📊 Market Volume:**\nTotal turnover: NPR {turnover:,.0f}\nHigh-volume stocks often signal institutional activity. Check the dashboard for per-stock volume charts."
        return "Volume analysis shows stocks with unusual traded quantities. High volume with price increase = bullish confirmation. High volume with price decrease = potential reversal."
