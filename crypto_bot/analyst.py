import os
import json
import logging
import httpx
import asyncio
import xml.etree.ElementTree as ET
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class AIAnalyst:
    RSS_FEEDS = [
        "https://cointelegraph.com/rss",
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://bitcoinmagazine.com/.rss/full/"
    ]

    def __init__(self):
        self.api_key = os.getenv("ZAI_API_KEY")
        if not self.api_key:
            raise ValueError("ZAI_API_KEY not found in environment variables")
        # Use AsyncOpenAI SDK
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url="https://api.z.ai/api/coding/paas/v4/"
        )

    async def _fetch_rss_news(self) -> str:
        """
        Fallback: Fetch news from RSS feeds if API fails.
        """
        news_items = []
        async with httpx.AsyncClient(timeout=10) as client:
            for feed in self.RSS_FEEDS:
                try:
                    response = await client.get(feed)
                    if response.status_code == 200:
                        # Simple XML parsing
                        root = ET.fromstring(response.content)
                        # Standard RSS 2.0 structure: channel -> item -> title
                        count = 0
                        for item in root.findall("./channel/item"):
                            title = item.find("title").text
                            pubDate = item.find("pubDate").text
                            # Basic filtering
                            if title and len(title) > 20:
                                news_items.append(f"- {title} ({pubDate[:16]})")
                                count += 1
                            if count >= 3: break # Max 3 per feed
                except Exception:
                    continue
                
                if len(news_items) >= 5: break # Stop if we have enough
        
        if not news_items:
            return "No news available from RSS backup."
            
        return "Latest Headlines (RSS):\n" + "\n".join(news_items[:8])

    async def fetch_crypto_news(self, symbol: str) -> str:
        """
        Fetch news from CryptoPanic (Primary) -> RSS Feeds (Fallback).
        """
        # 1. Try CryptoPanic
        try:
            base_currency = symbol.replace("USDT", "").replace("USD", "").replace("BUSD", "")
            url = f"https://cryptopanic.com/api/free/v1/posts/"
            params = {"currencies": base_currency, "kind": "news", "public": "true"}
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(url, params=params, headers=headers)
                
                if response.status_code == 200:
                    data = response.json()
                    news_items = data.get("results", [])[:5]
                    if news_items:
                        news_text = f"News for {base_currency}:\n"
                        for i, item in enumerate(news_items, 1):
                            title = item.get("title", "No title")
                            source = item.get("source", {}).get("title", "Unknown")
                            news_text += f"{i}. {title} (Source: {source})\n"
                        
                        # Truncate to avoid token overflow
                        return news_text[:1500]
        except Exception:
            pass # Fallthrough to backup

        # 2. Fallback to RSS
        rss_news = await self._fetch_rss_news()
        return rss_news[:1500]

    async def analyze_market(self, market_data: dict, style: str = "intraday"):
        """
        Sends market data to Z.AI GLM-4.5 (Async) with Advanced Deep Reasoning.
        Style: "scalp" | "swing" | "intraday"
        """
        
        symbol = market_data.get('symbol', 'BTC')
        
        # Fetch news concurrently
        crypto_news = await self.fetch_crypto_news(symbol)

        # --- PERSONA DEFINITIONS ---
        personas = {
            "scalp": """
                ROLE: You are an aggressive High-Frequency Scalper. 
                FOCUS: Order Book imbalances, 15m FVGs, Futures OI spikes, and quick momentum. 
                MINDSET: Get in, grab profits (0.5% - 1.5%), get out. Tight invalidation.
                IGNORE: Long-term macro fundamentals (unless breaking news).
            """,
            "swing": """
                ROLE: You are a patient Institutional Swing Trader. 
                FOCUS: Daily Trend, 4H Divergences, Macro Sentiment, and Risk/Reward ratios > 1:3.
                MINDSET: Capture the main leg of the trend. Wide stops based on ATR.
                IGNORE: 15m intraday noise.
            """,
            "intraday": """
                ROLE: You are an elite Day Trader using Smart Money Concepts (SMC).
                FOCUS: Balance of Daily Bias and 15m/1h execution.
                MINDSET: Trade the session liquidity.
            """
        }
        
        selected_persona = personas.get(style, personas["intraday"])
        
        # --- SYSTEM PROMPT CONSTRUCTION ---
        system_prompt = f"""
{selected_persona}

Your task is to provide a high-precision trade setup for {symbol}.

**METHODOLOGY (THE DEVIL'S ADVOCATE):**
1. **Trend Check:** Respect the 'Trend Alignment' (e.g., only Long if Daily is Bullish or at Support).
2. **Bullish Case:** Analyze reasons to go LONG.
3. **Bearish Case:** Analyze reasons to go SHORT.
4. **Clusters:** Use 'SR Clusters' for precise Entry/SL levels (e.g., "Buy at 58k Cluster").
5. **Conclusion:** Weigh evidence. If conflicting, 'WAIT'.

**DATA GUIDELINES:**
- **Trend Alignment:** "Strong Bullish" = Buy Dips. "Strong Bearish" = Sell Rallies.
- **Futures:** Positive Funding (>0.01%) = Bullish but squeeze risk.
- **Order Book:** Ratio > 1.2 (Bullish).

**OUTPUT SCHEMA (JSON Only):**
{{
  "symbol": "{symbol}",
  "market_sentiment": "bullish|bearish|neutral",
  "bull_bear_case": "Concise summary of the conflict.",
  "news_summary": "Brief summary.",
  "technical_summary": "Top 3 drivers.",
  "trade_setup": {{
    "style": "{style}",
    "direction": "long|short|wait",
    "entry_zone": "price",
    "take_profit": "targets",
    "stop_loss": "level",
    "confidence_score": 0-100,
    "reasoning": "Concise logic for the decision.",
    "invalidation": "level",
    "alternative_scenario": "If invalidated..."
  }},
  "risk_disclaimer": "NFA"
}}
"""

        # 2. Construct User Message (Enriched Data)
        # Use default separators for compact JSON to save tokens
        user_content = f"""
=== DATA {symbol} ({style.upper()}) ===

[FUTURES]
{json.dumps(market_data.get('futures_data', {}))}

[ORDER BOOK]
{json.dumps(market_data.get('order_book', {}))}

[TIMEFRAMES]
1D: {json.dumps(market_data.get('timeframes', {}).get('1d', {}))}
4H: {json.dumps(market_data.get('timeframes', {}).get('4h', {}))}
15M: {json.dumps(market_data.get('timeframes', {}).get('15m', {}))}

[NEWS]
{crypto_news}

**INSTRUCTION:**
Act as {style}. Analyze. Be CONCISE. Output JSON.
"""

        # 3. Call API
        try:
            response = await self.client.chat.completions.create(
                model="glm-4.5",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.3, 
                max_tokens=4000
            )
            
            # Debug: Check if we got a valid response
            if not response.choices:
                return {"error": "No choices in AI response"}
            
            content = response.choices[0].message.content
            
            # Robust JSON Extraction
            import re
            
            # 1. Try regex for the first JSON object
            json_match = re.search(r'(\{.*\})', content, re.DOTALL)
            
            try:
                if json_match:
                    json_str = json_match.group(1)
                    return json.loads(json_str)
                
                # 2. Fallback: Explicit markdown cleanup
                cleaned_content = content
                if "```json" in cleaned_content:
                    cleaned_content = cleaned_content.split("```json")[1].split("```")[0]
                elif "```" in cleaned_content:
                    cleaned_content = cleaned_content.split("```")[1]
                
                cleaned_content = cleaned_content.strip()
                if cleaned_content.startswith("{") and cleaned_content.endswith("}"):
                     return json.loads(cleaned_content)

                # 3. Final Fallback: Wrap raw text
                logger.warning(f"No JSON found. Content preview: {content[:100]}...")
                return {
                        "symbol": symbol,
                        "market_sentiment": "neutral",
                        "bull_bear_case": "AI Response format error. See reasoning.",
                        "technical_summary": "Analysis available in reasoning.",
                        "trade_setup": {
                            "style": style,
                            "direction": "wait",
                            "confidence_score": 0,
                            "reasoning": content,  
                            "invalidation": "N/A"
                        },
                        "risk_disclaimer": "Partial parsing error. Review text carefully."
                    }

            except json.JSONDecodeError as e:
                logger.error(f"JSON Decode Error: {e}")
                # Fallback for malformed JSON
                return {
                    "symbol": symbol,
                    "market_sentiment": "neutral",
                    "bull_bear_case": "JSON Error",
                    "trade_setup": {
                        "style": style,
                        "direction": "wait",
                        "reasoning": content, # Return raw content
                        "confidence_score": 0
                    }
                }
        except Exception as e:
            return {
                "error": str(e),
                "raw_response": content if 'content' in locals() else "No response"
            }
