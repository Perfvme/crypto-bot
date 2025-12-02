import httpx
import pandas as pd
import pandas_ta as ta
import asyncio
import logging
import numpy as np
from scipy.signal import argrelextrema

logger = logging.getLogger(__name__)

class MarketData:
    # Spot Endpoints
    SPOT_ENDPOINTS = [
        "https://api.binance.com/api/v3",
        "https://api1.binance.com/api/v3",
        "https://api2.binance.com/api/v3",
        "https://api3.binance.com/api/v3",
    ]
    
    # Futures Endpoints
    FUTURES_ENDPOINTS = [
        "https://fapi.binance.com/fapi/v1",
        "https://fapi.binance.com/fapi/v1", 
    ]

    TIMEOUT = 10

    def __init__(self):
        self.base_url = None
        self.futures_base_url = None

    async def _request(self, endpoint_path: str, params: dict, is_futures: bool = False) -> dict:
        """Make an async request with fallback endpoints."""
        endpoints = self.FUTURES_ENDPOINTS.copy() if is_futures else self.SPOT_ENDPOINTS.copy()
        
        # Prioritize last working URL
        current_base = self.futures_base_url if is_futures else self.base_url
        if current_base:
            if current_base in endpoints:
                endpoints.remove(current_base)
            endpoints.insert(0, current_base)

        last_error = None
        async with httpx.AsyncClient(timeout=self.TIMEOUT) as client:
            for base_url in endpoints:
                try:
                    url = f"{base_url}{endpoint_path}"
                    response = await client.get(url, params=params)
                    response.raise_for_status()
                    
                    if is_futures:
                        self.futures_base_url = base_url
                    else:
                        self.base_url = base_url
                        
                    return response.json()
                except Exception as e:
                    last_error = e
                    continue
        
        raise last_error or Exception(f"All endpoints failed (Futures={is_futures})")

    async def _get_klines(self, symbol: str, interval: str, limit: int = 100) -> pd.DataFrame:
        """Fetch OHLCV data from Binance Spot."""
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        data = await self._request("/klines", params, is_futures=False)
        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore"
        ])
        
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
            
        return df

    async def _get_btc_context(self) -> dict:
        """Fetch BTC daily trend for correlation context."""
        try:
            # Fetch just enough data for a quick SMA check
            df = await self._get_klines("BTCUSDT", "1d", limit=21)
            if df.empty: return None
            
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            # Simple Trend (Price vs SMA 20)
            sma20 = df['close'].rolling(20).mean().iloc[-1]
            trend = "Bullish" if latest['close'] > sma20 else "Bearish"
            
            change_24h = ((latest['close'] - prev['close']) / prev['close']) * 100
            
            return {
                "trend": trend,
                "change_24h": round(change_24h, 2),
                "price": latest['close']
            }
        except Exception:
            return None

    async def get_futures_data(self, symbol: str) -> dict:
        """
        Fetch Open Interest and Funding Rate from Binance Futures.
        """
        try:
            # 1. Funding Rate & Mark Price
            # /premiumIndex returns markPrice, indexPrice, lastFundingRate
            funding_data = await self._request("/premiumIndex", {"symbol": symbol}, is_futures=True)
            
            # 2. Open Interest
            oi_data = await self._request("/openInterest", {"symbol": symbol}, is_futures=True)
            
            # Safe extraction
            funding_rate = float(funding_data.get("lastFundingRate", 0))
            open_interest = float(oi_data.get("openInterest", 0))
            
            return {
                "funding_rate": funding_rate,
                "funding_rate_pct": round(funding_rate * 100, 4),  # e.g., 0.01%
                "open_interest": open_interest,
                "mark_price": float(funding_data.get("markPrice", 0))
            }
        except Exception as e:
            logger.error(f"Error fetching futures data: {e}")
            return None

    async def get_order_book_imbalance(self, symbol: str) -> dict:
        """Fetch current order book depth (Spot)."""
        try:
            data = await self._request("/depth", {"symbol": symbol, "limit": 20}, is_futures=False)
            bids = data.get("bids", [])
            asks = data.get("asks", [])
            
            bid_vol = sum(float(b[1]) for b in bids)
            ask_vol = sum(float(a[1]) for a in asks)
            
            if ask_vol == 0:
                ratio = 999.0
            else:
                ratio = bid_vol / ask_vol

            return {
                "bid_vol": round(bid_vol, 2),
                "ask_vol": round(ask_vol, 2),
                "ratio": round(ratio, 2),
                "interpretation": "Bullish" if ratio > 1.2 else "Bearish" if ratio < 0.8 else "Neutral"
            }
        except Exception as e:
            logger.error(f"Error fetching order book: {e}")
            return None

    def calculate_pivots(self, df_daily: pd.DataFrame) -> dict:
        """Calculate Standard Pivot Points."""
        try:
            last_close = df_daily.iloc[-2]
            high = last_close["high"]
            low = last_close["low"]
            close = last_close["close"]
            
            pivot = (high + low + close) / 3
            r1 = (2 * pivot) - low
            s1 = (2 * pivot) - high
            r2 = pivot + (high - low)
            s2 = pivot - (high - low)
            
            return {
                "pivot": round(pivot, 4),
                "r1": round(r1, 4),
                "s1": round(s1, 4),
                "r2": round(r2, 4),
                "s2": round(s2, 4)
            }
        except Exception:
            return None

    # --- NEW: Advanced Technical Detection ---

    def detect_divergences(self, df: pd.DataFrame, lookback=20) -> list:
        """
        Detect Regular Bullish/Bearish Divergences between Price and RSI.
        Returns a list of strings describing detected divergences.
        """
        divergences = []
        try:
            if len(df) < lookback:
                return []

            # Get local peaks (max) and valleys (min)
            # order=5 means comparison with 5 neighbors on each side
            high_idx = argrelextrema(df['high'].values, np.greater, order=5)[0]
            low_idx = argrelextrema(df['low'].values, np.less, order=5)[0]
            
            # We need at least two peaks/valleys to compare
            if len(high_idx) >= 2:
                last_peak_idx = high_idx[-1]
                prev_peak_idx = high_idx[-2]
                
                # Check Bearish Divergence: Price Higher High, RSI Lower High
                price_hh = df['high'].iloc[last_peak_idx] > df['high'].iloc[prev_peak_idx]
                rsi_lh = df['rsi'].iloc[last_peak_idx] < df['rsi'].iloc[prev_peak_idx]
                
                # Ensure the "last peak" is relatively recent (within last 5-8 candles)
                if price_hh and rsi_lh and (len(df) - last_peak_idx) <= 8:
                    divergences.append("Bearish Divergence (Price HH, RSI LH)")

            if len(low_idx) >= 2:
                last_valley_idx = low_idx[-1]
                prev_valley_idx = low_idx[-2]
                
                # Check Bullish Divergence: Price Lower Low, RSI Higher Low
                price_ll = df['low'].iloc[last_valley_idx] < df['low'].iloc[prev_valley_idx]
                rsi_hl = df['rsi'].iloc[last_valley_idx] > df['rsi'].iloc[prev_valley_idx]
                
                if price_ll and rsi_hl and (len(df) - last_valley_idx) <= 8:
                    divergences.append("Bullish Divergence (Price LL, RSI HL)")

        except Exception as e:
            logger.error(f"Divergence check failed: {e}")
            
        return divergences

    def detect_fvgs(self, df: pd.DataFrame) -> list:
        """
        Identify unmitigated Fair Value Gaps (FVG) in the last few candles.
        Returns list of FVGs: {'type': 'bull/bear', 'top': float, 'bottom': float}
        """
        fvgs = []
        try:
            # Look at last 5 completed candles (excluding current)
            # Candle index: 0, 1, 2. FVG forms between 0 and 2.
            # We iterate backwards
            for i in range(len(df) - 2, len(df) - 7, -1):
                curr = df.iloc[i]     # Candle C (Recent)
                # prev = df.iloc[i-1] # Candle B
                prev_2 = df.iloc[i-2] # Candle A (Oldest)
                
                # Bullish FVG: High of A < Low of C
                if prev_2['high'] < curr['low']:
                    fvgs.append({
                        "type": "Bullish FVG",
                        "price_range": f"{prev_2['high']} - {curr['low']}",
                        "age": f"{len(df) - i} bars ago"
                    })

                # Bearish FVG: Low of A > High of C
                if prev_2['low'] > curr['high']:
                    fvgs.append({
                        "type": "Bearish FVG",
                        "price_range": f"{curr['high']} - {prev_2['low']}",
                        "age": f"{len(df) - i} bars ago"
                    })
                    
        except Exception as e:
            pass
        return fvgs[:3] # Return top 3 most recent

    async def get_market_analysis(self, symbol: str) -> dict:
        timeframes = ["1d", "4h", "15m"]
        
        # 1. Fetch Basic Data
        tasks = [self._get_klines(symbol, tf, limit=200) for tf in timeframes]
        task_ob = self.get_order_book_imbalance(symbol)
        task_futures = self.get_futures_data(symbol)

        # --- NEW: BTC Context ---
        # If we are analyzing BTC, we don't need to fetch it again as context
        is_btc = "BTC" in symbol.upper()
        task_btc = asyncio.sleep(0, result=None) if is_btc else self._get_btc_context()

        results = await asyncio.gather(*tasks, task_ob, task_futures, task_btc, return_exceptions=True)
        
        kline_results = results[:3]
        order_book = results[3]
        futures_data = results[4]
        btc_context = results[5]
        
        analysis = {
            "symbol": symbol,
            "order_book": order_book if not isinstance(order_book, Exception) else "Error",
            "futures_data": futures_data if not isinstance(futures_data, Exception) else "Unavailable",
            "btc_context": btc_context if not isinstance(btc_context, Exception) else None,
            "timeframes": {}
        }

        trend_signals = {}
        atr_values = {}

        for tf, result in zip(timeframes, kline_results):
            if isinstance(result, Exception):
                analysis["timeframes"][tf] = {"error": str(result)}
                continue
                
            df = result
            
            # Indicators
            df["rsi"] = ta.rsi(df["close"], length=14)
            macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
            df = pd.concat([df, macd], axis=1)
            df["ema_50"] = ta.ema(df["close"], length=50)
            df["ema_200"] = ta.ema(df["close"], length=200)
            df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
            adx_df = ta.adx(df["high"], df["low"], df["close"], length=14)
            df = pd.concat([df, adx_df], axis=1)

            # --- NEW: Volume Analysis ---
            # Simple RVOL (Relative Volume) vs 20 SMA
            vol_sma = df['volume'].rolling(window=20).mean()
            df['rvol'] = df['volume'] / vol_sma

            # Advanced Detection
            divergences = self.detect_divergences(df)
            fvgs = self.detect_fvgs(df)
            
            # --- NEW: Trend Alignment ---
            trend_state = self.calculate_trend_alignment(df)

            latest = df.iloc[-1]
            
            # Capture signals for Trend Matrix & ATR
            if tf == "1d":
                trend_signals["daily_ema200"] = latest["ema_200"] if pd.notnull(latest["ema_200"]) else None
                trend_signals["daily_close"] = latest["close"]
            elif tf == "4h":
                # MACD Histogram > 0? (MACD line - Signal line)
                # pandas_ta names: MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
                hist_col = [c for c in df.columns if c.startswith("MACDh")][0]
                trend_signals["4h_macd_hist"] = latest[hist_col]
                atr_values["4h"] = latest["atr"]
            elif tf == "15m":
                trend_signals["15m_ema50"] = latest["ema_50"] if pd.notnull(latest["ema_50"]) else None
                trend_signals["15m_close"] = latest["close"]
                atr_values["15m"] = latest["atr"]

            tf_data = {
                "close": latest["close"],
                "rsi": round(latest["rsi"], 2) if pd.notnull(latest["rsi"]) else None,
                "adx": round(latest["ADX_14"], 2) if "ADX_14" in latest else None,
                "ema_50": round(latest["ema_50"], 2) if pd.notnull(latest["ema_50"]) else None,
                "ema_200": round(latest["ema_200"], 2) if pd.notnull(latest["ema_200"]) else None,
                "atr": round(latest["atr"], 4) if pd.notnull(latest["atr"]) else None,
                "rvol": round(latest['rvol'], 2) if pd.notnull(latest['rvol']) else 1.0, # NEW
                "trend": trend_state,
                "divergences": divergences,
                "fvgs": fvgs
            }
            
            if tf == "1d":
                pivots = self.calculate_pivots(df)
                tf_data["pivots"] = pivots
                # --- NEW: S/R Clusters (Only calculated on Daily for major levels) ---
                tf_data["sr_clusters"] = self.detect_support_resistance_clusters(df, pivots)

            analysis["timeframes"][tf] = tf_data

        # --- NEW: Post-Processing (Trend Matrix & ATR Stops) ---
        analysis["trend_matrix"] = self._calculate_trend_matrix(trend_signals)
        analysis["risk_data"] = self._calculate_risk_levels(trend_signals.get("15m_close"), atr_values)

        return analysis

    def _calculate_trend_matrix(self, signals: dict) -> dict:
        """
        Generates a multi-timeframe trend score (-3 to +3).
        """
        score = 0
        details = {}

        # 1. Daily Trend (Price vs EMA 200)
        if signals.get("daily_ema200") and signals.get("daily_close"):
            if signals["daily_close"] > signals["daily_ema200"]:
                score += 1
                details["daily"] = "Bullish"
            else:
                score -= 1
                details["daily"] = "Bearish"
        else:
            details["daily"] = "Neutral"

        # 2. 4H Momentum (MACD Hist)
        if signals.get("4h_macd_hist") is not None:
            if signals["4h_macd_hist"] > 0:
                score += 1
                details["4h"] = "Bullish"
            else:
                score -= 1
                details["4h"] = "Bearish"
        else:
            details["4h"] = "Neutral"

        # 3. 15m Trend (Price vs EMA 50)
        if signals.get("15m_ema50") and signals.get("15m_close"):
            if signals["15m_close"] > signals["15m_ema50"]:
                score += 1
                details["15m"] = "Bullish"
            else:
                score -= 1
                details["15m"] = "Bearish"
        else:
            details["15m"] = "Neutral"

        return {
            "total_score": score,
            "max_score": 3,
            "interpretation": "Strong Buy" if score == 3 else "Strong Sell" if score == -3 else "Neutral/Mixed",
            "details": details
        }

    def _calculate_risk_levels(self, current_price: float, atr_values: dict) -> dict:
        """
        Calculates suggested SL levels based on ATR.
        """
        if not current_price: return {}

        risk = {}
        
        # Scalp (15m ATR)
        if atr_values.get("15m"):
            atr = atr_values["15m"]
            risk["scalp"] = {
                "long_sl": round(current_price - (1.5 * atr), 2),
                "short_sl": round(current_price + (1.5 * atr), 2),
                "atr_value": round(atr, 2)
            }
            
        # Swing (4h ATR)
        if atr_values.get("4h"):
            atr = atr_values["4h"]
            risk["swing"] = {
                "long_sl": round(current_price - (2.0 * atr), 2),
                "short_sl": round(current_price + (2.0 * atr), 2),
                "atr_value": round(atr, 2)
            }
            
        return risk

    def calculate_trend_alignment(self, df: pd.DataFrame) -> str:
        """
        Determine the trend state based on EMA 50/200 relationship and Price.
        """
        try:
            if len(df) < 200: return "Insufficient Data"
            
            curr = df.iloc[-1]
            price = curr['close']
            ema50 = curr['ema_50']
            ema200 = curr['ema_200']
            
            if ema50 > ema200:
                if price > ema50: return "Strong Bullish (Uptrend)"
                elif price > ema200: return "Weak Bullish (Pullback)"
                else: return "Bullish Trend Threatened"
            else:
                if price < ema50: return "Strong Bearish (Downtrend)"
                elif price < ema200: return "Weak Bearish (Rally)"
                else: return "Bearish Trend Threatened"
        except Exception:
            return "Unknown"

    def detect_support_resistance_clusters(self, df: pd.DataFrame, pivots: dict, tolerance_pct=0.01) -> list:
        """
        Identify Key Zones where multiple levels (Pivots, EMAs, Swings) overlap.
        """
        clusters = []
        try:
            current_price = df.iloc[-1]['close']
            levels = []
            
            # 1. Add Pivots
            if pivots:
                for name, val in pivots.items():
                    levels.append({"price": val, "type": f"Pivot {name.upper()}"})
            
            # 2. Add EMAs (Daily)
            curr = df.iloc[-1]
            if pd.notnull(curr['ema_50']):
                levels.append({"price": curr['ema_50'], "type": "Daily EMA 50"})
            if pd.notnull(curr['ema_200']):
                levels.append({"price": curr['ema_200'], "type": "Daily EMA 200"})
                
            # 3. Add Recent Swing Highs/Lows (Last 90 days)
            # Using local minima/maxima
            highs = argrelextrema(df['high'].values[-90:], np.greater, order=10)[0]
            lows = argrelextrema(df['low'].values[-90:], np.less, order=10)[0]
            
            # Map indices back to prices (offset by len(df)-90)
            offset = len(df) - 90
            for idx in highs:
                val = df.iloc[offset + idx]['high']
                levels.append({"price": val, "type": "Swing High"})
            for idx in lows:
                val = df.iloc[offset + idx]['low']
                levels.append({"price": val, "type": "Swing Low"})

            # 4. Cluster Algorithm
            # Sort by price
            levels.sort(key=lambda x: x['price'])
            
            if not levels: return []

            current_cluster = [levels[0]]
            
            for i in range(1, len(levels)):
                prev = current_cluster[-1]
                curr = levels[i]
                
                # Check % distance
                diff = (curr['price'] - prev['price']) / prev['price']
                
                if diff <= tolerance_pct:
                    current_cluster.append(curr)
                else:
                    # Process completed cluster
                    if len(current_cluster) >= 2:
                        # Only keep clusters relatively close to current price (+/- 15%)
                        avg_price = sum(x['price'] for x in current_cluster) / len(current_cluster)
                        if 0.85 * current_price <= avg_price <= 1.15 * current_price:
                            clusters.append({
                                "avg_price": round(avg_price, 2),
                                "strength": len(current_cluster),
                                "confluence": ", ".join([x['type'] for x in current_cluster])
                            })
                    current_cluster = [curr]
            
            # Check last cluster
            if len(current_cluster) >= 2:
                avg_price = sum(x['price'] for x in current_cluster) / len(current_cluster)
                if 0.85 * current_price <= avg_price <= 1.15 * current_price:
                    clusters.append({
                        "avg_price": round(avg_price, 2),
                        "strength": len(current_cluster),
                        "confluence": ", ".join([x['type'] for x in current_cluster])
                    })

            # Sort clusters by strength (descending)
            clusters.sort(key=lambda x: x['strength'], reverse=True)
            
        except Exception as e:
            logger.error(f"Cluster detection failed: {e}")
            
        return clusters[:3] # Return top 3 strongest zones