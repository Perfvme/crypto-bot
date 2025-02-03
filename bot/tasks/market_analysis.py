import pandas as pd
import numpy as np
from typing import Dict
from .indicators import TechnicalIndicators
from api.deepseek_client import DeepseekClient
from api.binance_client import BinanceClient

class MarketAnalyzer:
    def __init__(self):
        self.indicators = TechnicalIndicators()
        self.deepseek = DeepseekClient()
        self.binance = BinanceClient()
        
    def full_analysis(self, df: pd.DataFrame, coin: str) -> Dict:
        df = self.indicators.calculate_all(df)
        latest = df.iloc[-1]
        
        # Technical Analysis
        analysis = {
            'coin': coin,
            'price': latest['close'],
            'trend': self._get_trend_direction(df),
            'confidence': self._calculate_confidence(df),
            'support': self._find_support(df),
            'resistance': self._find_resistance(df),
            'volume_change': self._volume_change(df),
            'indicators': latest.to_dict()
        }
        
        # Add Deepseek Analysis
        ds_analysis = self.deepseek.analyze_market({
            'coin': coin,
            'indicators': analysis['indicators']
        })
        analysis.update(ds_analysis)
        
        return analysis

    def _get_trend_direction(self, df: pd.DataFrame) -> str:
        ema_status = (
            df['ema_8'].iloc[-1] > df['ema_21'].iloc[-1],
            df['ema_21'].iloc[-1] > df['ema_55'].iloc[-1]
        )
        if all(ema_status):
            return 'strong_bull'
        elif any(ema_status):
            return 'weak_bull'
        elif not any(ema_status):
            return 'strong_bear'
        return 'neutral'

    def _calculate_confidence(self, df: pd.DataFrame) -> float:
        factors = [
            df['macd_diff'].iloc[-1] * 0.3,
            df['rsi'].iloc[-1] / 100 * 0.2,
            (df['volume'].iloc[-1] / df['volume'].mean()) * 0.2,
            self._trend_consistency(df) * 0.3
        ]
        return np.clip(sum(factors), 0, 1)

    def _trend_consistency(self, df: pd.DataFrame) -> float:
        return (df['ema_8'] > df['ema_21']).rolling(5).mean().iloc[-1]

    def _find_support(self, df: pd.DataFrame) -> float:
        return df['low'].rolling(20).min().iloc[-1]

    def _find_resistance(self, df: pd.DataFrame) -> float:
        return df['high'].rolling(20).max().iloc[-1]

    def _volume_change(self, df: pd.DataFrame) -> float:
        return (df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]) - 1

    def check_alert_conditions(self, df: pd.DataFrame) -> bool:
        conditions = [
            df['rsi'].iloc[-1] < 30 or df['rsi'].iloc[-1] > 70,
            df['volume'].iloc[-1] > 2 * df['volume'].rolling(20).mean().iloc[-1],
            abs(df['macd_diff'].iloc[-1]) > 0.5,
            self._trend_consistency(df) > 0.8
        ]
        return sum(conditions) >= 3
