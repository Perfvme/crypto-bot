import pandas as pd
import talib
from ta.trend import EMAIndicator, MACD
from ta.volatility import AverageTrueRange, BollingerBands
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volume import VolumeWeightedAveragePrice

class TechnicalIndicators:
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._add_trend_indicators(df)
        df = self._add_volatility_indicators(df)
        df = self._add_momentum_indicators(df)
        df = self._add_volume_indicators(df)
        df = self._add_custom_indicators(df)
        return df.dropna()

    def _add_trend_indicators(self, df):
        # EMA Ribbon
        for period in [8, 13, 21, 34, 55]:
            df[f'ema_{period}'] = EMAIndicator(df['close'], window=period).ema_indicator()
        
        # MACD
        macd = MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # Ichimoku Cloud
        high_9 = df['high'].rolling(9).max()
        low_9 = df['low'].rolling(9).min()
        df['tenkan_sen'] = (high_9 + low_9) / 2
        
        high_26 = df['high'].rolling(26).max()
        low_26 = df['low'].rolling(26).min()
        df['kijun_sen'] = (high_26 + low_26) / 2
        
        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
        df['senkou_span_b'] = ((df['high'].rolling(52).max() + 
                               df['low'].rolling(52).min()) / 2).shift(26)
        return df

    def _add_volatility_indicators(self, df):
        # ATR
        df['atr'] = AverageTrueRange(
            df['high'], df['low'], df['close'], window=14
        ).average_true_range()
        
        # Bollinger Bands
        bb = BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        return df

    def _add_momentum_indicators(self, df):
        # RSI
        df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
        
        # Stochastic
        stoch = StochasticOscillator(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=14,
            smooth_window=3
        )
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        return df

    def _add_volume_indicators(self, df):
        # VWAP
        df['vwap'] = VolumeWeightedAveragePrice(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            volume=df['volume'],
            window=20
        ).volume_weighted_average_price()
        return df

    def _add_custom_indicators(self, df):
        # Fractal Indicators
        df['fractal_bull'] = (
            (df['high'] > df['high'].shift(2)) &
            (df['high'] > df['high'].shift(1)) &
            (df['high'] > df['high'].shift(-1)) &
            (df['high'] > df['high'].shift(-2))
        )
        df['fractal_bear'] = (
            (df['low'] < df['low'].shift(2)) &
            (df['low'] < df['low'].shift(1)) &
            (df['low'] < df['low'].shift(-1)) &
            (df['low'] < df['low'].shift(-2))
        )
        
        # Supertrend
        df['supertrend'] = self._calculate_supertrend(df)
        return df

    def _calculate_supertrend(self, df, period=10, multiplier=3):
        hl2 = (df['high'] + df['low']) / 2
        matr = multiplier * df['atr']
        
        upper = hl2 + matr
        lower = hl2 - matr
        
        supertrend = pd.Series(index=df.index)
        direction = pd.Series(1, index=df.index)
        
        for i in range(1, len(df)):
            if df['close'][i] > upper[i-1]:
                direction[i] = 1
            elif df['close'][i] < lower[i-1]:
                direction[i] = -1
            else:
                direction[i] = direction[i-1]
                
            supertrend[i] = lower[i] if direction[i] == 1 else upper[i]
            
        return supertrend
