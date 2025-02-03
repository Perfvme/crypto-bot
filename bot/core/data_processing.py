import pandas as pd
import talib
from ta.trend import EMAIndicator, MACD
from ta.volatility import AverageTrueRange, BollingerBands
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volume import VolumeWeightedAveragePrice

class DataEnhancer:
    def __init__(self, df):
        self.df = df.copy()
        
    def add_all_indicators(self):
        self._add_trend_indicators()
        self._add_volatility_indicators()
        self._add_momentum_indicators()
        self._add_volume_indicators()
        self._add_custom_indicators()
        return self.df
    
    def _add_trend_indicators(self):
        # EMA Ribbon
        periods = [8, 13, 21, 34, 55]
        for period in periods:
            self.df[f'ema_{period}'] = EMAIndicator(self.df['close'], window=period).ema_indicator()
        
        # MACD
        macd = MACD(self.df['close'])
        self.df['macd'] = macd.macd()
        self.df['macd_signal'] = macd.macd_signal()
        self.df['macd_diff'] = macd.macd_diff()
        
        # Ichimoku Cloud
        self.df['tenkan_sen'] = (self.df['high'].rolling(9).max() + self.df['low'].rolling(9).min()) / 2
        self.df['kijun_sen'] = (self.df['high'].rolling(26).max() + self.df['low'].rolling(26).min()) / 2
        self.df['senkou_span_a'] = ((self.df['tenkan_sen'] + self.df['kijun_sen']) / 2).shift(26)
        self.df['senkou_span_b'] = ((self.df['high'].rolling(52).max() + self.df['low'].rolling(52).min()) / 2).shift(26)
        
    def _add_volatility_indicators(self):
        # ATR
        atr = AverageTrueRange(self.df['high'], self.df['low'], self.df['close'], window=14)
        self.df['atr'] = atr.average_true_range()
        
        # Bollinger Bands
        bb = BollingerBands(self.df['close'], window=20, window_dev=2)
        self.df['bb_upper'] = bb.bollinger_hband()
        self.df['bb_middle'] = bb.bollinger_mavg()
        self.df['bb_lower'] = bb.bollinger_lband()
        
    def _add_momentum_indicators(self):
        # RSI
        self.df['rsi'] = RSIIndicator(self.df['close'], window=14).rsi()
        
        # Stochastic RSI
        stoch = StochasticOscillator(
            high=self.df['high'],
            low=self.df['low'],
            close=self.df['close'],
            window=14,
            smooth_window=3
        )
        self.df['stoch_k'] = stoch.stoch()
        self.df['stoch_d'] = stoch.stoch_signal()
        
    def _add_volume_indicators(self):
        # VWAP
        vwap = VolumeWeightedAveragePrice(
            high=self.df['high'],
            low=self.df['low'],
            close=self.df['close'],
            volume=self.df['volume'],
            window=20
        )
        self.df['vwap'] = vwap.volume_weighted_average_price()
        
    def _add_custom_indicators(self):
        # Fractal Indicators
        self.df['fractal_bull'] = (
            (self.df['high'] > self.df['high'].shift(2)) &
            (self.df['high'] > self.df['high'].shift(1)) &
            (self.df['high'] > self.df['high'].shift(-1)) &
            (self.df['high'] > self.df['high'].shift(-2))
        )
        self.df['fractal_bear'] = (
            (self.df['low'] < self.df['low'].shift(2)) &
            (self.df['low'] < self.df['low'].shift(1)) &
            (self.df['low'] < self.df['low'].shift(-1)) &
            (self.df['low'] < self.df['low'].shift(-2))
        )
        
        # Custom Supertrend
        self.df['supertrend'] = self._calculate_supertrend()
        
    def _calculate_supertrend(self, period=10, multiplier=3):
        hl2 = (self.df['high'] + self.df['low']) / 2
        atr = self.df['atr']
        
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)
        
        supertrend = pd.Series(index=self.df.index)
        direction = pd.Series(1, index=self.df.index)
        
        for i in range(1, len(self.df)):
            if self.df['close'][i] > upper_band[i-1]:
                direction[i] = 1
            elif self.df['close'][i] < lower_band[i-1]:
                direction[i] = -1
            else:
                direction[i] = direction[i-1]
                
            supertrend[i] = lower_band[i] if direction[i] == 1 else upper_band[i]
            
        return supertrend
