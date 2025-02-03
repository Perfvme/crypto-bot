import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from .data_processing import DataEnhancer
from api.deepseek_client import DeepseekClient

class CryptoML:
    def __init__(self):
        self.scaler = StandardScaler()
        self.ds_client = DeepseekClient()
        self.models = {
            'm5': self._load_model('m5_model'),
            'm15': self._load_model('m15_model')
        }
        
    def predict(self, timeframe, data):
        model = self.models[timeframe]
        features = self._prepare_features(data)
        scaled_features = self.scaler.transform(features)
        
        prediction = model.predict(scaled_features)[0]
        ml_confidence = np.max(model.predict_proba(scaled_features))
        
        # Get DeepSeek analysis
        ds_analysis = self.ds_client.analyze_market(data.iloc[-1].to_dict())
        ds_confidence = ds_analysis.get('confidence_score', 0)
        
        return {
            'direction': 'buy' if prediction == 1 else 'sell',
            'ml_confidence': float(ml_confidence),
            'deepseek_confidence': float(ds_confidence),
            'combined_confidence': (ml_confidence * 0.7 + ds_confidence * 0.3)
        }
    
    def _prepare_features(self, df):
        features = pd.DataFrame()
        # Trend Features
        features['ema_cross_8_21'] = (df['ema_8'] > df['ema_21']).astype(int)
        features['macd_cross'] = (df['macd'] > df['macd_signal']).astype(int)
        
        # Momentum Features
        features['rsi'] = df['rsi']
        features['stoch_cross'] = (df['stoch_k'] > df['stoch_d']).astype(int)
        
        # Volatility Features
        features['bb_percent'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        features['atr_percent'] = df['atr'] / df['close']
        
        # Volume Features
        features['volume_spike'] = (df['volume'] > df['volume'].rolling(20).mean() * 1.5).astype(int)
        features['vwap_distance'] = (df['close'] - df['vwap']) / df['vwap']
        
        # Custom Features
        features['supertrend_dir'] = (df['close'] > df['supertrend']).astype(int)
        features['fractal_bull'] = df['fractal_bull'].astype(int)
        features['fractal_bear'] = df['fractal_bear'].astype(int)
        
        return features.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    def retrain_model(self, timeframe, new_data):
        X = self._prepare_features(new_data)
        y = (new_data['close'].shift(-1) > new_data['close']).astype(int).iloc[:-1]
        X = X.iloc[:-1]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        model = XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False
        )
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        X_test_scaled = self.scaler.transform(X_test)
        preds = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, preds)
        
        # Save model
        joblib.dump(model, f'models/{timeframe}_model.joblib')
        return accuracy
    
    def _load_model(self, model_name):
        try:
            return joblib.load(f'models/{model_name}.joblib')
        except:
            return None
