from apscheduler.schedulers.background import BackgroundScheduler
from core.ml_models import CryptoML
from api.binance_client import BinanceClient
import datetime

def retrain_job():
    binance = BinanceClient()
    ml = CryptoML()
    
    for symbol in ['BTC/USDT', 'ETH/USDT']:
        data = binance.get_historical_data(symbol, '15m', 5000)
        accuracy = ml.retrain_model('m15', data)
        print(f"Retrained M15 model for {symbol} - Accuracy: {accuracy:.2%}")
        
    scheduler = BackgroundScheduler()
    scheduler.add_job(retrain_job, 'interval', hours=int(os.getenv('MODEL_RETRAIN_HOURS')))
    scheduler.start()
