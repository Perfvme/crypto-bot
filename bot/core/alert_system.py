from telegram import Bot
import os
from dotenv import load_dotenv

load_dotenv()

class AlertSystem:
    def __init__(self):
        self.bot = Bot(token=os.getenv('TELEGRAM_BOT_TOKEN'))
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
    def send_alert(self, message):
        self.bot.send_message(chat_id=self.chat_id, text=message)
        
    def check_conditions(self, analysis):
        for coin, data in analysis.items():
            if self._is_strong_signal(data):
                self.send_alert(f"🚨 STRONG SIGNAL: {coin}\n"
                               f"Direction: {data['consensus_direction']}\n"
                               f"Confidence: {data['final_confidence']}%")
    
    def _is_strong_signal(self, data):
        timeframes = ['4h', '1h', '15m', '5m']
        directions = [data[tf]['direction'] for tf in timeframes]
        confidences = [data[tf]['combined_confidence'] for tf in timeframes]
        
        return (
            all(c >= 0.85 for c in confidences) and
            all(d == directions[0] for d in directions)
        )
