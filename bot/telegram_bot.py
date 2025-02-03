import os
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    CallbackQueryHandler,
    JobQueue
)
from dotenv import load_dotenv
from api.binance_client import BinanceClient
from api.news_api import NewsAnalyzer
from core.market_analysis import MarketAnalyzer
from core.alert_system import AlertSystem

load_dotenv()

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

class CryptoTelegramBot:
    def __init__(self):
        self.binance = BinanceClient()
        self.analyzer = MarketAnalyzer()
        self.news = NewsAnalyzer()
        self.alerts = AlertSystem()
        self.coins = ['BTC', 'ETH', 'SOL', 'XRP', 'DOGE', 'LTC']
        
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        keyboard = [
            [InlineKeyboardButton(f"Analyze {coin}", callback_data=f"analyze_{coin}") 
             for coin in self.coins[:3]],
            [InlineKeyboardButton(f"Analyze {coin}", callback_data=f"analyze_{coin}") 
             for coin in self.coins[3:]],
            [InlineKeyboardButton("System Status", callback_data="status"),
             InlineKeyboardButton("Active Alerts", callback_data="alerts")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "🤖 Crypto Trading Bot Active\n\n"
            "Available Commands:\n"
            "/analyze [coin] - Detailed analysis\n"
            "/news [coin] - Latest news impact\n"
            "/alerts - Active trading signals\n"
            "/status - System health check",
            reply_markup=reply_markup
        )

    async def analyze_coin(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        coin = query.data.split('_')[1]
        
        try:
            # Get analysis data
            df = self.binance.get_historical_data(f"{coin}/USDT", "15m")
            analysis = self.analyzer.full_analysis(df, coin)
            news = self.news.get_coin_news(coin)
            
            # Format message
            message = self._format_analysis(analysis, news)
            await query.edit_message_text(
                text=message,
                parse_mode='Markdown',
                reply_markup=self._analysis_keyboard(coin)
            )
        except Exception as e:
            logging.error(f"Analysis error: {str(e)}")
            await query.edit_message_text(text=f"⚠️ Error analyzing {coin}: {str(e)}")

    def _format_analysis(self, analysis, news):
        return (
            f"🔍 *{analysis['coin']}/USDT Analysis*\n\n"
            f"*Price*: ${analysis['price']:.2f}\n"
            f"*Trend*: {analysis['trend']} ({analysis['confidence']:.1%})\n"
            f"*Volume*: {analysis['volume_change']:.1%} change\n\n"
            f"*Key Levels*\n"
            f"Support: ${analysis['support']:.2f}\n"
            f"Resistance: ${analysis['resistance']:.2f}\n\n"
            f"*Latest News*\n{news['summary']}\n"
            f"Sentiment: {news['sentiment']} ({news['sentiment_score']:.2f})"
        )

    def _analysis_keyboard(self, coin):
        return InlineKeyboardMarkup([
            [InlineKeyboardButton("📈 15m Chart", callback_data=f"chart_15m_{coin}"),
             InlineKeyboardButton("📊 4h Chart", callback_data=f"chart_4h_{coin}")],
            [InlineKeyboardButton("🔔 Set Alert", callback_data=f"alert_{coin}"),
             InlineKeyboardButton("📰 More News", callback_data=f"news_{coin}")]
        ])

    async def alert_checker(self, context: ContextTypes.DEFAULT_TYPE):
        for coin in self.coins:
            df = self.binance.get_historical_data(f"{coin}/USDT", "15m")
            if self.analyzer.check_alert_conditions(df):
                alert_msg = self.alerts.create_alert_message(coin, df)
                await context.bot.send_message(
                    chat_id=os.getenv('TELEGRAM_CHAT_ID'),
                    text=alert_msg,
                    parse_mode='Markdown'
                )

    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        logging.error(f"Update {update} caused error {context.error}")
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="⚠️ An error occurred. Please try again later."
        )

    def run(self):
        application = ApplicationBuilder().token(os.getenv('TELEGRAM_BOT_TOKEN')).build()
        
        # Handlers
        application.add_handler(CommandHandler('start', self.start))
        application.add_handler(CallbackQueryHandler(self.analyze_coin, pattern='^analyze_'))
        
        # Job Queue for alerts
        job_queue = application.job_queue
        job_queue.run_repeating(self.alert_checker, interval=300, first=10)
        
        application.add_error_handler(self.error_handler)
        application.run_polling()

if __name__ == '__main__':
    bot = CryptoTelegramBot()
    bot.run()
