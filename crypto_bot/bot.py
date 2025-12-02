import os
import logging
import asyncio
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters

from market import MarketData
from analyst import AIAnalyst

# Load environment variables
load_dotenv()

# Logging setup
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

class CryptoBot:
    def __init__(self):
        self.token = os.getenv("TELEGRAM_BOT_TOKEN")
        if not self.token:
            raise ValueError("TELEGRAM_BOT_TOKEN not found in .env")
        
        self.market = MarketData()
        self.analyst = AIAnalyst()

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "🚀 **Crypto AI Analyst Bot Ready**\n\n"
            "Commands:\n"
            "`/BTCUSD` - Analyze BTC/USDT\n"
            "`/ETHUSD` - Analyze ETH/USDT\n"
            "Or just type a symbol like `SOL` to analyze SOL/USDT.",
            parse_mode="Markdown"
        )

    async def _parse_symbol(self, user_input: str) -> str:
        """Helper to clean and format the symbol string."""
        symbol = user_input.strip().upper().replace("/", "")
        
        # Handle generic text input or command input removal if passed raw
        if symbol.startswith("/"):
            symbol = symbol.split(" ")[-1]

        # Normalize symbol to Binance format (e.g., BTCUSDT)
        if symbol.endswith("USD") and not symbol.endswith("USDT"):
            symbol = symbol + "T"
        
        quote_currencies = ["USDT", "BUSD", "BTC", "ETH", "BNB"]
        has_valid_quote = False
        for q in quote_currencies:
            if symbol.endswith(q) and len(symbol) > len(q):
                has_valid_quote = True
                break
        
        if not has_valid_quote:
            symbol += "USDT"
            
        return symbol

    async def _run_analysis(self, update: Update, symbol_input: str, style: str):
        """Common analysis logic for all styles."""
        # Handle "START" text if it leaks through
        if symbol_input.upper().startswith("START"):
            return

        symbol = await self._parse_symbol(symbol_input)
        
        style_emoji = {
            "scalp": "⚡",
            "swing": "🌊",
            "intraday": "🔍"
        }
        emoji = style_emoji.get(style, "🔍")

        status_msg = await update.message.reply_text(f"{emoji} {style.capitalize()} Analysis for {symbol}...\n1. Fetching market data...")

        try:
            # 1. Get Data (Async)
            market_data = await self.market.get_market_analysis(symbol)
            
            # 2. AI Analysis (Async)
            await status_msg.edit_text(f"🤖 AI {style.capitalize()} Analyst Thinking (Checking 🐂 vs 🐻 cases)...")
            
            # Pass the style to the analyst
            ai_response = await self.analyst.analyze_market(market_data, style=style)
            
            if "error" in ai_response:
                await status_msg.edit_text(f"❌ AI Error: {ai_response['error']}")
                return

            # 3. Format Output
            setup = ai_response.get("trade_setup", {})
            
            def safe_str(val):
                return "N/A" if val is None else str(val)
            
            formatted_msg = (
                f"📊 {style.capitalize()} Analysis for {safe_str(ai_response.get('symbol', symbol))}\n\n"
                f"Sentiment: {safe_str(ai_response.get('market_sentiment', 'N/A')).upper()}\n\n"
                f"⚖️ Bull vs Bear:\n{safe_str(ai_response.get('bull_bear_case', 'N/A'))}\n\n"
                f"📈 Technicals:\n{safe_str(ai_response.get('technical_summary', 'N/A'))}\n\n"
                f"🎯 Trade Setup ({safe_str(setup.get('style', style))})\n"
                f"• Direction: {safe_str(setup.get('direction', 'N/A')).upper()}\n"
                f"• Entry: {safe_str(setup.get('entry_zone', 'N/A'))}\n"
                f"• TP: {safe_str(setup.get('take_profit', 'N/A'))}\n"
                f"• SL: {safe_str(setup.get('stop_loss', 'N/A'))}\n"
                f"• Invalidation: {safe_str(setup.get('invalidation', 'N/A'))}\n"
                f"• Confidence: {safe_str(setup.get('confidence_score', 0))}/100\n\n"
                f"📝 Reasoning:\n{safe_str(setup.get('reasoning', 'N/A'))}\n\n"
                f"{safe_str(ai_response.get('risk_disclaimer', 'Not financial advice.'))}"
            )

            await status_msg.edit_text(formatted_msg)

        except Exception as e:
            logging.error(f"Error analyzing {symbol}: {e}")
            await status_msg.edit_text(f"❌ Error: {str(e)}\nMake sure the symbol is valid on Binance (e.g., BTCUSDT).")

    async def analyze(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Default Intraday Analysis"""
        user_input = update.message.text
        # Remove command if present (e.g. /BTCUSD)
        if user_input.startswith("/"):
             user_input = user_input[1:]
        await self._run_analysis(update, user_input, style="intraday")

    async def scalp(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Scalping Analysis (Short-term)"""
        if not context.args:
            await update.message.reply_text("Usage: /scalp <SYMBOL> (e.g., /scalp BTC)")
            return
        symbol = context.args[0]
        await self._run_analysis(update, symbol, style="scalp")

    async def swing(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Swing Analysis (Medium/Long-term)"""
        if not context.args:
            await update.message.reply_text("Usage: /swing <SYMBOL> (e.g., /swing ETH)")
            return
        symbol = context.args[0]
        await self._run_analysis(update, symbol, style="swing")

def main():
    bot = CryptoBot()
    application = ApplicationBuilder().token(bot.token).build()

    # Handlers
    application.add_handler(CommandHandler("start", bot.start))
    application.add_handler(CommandHandler("scalp", bot.scalp))
    application.add_handler(CommandHandler("swing", bot.swing))
    
    # Handle commands like /BTCUSD (Default to intraday)
    application.add_handler(MessageHandler(filters.COMMAND, bot.analyze))
    
    # Handle plain text symbols (Default to intraday)
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.analyze))

    print("Bot is running...")
    application.run_polling()

if __name__ == '__main__':
    main()
