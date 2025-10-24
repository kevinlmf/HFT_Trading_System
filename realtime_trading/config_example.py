"""
Example Configuration for Real-Time Trading System

Copy this file to config.py and fill in your credentials
"""

# ==================== API CREDENTIALS ====================

# Alpaca Markets (Free tier available)
ALPACA_API_KEY = "YOUR_API_KEY_HERE"
ALPACA_API_SECRET = "YOUR_API_SECRET_HERE"
ALPACA_PAPER = True  # Use paper trading

# Add other brokers/data providers as needed
# BINANCE_API_KEY = "..."
# INTERACTIVE_BROKERS_KEY = "..."


# ==================== TRADING PARAMETERS ====================

# Initial capital
INITIAL_CAPITAL = 100000.0

# Symbols to trade
TRADING_SYMBOLS = [
    'AAPL',   # Apple
    'MSFT',   # Microsoft
    'GOOGL',  # Google
    'AMZN',   # Amazon
    'TSLA',   # Tesla
]

# Update intervals
SIGNAL_UPDATE_INTERVAL = 5.0  # Check for signals every 5 seconds
PNL_SNAPSHOT_INTERVAL = 60.0  # Snapshot P&L every minute
STATUS_REPORT_INTERVAL = 300.0  # Status report every 5 minutes


# ==================== RISK PARAMETERS ====================

# Position sizing
MAX_POSITION_SIZE = 0.2  # Max 20% of capital per position
MAX_TOTAL_EXPOSURE = 0.8  # Max 80% total exposure

# Commission and fees
COMMISSION_RATE = 0.0001  # 0.01% per trade

# Stop loss
USE_STOP_LOSS = True
STOP_LOSS_PCT = 0.05  # 5% stop loss

# Take profit
USE_TAKE_PROFIT = True
TAKE_PROFIT_PCT = 0.10  # 10% take profit


# ==================== STRATEGY CONFIGURATION ====================

# Strategies to use
ENABLED_STRATEGIES = {
    'momentum': {
        'enabled': True,
        'lookback': 20,
        'threshold': 0.02,
    },
    'pairs_trading': {
        'enabled': True,
        'window': 60,
        'entry_z': 2.0,
        'exit_z': 0.5,
    },
    'market_making': {
        'enabled': True,
        'spread': 0.01,
        'max_inventory': 1000,
    },
    'lstm': {
        'enabled': False,  # Requires training
        'input_size': 10,
        'hidden_size': 64,
    },
}

# Strategy router parameters
STRATEGY_ROUTER = {
    'lookback_window': 100,
    'regime_window': 50,
    'min_trades_for_selection': 10,
}


# ==================== LOGGING ====================

LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
LOG_TO_FILE = True
LOG_FILE = "logs/trading.log"


# ==================== SAFETY CHECKS ====================

# Maximum daily loss (stop trading if exceeded)
MAX_DAILY_LOSS = 5000.0  # $5000

# Maximum drawdown (stop trading if exceeded)
MAX_DRAWDOWN = 0.20  # 20%

# Trading hours (US Eastern Time)
TRADING_START_HOUR = 9  # 9:30 AM
TRADING_START_MINUTE = 30
TRADING_END_HOUR = 16  # 4:00 PM
TRADING_END_MINUTE = 0

# Allow after-hours trading
ALLOW_AFTER_HOURS = False


# ==================== DATABASE ====================

# Save trades to database
USE_DATABASE = False
DATABASE_URL = "sqlite:///trading_history.db"  # Or PostgreSQL URL


# ==================== NOTIFICATIONS ====================

# Email notifications
ENABLE_EMAIL_NOTIFICATIONS = False
EMAIL_ADDRESS = "your_email@example.com"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

# Telegram notifications
ENABLE_TELEGRAM = False
TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN"
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID"


# ==================== BACKTESTING MODE ====================

# Use historical data instead of live
BACKTEST_MODE = False
BACKTEST_START_DATE = "2024-01-01"
BACKTEST_END_DATE = "2024-12-31"
BACKTEST_DATA_PATH = "data/historical/"
