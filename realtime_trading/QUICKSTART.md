# Real-Time Trading - Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Step 1: Get Free API Keys

1. Go to [https://alpaca.markets](https://alpaca.markets)
2. Sign up for a free account (no credit card required)
3. Navigate to your dashboard
4. Copy your **API Key** and **Secret Key**
5. Make sure you're using **Paper Trading** (not live)

### Step 2: Set API Keys

#### Option A: Environment Variables (Recommended)

```bash
# On Mac/Linux
export ALPACA_API_KEY='your_key_here'
export ALPACA_API_SECRET='your_secret_here'

# On Windows (PowerShell)
$env:ALPACA_API_KEY='your_key_here'
$env:ALPACA_API_SECRET='your_secret_here'

# On Windows (Command Prompt)
set ALPACA_API_KEY=your_key_here
set ALPACA_API_SECRET=your_secret_here
```

#### Option B: Config File

Create `realtime_trading/config.py`:

```python
ALPACA_API_KEY = "your_key_here"
ALPACA_API_SECRET = "your_secret_here"
ALPACA_PAPER = True  # Always use paper trading for testing
```

### Step 3: Install Dependencies

```bash
cd ~/Downloads/System/Quant/HFT/HFT_System

# Install required packages
pip install websockets asyncio pandas numpy
```

### Step 4: Run the Demo

```bash
# Navigate to the HFT_System directory
cd ~/Downloads/System/Quant/HFT/HFT_System

# Run the demo
python realtime_trading/demo_trading.py
```

### Expected Output

```
======================================================================
REAL-TIME TRADING SYSTEM DEMO
======================================================================

✓ API keys found
✓ Using paper trading (no real money)

Connecting to Alpaca Markets...
✓ Connected to Alpaca WebSocket

✓ Subscribed to: AAPL, MSFT, GOOGL

Receiving real-time market data...
(Press Ctrl+C to stop)
----------------------------------------------------------------------
[09:30:15] AAPL: $150.25 (Bid: $150.24, Ask: $150.26) [10 ticks]
[09:30:16] MSFT: $305.50 (Bid: $305.48, Ask: $305.52) [20 ticks]
[09:30:17] GOOGL: $125.75 (Bid: $125.74, Ask: $125.76) [30 ticks]
...
```

---

## 📊 Demo Modes

### Mode 1: Simple Demo (Default)

Just receives and displays market data. Good for testing your connection.

```bash
python realtime_trading/demo_trading.py
# Choose option 1
```

### Mode 2: Full Trading Demo

Includes strategy selection, regime detection, and P&L tracking.

```bash
python realtime_trading/demo_trading.py
# Choose option 2
```

---

## 🔧 Troubleshooting

### "API keys not found"

**Solution**: Make sure you've set the environment variables or created `config.py`

```bash
# Check if variables are set
echo $ALPACA_API_KEY
echo $ALPACA_API_SECRET
```

### "websockets not found"

**Solution**: Install the websockets package

```bash
pip install websockets
```

### "Authentication failed"

**Solution**:
1. Double-check your API keys (no extra spaces)
2. Make sure you're using **Paper Trading** keys
3. Verify your Alpaca account is active

### "Connection timeout"

**Solution**:
1. Check your internet connection
2. Make sure Alpaca services are operational: [https://status.alpaca.markets](https://status.alpaca.markets)
3. Try again in a few minutes

---

## 🎓 Next Steps

### 1. Explore the Code

```bash
# Main trading engine
realtime_trading/trading_engine.py

# Strategy router (auto-selection)
realtime_trading/strategy_router/adaptive_router.py

# P&L tracker
realtime_trading/pnl_tracker/realtime_pnl.py
```

### 2. Add Your Own Strategies

```python
# In strategy/ folder
from strategy.classical import MomentumStrategy

# Create your custom strategy
class MyStrategy(BaseStrategy):
    def generate_signal(self, data):
        # Your logic here
        return 1  # Buy signal
```

### 3. Run Full Trading System

```python
import asyncio
from realtime_trading import RealTimeTradingEngine
from realtime_trading.connectors import AlpacaConnector
from strategy.classical import MomentumStrategy

async def main():
    connector = AlpacaConnector(
        api_key="YOUR_KEY",
        api_secret="YOUR_SECRET",
        paper=True
    )

    strategies = {
        'momentum': MomentumStrategy(lookback=20),
    }

    engine = RealTimeTradingEngine(
        connector=connector,
        strategies=strategies,
        initial_capital=100000.0,
        symbols=['AAPL', 'MSFT']
    )

    await engine.start()

asyncio.run(main())
```

### 4. Monitor Performance

The system automatically tracks:
- Total P&L
- Win rate
- Sharpe ratio
- Max drawdown
- Per-strategy performance

---

## ⚠️ Important Notes

### Paper Trading Only

**ALWAYS** use paper trading when testing:
```python
AlpacaConnector(paper=True)  # ✓ Safe
AlpacaConnector(paper=False) # ✗ Real money!
```

### Market Hours

US stock market hours (Eastern Time):
- **Regular**: 9:30 AM - 4:00 PM
- **Pre-market**: 4:00 AM - 9:30 AM
- **After-hours**: 4:00 PM - 8:00 PM

The demo works during regular hours. Outside market hours, you'll receive limited or no data.

### Rate Limits

Alpaca free tier limits:
- 200 requests/minute for REST API
- Real-time data for IEX (US stocks)
- Paper trading: unlimited

---

## 📞 Need Help?

1. **Read the full documentation**: `realtime_trading/README.md`
2. **Check examples**: `examples/` folder
3. **Alpaca docs**: [https://alpaca.markets/docs](https://alpaca.markets/docs)
4. **GitHub Issues**: Report bugs or ask questions

---

## 🎉 You're Ready!

You now have a working real-time trading system that:
- ✅ Receives live market data
- ✅ Detects market regimes
- ✅ Selects optimal strategies
- ✅ Tracks P&L in real-time

**Happy Trading!** 🚀

---

*Remember: This is for educational purposes. Always test thoroughly before considering live trading.*
