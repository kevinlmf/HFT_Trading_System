#!/bin/bash
# End-to-End Trading System - Execution Script
# Usage: ./run_trading.sh [mode] [options]
# Modes: paper, backtest, live, monitor-only, demo

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

# Default configuration
MODE="paper"
SYMBOLS="AAPL,MSFT,GOOGL"
CAPITAL=100000
STRATEGIES="momentum,mean_reversion"
UPDATE_INTERVAL=5
ENABLE_DASHBOARD=false
ENABLE_LOGGING=true
CONFIG_FILE=""
DRY_RUN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        paper|backtest|live|monitor-only|demo)
            MODE=$1
            shift
            ;;
        --symbols)
            SYMBOLS="$2"
            shift 2
            ;;
        --capital)
            CAPITAL="$2"
            shift 2
            ;;
        --strategies)
            STRATEGIES="$2"
            shift 2
            ;;
        --interval)
            UPDATE_INTERVAL="$2"
            shift 2
            ;;
        --dashboard)
            ENABLE_DASHBOARD=true
            shift
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --no-log)
            ENABLE_LOGGING=false
            shift
            ;;
        --help|-h)
            cat << EOF
End-to-End Trading System - Execution Script

Usage: $0 [mode] [options]

MODES:
  paper          Run paper trading (default, no real money)
  backtest       Run historical backtest
  live           Run live trading (CAUTION: real money!)
  monitor-only   Only start monitoring dashboard
  demo           Run simple demo (market data only)

OPTIONS:
  --symbols      Comma-separated list of symbols (default: AAPL,MSFT,GOOGL)
  --capital      Initial capital (default: 100000)
  --strategies   Comma-separated strategies (default: momentum,mean_reversion)
                 Available: momentum, mean_reversion, pairs_trading,
                           market_making, statistical_arbitrage
  --interval     Update interval in seconds (default: 5)
  --dashboard    Enable real-time web dashboard
  --config       Path to config file (overrides other options)
  --dry-run      Show what would run without executing
  --no-log       Disable file logging
  --help         Show this help

EXAMPLES:
  # Paper trading with default settings
  $0 paper

  # Paper trading with custom symbols and strategies
  $0 paper --symbols AAPL,TSLA --strategies momentum,market_making

  # Backtest with dashboard
  $0 backtest --dashboard

  # Live trading (requires API keys)
  $0 live --capital 50000 --strategies momentum

  # Just monitor (no trading)
  $0 monitor-only --dashboard

ENVIRONMENT VARIABLES:
  ALPACA_API_KEY      Alpaca API key (required for paper/live)
  ALPACA_API_SECRET   Alpaca API secret (required for paper/live)

BEFORE RUNNING:
  1. Build the system: ./build_system.sh --all
  2. Set API keys: export ALPACA_API_KEY='your_key'
  3. Test with demo: ./run_trading.sh demo

EOF
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage"
            exit 1
            ;;
    esac
done

# Banner
echo ""
echo -e "${CYAN}============================================================${NC}"
echo -e "${CYAN}     End-to-End Trading System - Execution Manager${NC}"
echo -e "${CYAN}============================================================${NC}"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 not found${NC}"
    exit 1
fi

# Create logs directory
LOGS_DIR="logs"
mkdir -p "$LOGS_DIR"

# Check if system is built
check_system_built() {
    if [ ! -d "Execution" ] || [ ! -f "Execution/trading/trading_engine.py" ]; then
        echo -e "${YELLOW}⚠️  System not built!${NC}"
        echo ""
        echo "Please run first:"
        echo "  ./build_system.sh --all"
        echo ""
        exit 1
    fi
    echo -e "${GREEN}✓ System build verified${NC}"
}

# Check API keys for live/paper trading
check_api_keys() {
    if [ "$MODE" = "paper" ] || [ "$MODE" = "live" ]; then
        if [ -z "$ALPACA_API_KEY" ] || [ -z "$ALPACA_API_SECRET" ]; then
            echo -e "${YELLOW}⚠️  API keys not found!${NC}"
            echo ""
            echo "Please set environment variables:"
            echo "  export ALPACA_API_KEY='your_key'"
            echo "  export ALPACA_API_SECRET='your_secret'"
            echo ""
            echo "Get free API keys at: https://alpaca.markets"
            echo ""
            exit 1
        fi
        echo -e "${GREEN}✓ API keys found${NC}"
    fi
}

# Start monitoring dashboard
start_dashboard() {
    if [ "$ENABLE_DASHBOARD" = true ]; then
        echo ""
        echo -e "${BLUE}Starting Monitoring Dashboard...${NC}"

        if [ -f "Monitoring/dashboard/dashboard_server.py" ]; then
            python3 Monitoring/dashboard/dashboard_server.py &
            DASHBOARD_PID=$!
            echo -e "${GREEN}✓ Dashboard started (PID: $DASHBOARD_PID)${NC}"
            echo -e "${GREEN}  Access at: http://localhost:8050${NC}"
        else
            echo -e "${YELLOW}Warning: Dashboard not found${NC}"
        fi
    fi
}

# Cleanup function
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down...${NC}"

    # Kill dashboard if running
    if [ ! -z "$DASHBOARD_PID" ]; then
        kill $DASHBOARD_PID 2>/dev/null || true
        echo -e "${GREEN}✓ Dashboard stopped${NC}"
    fi

    # Kill trading engine if running
    if [ ! -z "$TRADING_PID" ]; then
        kill $TRADING_PID 2>/dev/null || true
        echo -e "${GREEN}✓ Trading engine stopped${NC}"
    fi

    echo ""
    echo -e "${CYAN}============================================================${NC}"
    echo -e "${CYAN}Trading session ended${NC}"
    echo -e "${CYAN}============================================================${NC}"
    echo ""
}

# Register cleanup handler
trap cleanup EXIT INT TERM

# Display configuration
echo ""
echo -e "${BLUE}Configuration:${NC}"
echo "  Mode:       $MODE"
echo "  Symbols:    $SYMBOLS"
echo "  Capital:    \$${CAPITAL}"
echo "  Strategies: $STRATEGIES"
echo "  Interval:   ${UPDATE_INTERVAL}s"
echo "  Dashboard:  $ENABLE_DASHBOARD"
echo "  Logging:    $ENABLE_LOGGING"
if [ ! -z "$CONFIG_FILE" ]; then
    echo "  Config:     $CONFIG_FILE"
fi
echo ""

# Dry run check
if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}DRY RUN MODE - Not executing${NC}"
    echo ""
    echo "Would execute:"
    case $MODE in
        paper)
            echo "  python3 Execution/trading/trading_engine.py --paper --symbols $SYMBOLS --capital $CAPITAL"
            ;;
        backtest)
            echo "  python3 examples/run_backtest.py --symbols $SYMBOLS --capital $CAPITAL"
            ;;
        live)
            echo "  python3 Execution/trading/trading_engine.py --live --symbols $SYMBOLS --capital $CAPITAL"
            ;;
        demo)
            echo "  python3 Execution/trading/demo_trading.py"
            ;;
        monitor-only)
            echo "  python3 Monitoring/dashboard/dashboard_server.py"
            ;;
    esac
    exit 0
fi

# Check prerequisites
check_system_built

# Execute based on mode
case $MODE in
    paper)
        echo -e "${GREEN}============================================================${NC}"
        echo -e "${GREEN}Starting PAPER TRADING${NC}"
        echo -e "${GREEN}(No real money will be used)${NC}"
        echo -e "${GREEN}============================================================${NC}"
        echo ""

        check_api_keys
        start_dashboard

        echo -e "${BLUE}Launching trading engine...${NC}"
        echo ""

        # Create a Python runner script
        cat > /tmp/run_paper_trading.py << 'PYTHON_SCRIPT'
import asyncio
import sys
import os
from datetime import datetime

# Add project to path
sys.path.insert(0, os.getcwd())

from Execution.trading.trading_engine import RealTimeTradingEngine
from Data.connectors.alpaca_connector import AlpacaConnector
from Execution.strategies.strategy_registry import get_strategy

async def main():
    # Get configuration from environment
    symbols = os.environ.get('SYMBOLS', 'AAPL,MSFT,GOOGL').split(',')
    capital = float(os.environ.get('CAPITAL', '100000'))
    strategy_names = os.environ.get('STRATEGIES', 'momentum').split(',')
    interval = float(os.environ.get('UPDATE_INTERVAL', '5'))

    print(f"Symbols: {symbols}")
    print(f"Capital: ${capital:,.2f}")
    print(f"Strategies: {strategy_names}")
    print("")

    # Initialize connector
    connector = AlpacaConnector(
        api_key=os.environ['ALPACA_API_KEY'],
        api_secret=os.environ['ALPACA_API_SECRET'],
        paper=True
    )

    # Initialize strategies
    strategies = {}
    for name in strategy_names:
        try:
            strategies[name] = get_strategy(name)
            print(f"✓ Loaded strategy: {name}")
        except Exception as e:
            print(f"⚠️  Could not load strategy {name}: {e}")

    if not strategies:
        print("❌ No strategies loaded!")
        return

    print("")

    # Create engine
    engine = RealTimeTradingEngine(
        connector=connector,
        strategies=strategies,
        initial_capital=capital,
        symbols=symbols,
        update_interval=interval
    )

    # Start trading
    await engine.start()

if __name__ == "__main__":
    asyncio.run(main())
PYTHON_SCRIPT

        # Export configuration
        export SYMBOLS="$SYMBOLS"
        export CAPITAL="$CAPITAL"
        export STRATEGIES="$STRATEGIES"
        export UPDATE_INTERVAL="$UPDATE_INTERVAL"

        # Run trading engine
        if [ "$ENABLE_LOGGING" = true ]; then
            LOGFILE="$LOGS_DIR/paper_trading_$(date +%Y%m%d_%H%M%S).log"
            python3 /tmp/run_paper_trading.py 2>&1 | tee "$LOGFILE"
        else
            python3 /tmp/run_paper_trading.py
        fi
        ;;

    backtest)
        echo -e "${GREEN}============================================================${NC}"
        echo -e "${GREEN}Starting BACKTESTING${NC}"
        echo -e "${GREEN}============================================================${NC}"
        echo ""

        start_dashboard

        echo -e "${BLUE}Running backtest...${NC}"
        echo ""

        if [ -f "Environment/backtester/simple_backtester.py" ]; then
            if [ "$ENABLE_LOGGING" = true ]; then
                LOGFILE="$LOGS_DIR/backtest_$(date +%Y%m%d_%H%M%S).log"
                python3 -c "
import sys
sys.path.insert(0, '.')
from Environment.backtester.simple_backtester import run_backtest
symbols = '$SYMBOLS'.split(',')
run_backtest(symbols=symbols, capital=$CAPITAL)
" 2>&1 | tee "$LOGFILE"
            else
                python3 -c "
import sys
sys.path.insert(0, '.')
from Environment.backtester.simple_backtester import run_backtest
symbols = '$SYMBOLS'.split(',')
run_backtest(symbols=symbols, capital=$CAPITAL)
"
            fi
        else
            echo -e "${RED}Error: Backtester not found${NC}"
            exit 1
        fi
        ;;

    live)
        echo -e "${RED}============================================================${NC}"
        echo -e "${RED}⚠️  LIVE TRADING MODE${NC}"
        echo -e "${RED}⚠️  REAL MONEY WILL BE USED${NC}"
        echo -e "${RED}============================================================${NC}"
        echo ""
        echo -e "${YELLOW}Are you sure you want to continue?${NC}"
        echo "Type 'yes' to proceed, anything else to cancel:"
        read -r confirmation

        if [ "$confirmation" != "yes" ]; then
            echo "Cancelled"
            exit 0
        fi

        check_api_keys
        start_dashboard

        echo ""
        echo -e "${RED}Starting LIVE trading...${NC}"
        echo ""

        # Similar to paper trading but with paper=False
        cat > /tmp/run_live_trading.py << 'PYTHON_SCRIPT'
import asyncio
import sys
import os

sys.path.insert(0, os.getcwd())

from Execution.trading.trading_engine import RealTimeTradingEngine
from Data.connectors.alpaca_connector import AlpacaConnector
from Execution.strategies.strategy_registry import get_strategy

async def main():
    symbols = os.environ.get('SYMBOLS', 'AAPL').split(',')
    capital = float(os.environ.get('CAPITAL', '10000'))
    strategy_names = os.environ.get('STRATEGIES', 'momentum').split(',')
    interval = float(os.environ.get('UPDATE_INTERVAL', '5'))

    print("⚠️  LIVE TRADING - USING REAL MONEY")
    print(f"Symbols: {symbols}")
    print(f"Capital: ${capital:,.2f}")
    print("")

    # Initialize connector (LIVE mode)
    connector = AlpacaConnector(
        api_key=os.environ['ALPACA_API_KEY'],
        api_secret=os.environ['ALPACA_API_SECRET'],
        paper=False  # LIVE TRADING
    )

    strategies = {}
    for name in strategy_names:
        try:
            strategies[name] = get_strategy(name)
        except:
            pass

    engine = RealTimeTradingEngine(
        connector=connector,
        strategies=strategies,
        initial_capital=capital,
        symbols=symbols,
        update_interval=interval
    )

    await engine.start()

if __name__ == "__main__":
    asyncio.run(main())
PYTHON_SCRIPT

        export SYMBOLS="$SYMBOLS"
        export CAPITAL="$CAPITAL"
        export STRATEGIES="$STRATEGIES"
        export UPDATE_INTERVAL="$UPDATE_INTERVAL"

        if [ "$ENABLE_LOGGING" = true ]; then
            LOGFILE="$LOGS_DIR/live_trading_$(date +%Y%m%d_%H%M%S).log"
            python3 /tmp/run_live_trading.py 2>&1 | tee "$LOGFILE"
        else
            python3 /tmp/run_live_trading.py
        fi
        ;;

    demo)
        echo -e "${GREEN}============================================================${NC}"
        echo -e "${GREEN}Starting DEMO MODE${NC}"
        echo -e "${GREEN}(Market data only, no trading)${NC}"
        echo -e "${GREEN}============================================================${NC}"
        echo ""

        check_api_keys

        if [ -f "Execution/trading/demo_trading.py" ]; then
            python3 Execution/trading/demo_trading.py
        else
            echo -e "${RED}Error: demo_trading.py not found${NC}"
            exit 1
        fi
        ;;

    monitor-only)
        echo -e "${GREEN}============================================================${NC}"
        echo -e "${GREEN}Starting MONITORING DASHBOARD${NC}"
        echo -e "${GREEN}============================================================${NC}"
        echo ""

        if [ -f "Monitoring/dashboard/dashboard_server.py" ]; then
            echo -e "${GREEN}Dashboard running at: http://localhost:8050${NC}"
            echo ""
            echo "Press Ctrl+C to stop"
            python3 Monitoring/dashboard/dashboard_server.py
        else
            echo -e "${YELLOW}Creating simple monitoring server...${NC}"

            # Create a simple Flask monitoring server
            cat > /tmp/simple_monitor.py << 'PYTHON_SCRIPT'
from flask import Flask, render_template_string
import json

app = Flask(__name__)

@app.route('/')
def dashboard():
    return render_template_string('''
    <html>
    <head>
        <title>Trading System Monitor</title>
        <style>
            body { font-family: Arial; background: #1e1e1e; color: #fff; padding: 20px; }
            .panel { background: #2d2d2d; padding: 20px; margin: 10px; border-radius: 8px; }
            h1 { color: #4CAF50; }
            .metric { font-size: 24px; margin: 10px 0; }
        </style>
    </head>
    <body>
        <h1>Trading System Monitor</h1>
        <div class="panel">
            <h2>System Status</h2>
            <div class="metric">Status: <span style="color:#4CAF50">Running</span></div>
            <div class="metric">Mode: Paper Trading</div>
        </div>
        <div class="panel">
            <h2>Portfolio</h2>
            <div class="metric">Total P&L: $0.00</div>
            <div class="metric">Equity: $100,000.00</div>
        </div>
    </body>
    </html>
    ''')

if __name__ == '__main__':
    print("Dashboard at: http://localhost:8050")
    app.run(host='0.0.0.0', port=8050, debug=False)
PYTHON_SCRIPT

            python3 /tmp/simple_monitor.py
        fi
        ;;

    *)
        echo -e "${RED}Unknown mode: $MODE${NC}"
        echo "Use --help for available modes"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}Trading session complete${NC}"
echo ""
