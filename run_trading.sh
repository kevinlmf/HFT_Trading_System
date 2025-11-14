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
RISK_MODEL="risk_parity"
MONTE_CARLO_PATHS=100000
SLIPPAGE_IMPL=""
CONNECTOR="alpaca"  # HFT-ready: alpaca, polygon, binance, coinbase | Testing: yahoo, alphavantage, iexcloud
# QDB configuration
ENABLE_QDB=true
QDB_PATH="./Data/datasets/qdb"
QDB_DATA_VERSION="qdb_$(date +%Y%m%d)"
QDB_BUFFER_SIZE=1000
QDB_OPTIMIZED=false  # Use optimized indexer (O(log n) instead of O(n))

# Research Framework configuration
ENABLE_RESEARCH=true  # Enable Research Framework (microstructure profiling + factor discovery) - DEFAULT ENABLED
RESEARCH_MODE="full"   # Options: "profiling_only", "factors_only", "full"
RESEARCH_OUTPUT="./results/research"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        paper|backtest|live|monitor-only|demo|complete-flow|benchmark-slippage)
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
        --risk-model)
            RISK_MODEL="$2"
            shift 2
            ;;
        --monte-carlo-paths)
            MONTE_CARLO_PATHS="$2"
            shift 2
            ;;
        --slippage-impl)
            SLIPPAGE_IMPL="$2"
            shift 2
            ;;
        --qdb-path)
            QDB_PATH="$2"
            shift 2
            ;;
        --no-qdb)
            ENABLE_QDB=false
            shift
            ;;
        --qdb-version)
            QDB_DATA_VERSION="$2"
            shift 2
            ;;
        --qdb-optimized)
            QDB_OPTIMIZED=true
            shift
            ;;
        --enable-research)
            ENABLE_RESEARCH=true
            shift
            ;;
        --research-mode)
            RESEARCH_MODE="$2"
            shift 2
            ;;
        --research-output)
            RESEARCH_OUTPUT="$2"
            shift 2
            ;;
        --connector)
            CONNECTOR="$2"
            shift 2
            ;;
        --help|-h)
            cat << EOF
End-to-End Trading System - Execution Script

Usage: $0 [mode] [options]

MODES:
  paper              Run paper trading (default, no real money)
  backtest           Run historical backtest
  live               Run live trading (CAUTION: real money!)
  monitor-only       Only start monitoring dashboard
  demo               Run simple demo (market data only)
  complete-flow      Run complete trading flow (EDA + Strategy + Risk + Position)
  benchmark-slippage Run slippage performance benchmark

OPTIONS:
  --symbols      Comma-separated list of symbols (default: AAPL,MSFT,GOOGL)
  --capital      Initial capital (default: 100000)
  --strategies   Comma-separated strategies (default: all)
                 Use 'all' to test all available strategies (Classical, ML, RL, HFT)
                 Or specify: momentum, mean_reversion, rsi, macd, ml_random_forest, etc.
  --interval     Update interval in seconds (default: 5)
  --dashboard    Enable real-time web dashboard
  --config       Path to config file (overrides other options)
  --risk-model   Risk model for position management (default: risk_parity)
                 Options: equal_weight, inverse_volatility, mean_variance,
                         risk_parity, black_litterman, hrp
  --monte-carlo-paths  Number of Monte Carlo paths (default: 100000)
  --slippage-impl Force smart executor implementation (python_vectorized, cpp, cuda)
  --qdb-path     QDB data storage path (default: ./Data/datasets/qdb)
  --no-qdb       Disable QDB integration (use legacy data loading)
  --qdb-version  QDB data version tag (default: qdb_YYYYMMDD)
  --qdb-optimized Use optimized QDB indexer (O(log n) instead of O(n))
  --enable-research Enable Research Framework (microstructure profiling + factor discovery)
  --research-mode Research mode: profiling_only, factors_only, or full (default: full)
  --research-output Output directory for research results (default: ./results/research)
  --connector    Data connector: alpaca (default), yahoo, binance, polygon, alphavantage, coinbase, iexcloud
                 - alpaca: Alpaca Markets (requires API keys)
                 - yahoo: Yahoo Finance (free, no API key needed, 15-20min delay)
                 - binance: Binance (cryptocurrency, testnet available)
                 - polygon: Polygon.io (requires API key)
                 - alphavantage: Alpha Vantage (free, 500 calls/day, requires API key)
                 - coinbase: Coinbase Pro (cryptocurrency, sandbox available)
                 - iexcloud: IEX Cloud (free, 50k messages/month, requires API key)
  --dry-run      Show what would run without executing
  --no-log       Disable file logging
  --help         Show this help

EXAMPLES:
  # Complete trading flow (recommended - includes all features)
  $0 complete-flow --symbols AAPL,MSFT,GOOGL --capital 1000000

  # Complete flow with custom risk model and QDB
  $0 complete-flow --symbols AAPL,MSFT --risk-model mean_variance --qdb-path ./Data/qdb

  # Paper trading with QDB (default enabled)
  $0 paper --symbols AAPL,MSFT,GOOGL

  # Paper trading with Yahoo Finance (free, no API key needed)
  $0 paper --connector yahoo --symbols AAPL,MSFT

  # Paper trading with Coinbase Pro (cryptocurrency, no API key needed for data)
  $0 paper --connector coinbase --symbols BTC-USD,ETH-USD --interval 10

  # Paper trading with custom QDB version
  $0 paper --symbols AAPL --qdb-version qdb_2024Q1

  # Paper trading without QDB (legacy mode)
  $0 paper --no-qdb --symbols AAPL

  # Backtest with QDB data
  $0 backtest --symbols AAPL,MSFT --qdb-path ./Data/datasets/qdb

  # Backtest with optimized QDB (faster for large datasets)
  $0 backtest --symbols AAPL,MSFT --qdb-optimized

  # Live trading with QDB realtime collection
  $0 live --capital 50000 --strategies momentum --qdb-version qdb_live_2024

  # Slippage performance benchmark
  $0 benchmark-slippage

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

# Initialize QDB
init_qdb() {
    if [ "$ENABLE_QDB" = true ]; then
        echo -e "${BLUE}Initializing QDB...${NC}"
        
        # Check if QDB module exists
        if [ ! -f "Data/qdb/qdb.py" ]; then
            echo -e "${YELLOW}⚠️  QDB module not found, skipping QDB initialization${NC}"
            ENABLE_QDB=false
            return
        fi
        
        # Create QDB directory
        mkdir -p "$QDB_PATH"
        
        # Initialize QDB (create Python script to check)
        python3 -c "
import sys
sys.path.insert(0, '.')
try:
    from Data.qdb import create_qdb
    qdb = create_qdb(base_path='$QDB_PATH')
    print('✓ QDB initialized successfully')
    print(f'  Storage path: $QDB_PATH')
    print(f'  Data version: $QDB_DATA_VERSION')
    if '$QDB_OPTIMIZED' == 'true':
        print('  Using optimized indexer (O(log n))')
except Exception as e:
    print(f'⚠️  QDB initialization warning: {e}')
    sys.exit(0)
" 2>/dev/null || {
            echo -e "${YELLOW}⚠️  QDB initialization skipped (optional)${NC}"
            ENABLE_QDB=false
        }
        
        if [ "$ENABLE_QDB" = true ]; then
            echo -e "${GREEN}✓ QDB ready${NC}"
        fi
    fi
}

# Check API keys for live/paper trading
check_api_keys() {
    if [ "$MODE" = "paper" ] || [ "$MODE" = "live" ]; then
        # Yahoo Finance doesn't need API keys
        if [ "$CONNECTOR" = "yahoo" ]; then
            echo -e "${GREEN}✓ Using Yahoo Finance (no API key needed)${NC}"
            return
        fi
        
        # Binance testnet doesn't need API keys for data
        if [ "$CONNECTOR" = "binance" ]; then
            echo -e "${GREEN}✓ Using Binance (testnet, no API key needed for data)${NC}"
            return
        fi
        
        # Coinbase sandbox doesn't need API keys for data
        if [ "$CONNECTOR" = "coinbase" ]; then
            echo -e "${GREEN}✓ Using Coinbase Pro (sandbox, no API key needed for data)${NC}"
            return
        fi
        
        # Alpha Vantage needs API key
        if [ "$CONNECTOR" = "alphavantage" ]; then
            if [ -z "$ALPHAVANTAGE_API_KEY" ]; then
                echo -e "${YELLOW}⚠️  Alpha Vantage API key not found!${NC}"
                echo ""
                echo "Please set environment variable:"
                echo "  export ALPHAVANTAGE_API_KEY='your_key'"
                echo ""
                echo "Get free API key at: https://www.alphavantage.co/support/#api-key"
                echo ""
                exit 1
            fi
            echo -e "${GREEN}✓ Alpha Vantage API key found${NC}"
        fi
        
        # IEX Cloud needs API key
        if [ "$CONNECTOR" = "iexcloud" ]; then
            if [ -z "$IEXCLOUD_API_KEY" ]; then
                echo -e "${YELLOW}⚠️  IEX Cloud API key not found!${NC}"
                echo ""
                echo "Please set environment variable:"
                echo "  export IEXCLOUD_API_KEY='your_key'"
                echo ""
                echo "Get free API key at: https://iexcloud.io/"
                echo ""
                exit 1
            fi
            echo -e "${GREEN}✓ IEX Cloud API key found${NC}"
        fi
        
        # Alpaca and Polygon need API keys
        if [ "$CONNECTOR" = "alpaca" ]; then
        if [ -z "$ALPACA_API_KEY" ] || [ -z "$ALPACA_API_SECRET" ]; then
                echo -e "${YELLOW}⚠️  Alpaca API keys not found!${NC}"
            echo ""
            echo "Please set environment variables:"
            echo "  export ALPACA_API_KEY='your_key'"
            echo "  export ALPACA_API_SECRET='your_secret'"
                echo ""
                echo "Or use Yahoo Finance instead:"
                echo "  $0 paper --connector yahoo --symbols AAPL"
            echo ""
            echo "Get free API keys at: https://alpaca.markets"
            echo ""
            exit 1
        fi
            echo -e "${GREEN}✓ Alpaca API keys found${NC}"
        elif [ "$CONNECTOR" = "polygon" ]; then
            if [ -z "$POLYGON_API_KEY" ]; then
                echo -e "${YELLOW}⚠️  Polygon API key not found!${NC}"
                echo ""
                echo "Please set environment variable:"
                echo "  export POLYGON_API_KEY='your_key'"
                echo ""
                echo "Or use Yahoo Finance instead:"
                echo "  $0 paper --connector yahoo --symbols AAPL"
                echo ""
                echo "Get free API key at: https://polygon.io"
                echo ""
                exit 1
            fi
            echo -e "${GREEN}✓ Polygon API key found${NC}"
        fi
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
echo "  Connector:  $CONNECTOR"
echo "  Symbols:    $SYMBOLS"
echo "  Capital:    \$${CAPITAL}"
echo "  Strategies: $STRATEGIES"
echo "  Interval:   ${UPDATE_INTERVAL}s"
echo "  Dashboard:  $ENABLE_DASHBOARD"
echo "  Logging:    $ENABLE_LOGGING"
echo "  QDB:        $ENABLE_QDB"
if [ "$ENABLE_QDB" = true ]; then
    echo "  QDB Path:   $QDB_PATH"
    echo "  QDB Version: $QDB_DATA_VERSION"
    echo "  QDB Optimized: $QDB_OPTIMIZED"
fi
if [ "$ENABLE_RESEARCH" = true ]; then
    echo "  Research:   Enabled"
    echo "  Research Mode: $RESEARCH_MODE"
    echo "  Research Output: $RESEARCH_OUTPUT"
fi
if [ "$MODE" = "complete-flow" ]; then
    echo "  Risk Model: $RISK_MODEL"
    echo "  MC Paths:   $MONTE_CARLO_PATHS"
    if [ -n "$SLIPPAGE_IMPL" ]; then
        echo "  Slippage Impl: $SLIPPAGE_IMPL"
    fi
fi
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
        complete-flow)
            echo "  python3 -c 'from Execution.engine.integrated_trading_flow import IntegratedTradingFlow; ...'"
            ;;
        benchmark-slippage)
            echo "  python3 Monitoring/benchmarks/benchmark_slippage.py"
            ;;
    esac
    exit 0
fi

# Check prerequisites
check_system_built

# Initialize QDB if enabled
init_qdb

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
        cat > /tmp/run_paper_trading.py << PYTHON_SCRIPT
import asyncio
import sys
import os
from datetime import datetime

# Add project to path
sys.path.insert(0, os.getcwd())

from Execution.trading.trading_engine import RealTimeTradingEngine
from Data.connectors.alpaca_connector import AlpacaConnector
from Execution.strategies.strategy_registry import get_strategy

# Import strategy adapters to register strategies
try:
    from Execution.strategies.strategy_adapters import MomentumStrategyAdapter, MeanReversionStrategyAdapter
    print("✓ Strategy adapters loaded")
except ImportError as e:
    print(f"⚠️  Strategy adapters not available: {e}")

# Select connector based on CONNECTOR environment variable
CONNECTOR_TYPE = os.environ.get('CONNECTOR', 'alpaca').lower()
print(f"Using connector: {CONNECTOR_TYPE}")

if CONNECTOR_TYPE == 'yahoo':
    try:
        from Data.connectors.yahoo_connector import YahooFinanceConnector
        print("✓ Yahoo Finance connector loaded")
    except ImportError as e:
        print(f"❌ Yahoo Finance connector not available: {e}")
        print("Please install: pip install yfinance")
        sys.exit(1)
elif CONNECTOR_TYPE == 'binance':
    try:
        from Data.connectors.binance_connector import BinanceConnector
        print("✓ Binance connector loaded")
    except ImportError as e:
        print(f"❌ Binance connector not available: {e}")
        print("Please install: pip install python-binance")
        sys.exit(1)
elif CONNECTOR_TYPE == 'polygon':
    try:
        from Data.connectors.polygon_connector import PolygonConnector
        print("✓ Polygon.io connector loaded")
    except ImportError as e:
        print(f"❌ Polygon.io connector not available: {e}")
        print("Please install: pip install websockets")
        sys.exit(1)
elif CONNECTOR_TYPE == 'alphavantage':
    try:
        from Data.connectors.alphavantage_connector import AlphaVantageConnector
        print("✓ Alpha Vantage connector loaded")
    except ImportError as e:
        print(f"❌ Alpha Vantage connector not available: {e}")
        sys.exit(1)
elif CONNECTOR_TYPE == 'coinbase':
    try:
        from Data.connectors.coinbase_connector import CoinbaseProConnector
        print("✓ Coinbase Pro connector loaded")
    except ImportError as e:
        print(f"❌ Coinbase Pro connector not available: {e}")
        print("Please install: pip install websockets")
        sys.exit(1)
elif CONNECTOR_TYPE == 'iexcloud':
    try:
        from Data.connectors.iexcloud_connector import IEXCloudConnector
        print("✓ IEX Cloud connector loaded")
    except ImportError as e:
        print(f"❌ IEX Cloud connector not available: {e}")
        print("Please install: pip install websockets")
        sys.exit(1)
else:  # Default to Alpaca
    try:
        from Data.connectors.alpaca_connector import AlpacaConnector
        print("✓ Alpaca connector loaded")
    except ImportError as e:
        print(f"❌ Alpaca connector not available: {e}")
        sys.exit(1)

# QDB integration
ENABLE_QDB = os.environ.get('ENABLE_QDB', 'true').lower() == 'true'
QDB_PATH = os.environ.get('QDB_PATH', './Data/datasets/qdb')
QDB_DATA_VERSION = os.environ.get('QDB_DATA_VERSION', 'qdb_' + datetime.now().strftime('%Y%m%d'))
QDB_BUFFER_SIZE = int(os.environ.get('QDB_BUFFER_SIZE', '1000'))
QDB_OPTIMIZED = os.environ.get('QDB_OPTIMIZED', 'false').lower() == 'true'

if ENABLE_QDB:
    try:
        from Data.qdb import create_qdb
        from Data.qdb.ingestion import RealtimeCollector
        
        # Use optimized indexer if requested
        if QDB_OPTIMIZED:
            try:
                from Data.qdb.improved_optimized_indexer import ImprovedOptimizedIndexer
                from Data.qdb.cache import DataCache, CacheConfig
                from Data.qdb.versioning import DataVersioning
                
                # Create QDB with optimized indexer
                import sys
                from pathlib import Path
                base_path = Path(QDB_PATH)
                base_path.mkdir(parents=True, exist_ok=True)
                
                # Create a custom QDB-like wrapper
                class OptimizedQDB:
                    def __init__(self):
                        self.indexer = ImprovedOptimizedIndexer(base_path=str(base_path))
                        self.cache = DataCache(config=CacheConfig())
                        self.versioning = DataVersioning(base_path=str(base_path))
                        self.base_path = base_path
                    
                    def store(self, symbol, df, **kwargs):
                        # Use original indexer for storing (compatibility)
                        from Data.qdb.indexer import DataIndexer
                        original_indexer = DataIndexer(base_path=str(base_path))
                        file_path = original_indexer.add_data(symbol, df, kwargs.get('data_version', '1.0'))
                        # Update optimized index
                        self.indexer.add_data(symbol, df.index.min(), df.index.max(), file_path)
                        return file_path
                    
                    def load(self, symbol, start=None, end=None, use_cache=True):
                        from datetime import datetime
                        start_time = pd.to_datetime(start) if start else None
                        end_time = pd.to_datetime(end) if end else None
                        return self.indexer.load_parallel(symbol, start_time, end_time)
                    
                    def sample(self, symbol, window=1000, start=None, end=None):
                        df = self.load(symbol, start, end)
                        if len(df) > window:
                            return df.sample(n=window).sort_index()
                        return df
                
                qdb = OptimizedQDB()
                print(f"✓ QDB initialized with OPTIMIZED indexer: {QDB_PATH}")
                print(f"  Data version: {QDB_DATA_VERSION}")
                print(f"  Performance: O(log n) indexing, parallel loading")
            except ImportError:
                print("⚠️  Optimized indexer not available, using standard QDB")
                qdb = create_qdb(base_path=QDB_PATH)
                print(f"✓ QDB initialized: {QDB_PATH}")
                print(f"  Data version: {QDB_DATA_VERSION}")
        else:
            qdb = create_qdb(base_path=QDB_PATH)
            print(f"✓ QDB initialized: {QDB_PATH}")
            print(f"  Data version: {QDB_DATA_VERSION}")
    except Exception as e:
        print(f"⚠️  QDB initialization failed: {e}")
        print("  Continuing without QDB...")
        ENABLE_QDB = False

async def main():
    # Get configuration from environment
    symbols = os.environ.get('SYMBOLS', 'AAPL,MSFT,GOOGL').split(',')
    capital = float(os.environ.get('CAPITAL', '100000'))
    strategy_names = os.environ.get('STRATEGIES', 'momentum').split(',')
    interval = float(os.environ.get('UPDATE_INTERVAL', '5'))

    print(f"Symbols: {symbols}")
    print(f"Capital: \${capital:,.2f}")
    print(f"Strategies: {strategy_names}")
    if ENABLE_QDB:
        print(f"QDB: Enabled (version: {QDB_DATA_VERSION})")
    
    # Research Framework integration
    ENABLE_RESEARCH = os.environ.get('ENABLE_RESEARCH', 'false').lower() == 'true'
    RESEARCH_MODE = os.environ.get('RESEARCH_MODE', 'full')
    RESEARCH_OUTPUT = os.environ.get('RESEARCH_OUTPUT', './results/research')
    
    if ENABLE_RESEARCH:
        print(f"Research Framework: Enabled (mode: {RESEARCH_MODE})")
        try:
            from Research import CompleteResearchFramework
            from pathlib import Path
            import pandas as pd
            
            # Create output directory
            Path(RESEARCH_OUTPUT).mkdir(parents=True, exist_ok=True)
            
            # Run research framework before trading
            print("\\n" + "="*80)
            print("Running Research Framework...")
            print("="*80)
            
            # Load market data from QDB if available
            if ENABLE_QDB and qdb:
                market_data = {}
                forward_returns = None
                
                for symbol in symbols:
                    try:
                        df = qdb.load(symbol=symbol, start=None, end=None)
                        if len(df) > 0:
                            if 'prices' not in market_data:
                                market_data['prices'] = df['last_price'] if 'last_price' in df.columns else df.get('close', df.iloc[:, 0])
                            if 'bid_prices' not in market_data and 'bid_price' in df.columns:
                                market_data['bid_prices'] = df['bid_price']
                            if 'ask_prices' not in market_data and 'ask_price' in df.columns:
                                market_data['ask_prices'] = df['ask_price']
                            if 'bid_sizes' not in market_data and 'bid_size' in df.columns:
                                market_data['bid_sizes'] = df['bid_size']
                            if 'ask_sizes' not in market_data and 'ask_size' in df.columns:
                                market_data['ask_sizes'] = df['ask_size']
                            if 'trades' not in market_data:
                                trades_df = df[df['volume'] > 0] if 'volume' in df.columns else df
                                if len(trades_df) > 0:
                                    market_data['trades'] = trades_df
                            
                            # Calculate returns for validation
                            if 'returns' not in market_data:
                                prices = df['last_price'] if 'last_price' in df.columns else df.get('close', df.iloc[:, 0])
                                returns = prices.pct_change().dropna()
                                market_data['returns'] = returns
                                forward_returns = returns.shift(-1).dropna()
                    except Exception as e:
                        print(f"  ⚠️  Could not load {symbol} from QDB: {e}")
                
                if market_data:
                    framework = CompleteResearchFramework()
                    results = framework.run_complete_research_pipeline(market_data, forward_returns)
                    
                    # Export results
                    framework.export_results(f"{RESEARCH_OUTPUT}/research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                    print(f"\\n✓ Research results exported to {RESEARCH_OUTPUT}")
                else:
                    print("  ⚠️  No market data available for research")
            else:
                print("  ⚠️  QDB not enabled or not available, skipping research")
        except ImportError as e:
            print(f"  ⚠️  Research Framework not available: {e}")
        except Exception as e:
            print(f"  ⚠️  Research Framework error: {e}")
    
    print("")

    # Initialize connector based on type
    if CONNECTOR_TYPE == 'yahoo':
        connector = YahooFinanceConnector(update_interval=interval)
        print("✓ Yahoo Finance connector initialized (free, 15-20min delay)")
    elif CONNECTOR_TYPE == 'binance':
        connector = BinanceConnector(testnet=True)
        print("✓ Binance connector initialized (testnet)")
    elif CONNECTOR_TYPE == 'polygon':
        polygon_key = os.environ.get('POLYGON_API_KEY')
        if not polygon_key:
            print("❌ POLYGON_API_KEY not set")
            sys.exit(1)
        connector = PolygonConnector(api_key=polygon_key)
        print("✓ Polygon.io connector initialized")
    elif CONNECTOR_TYPE == 'alphavantage':
        api_key = os.environ.get('ALPHAVANTAGE_API_KEY')
        if not api_key:
            print("❌ ALPHAVANTAGE_API_KEY not set")
            print("Get free API key at: https://www.alphavantage.co/support/#api-key")
            sys.exit(1)
        connector = AlphaVantageConnector(api_key=api_key, update_interval=interval)
        print("✓ Alpha Vantage connector initialized (500 calls/day free)")
    elif CONNECTOR_TYPE == 'coinbase':
        connector = CoinbaseProConnector(sandbox=True)
        print("✓ Coinbase Pro connector initialized (sandbox)")
    elif CONNECTOR_TYPE == 'iexcloud':
        api_key = os.environ.get('IEXCLOUD_API_KEY')
        if not api_key:
            print("❌ IEXCLOUD_API_KEY not set")
            print("Get free API key at: https://iexcloud.io/")
            sys.exit(1)
        connector = IEXCloudConnector(api_key=api_key, sandbox=True)
        print("✓ IEX Cloud connector initialized (50k messages/month free)")
    else:  # Alpaca (default)
        alpaca_key = os.environ.get('ALPACA_API_KEY')
        alpaca_secret = os.environ.get('ALPACA_API_SECRET')
        if not alpaca_key or not alpaca_secret:
            print("❌ ALPACA_API_KEY or ALPACA_API_SECRET not set")
            sys.exit(1)
        connector = AlpacaConnector(
            api_key=alpaca_key,
            api_secret=alpaca_secret,
            paper=True
        )
        print("✓ Alpaca connector initialized")

    # Initialize QDB realtime collector if enabled
    collector = None
    if ENABLE_QDB:
        try:
            collector = RealtimeCollector(
                connector=connector,
                qdb=qdb,
                buffer_size=QDB_BUFFER_SIZE
            )
            print("✓ QDB realtime collector initialized")
        except Exception as e:
            print(f"⚠️  QDB collector initialization failed: {e}")

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

    # Create engine with Research Framework and Validation enabled
    enable_research = ENABLE_RESEARCH
    research_min_ticks = int(os.environ.get('RESEARCH_MIN_TICKS', '100'))
    enable_validation = os.environ.get('ENABLE_VALIDATION', 'true').lower() == 'true'
    validation_min_ticks = int(os.environ.get('VALIDATION_MIN_TICKS', '50'))
    validation_timeout = float(os.environ.get('VALIDATION_TIMEOUT', '0.5'))
    
    engine = RealTimeTradingEngine(
        connector=connector,
        strategies=strategies,
        initial_capital=capital,
        symbols=symbols,
        update_interval=interval,
        enable_research=enable_research,
        research_min_ticks=research_min_ticks,
        enable_validation=enable_validation,
        validation_min_ticks=validation_min_ticks,
        validation_timeout=validation_timeout
    )

    # Start QDB collector if enabled
    if collector:
        await collector.start(symbols)
        print("✓ QDB realtime collection started")

    # Start trading
    try:
        await engine.start()
    finally:
        # Stop QDB collector
        if collector:
            await collector.stop()
            print("✓ QDB collector stopped")

if __name__ == "__main__":
    asyncio.run(main())
PYTHON_SCRIPT

        # Export configuration
        export SYMBOLS="$SYMBOLS"
        export CAPITAL="$CAPITAL"
        export STRATEGIES="$STRATEGIES"
        export UPDATE_INTERVAL="$UPDATE_INTERVAL"
        export CONNECTOR="$CONNECTOR"
        export ENABLE_QDB="$ENABLE_QDB"
        export QDB_PATH="$QDB_PATH"
        export QDB_DATA_VERSION="$QDB_DATA_VERSION"
        export QDB_BUFFER_SIZE="$QDB_BUFFER_SIZE"
        export ENABLE_RESEARCH="$ENABLE_RESEARCH"
        export RESEARCH_MODE="$RESEARCH_MODE"
        export RESEARCH_OUTPUT="$RESEARCH_OUTPUT"
        export ENABLE_VALIDATION="true"  # Enable quick validation by default
        export VALIDATION_MIN_TICKS="50"
        export VALIDATION_TIMEOUT="0.5"  # 500ms timeout

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
ENABLE_QDB = '$ENABLE_QDB'.lower() == 'true'
QDB_PATH = '$QDB_PATH'
QDB_DATA_VERSION = '$QDB_DATA_VERSION'
QDB_OPTIMIZED = '$QDB_OPTIMIZED'.lower() == 'true'

# Initialize QDB if enabled
if ENABLE_QDB:
    try:
        if QDB_OPTIMIZED:
            try:
                from Data.qdb.improved_optimized_indexer import ImprovedOptimizedIndexer
                indexer = ImprovedOptimizedIndexer(base_path=QDB_PATH)
                print(f'✓ QDB initialized with OPTIMIZED indexer for backtest: {QDB_PATH}')
                print(f'  Using data version: {QDB_DATA_VERSION}')
                print(f'  Performance: O(log n) indexing, parallel loading')
                # Create a simple wrapper for compatibility
                class OptimizedQDBWrapper:
                    def __init__(self, indexer):
                        self.indexer = indexer
                    def load(self, symbol, start=None, end=None):
                        from datetime import datetime
                        import pandas as pd
                        start_time = pd.to_datetime(start) if start else None
                        end_time = pd.to_datetime(end) if end else None
                        return self.indexer.load_parallel(symbol, start_time, end_time)
                qdb = OptimizedQDBWrapper(indexer)
            except ImportError:
                print('⚠️  Optimized indexer not available, using standard QDB')
                from Data.qdb import create_qdb
                qdb = create_qdb(base_path=QDB_PATH)
                print(f'✓ QDB initialized for backtest: {QDB_PATH}')
        else:
            from Data.qdb import create_qdb
            qdb = create_qdb(base_path=QDB_PATH)
            print(f'✓ QDB initialized for backtest: {QDB_PATH}')
            print(f'  Using data version: {QDB_DATA_VERSION}')
    except Exception as e:
        print(f'⚠️  QDB initialization failed: {e}')
        ENABLE_QDB = False

from Environment.backtester.simple_backtester import run_backtest

# Research Framework integration
ENABLE_RESEARCH = '$ENABLE_RESEARCH'.lower() == 'true'
RESEARCH_MODE = '$RESEARCH_MODE'
RESEARCH_OUTPUT = '$RESEARCH_OUTPUT'

if ENABLE_RESEARCH:
    try:
        from Research import CompleteResearchFramework
        from pathlib import Path
        from datetime import datetime
        import pandas as pd
        
        print('\\n' + '='*80)
        print('Running Research Framework (Backtest Mode)...')
        print('='*80)
        
        # Load market data from QDB if available
        if ENABLE_QDB and 'qdb' in locals():
            market_data = {}
            forward_returns = None
            
            for symbol in symbols:
                try:
                    df = qdb.load(symbol=symbol, start=None, end=None)
                    if len(df) > 0:
                        if 'prices' not in market_data:
                            market_data['prices'] = df['last_price'] if 'last_price' in df.columns else df.get('close', df.iloc[:, 0])
                        if 'bid_prices' not in market_data and 'bid_price' in df.columns:
                            market_data['bid_prices'] = df['bid_price']
                        if 'ask_prices' not in market_data and 'ask_price' in df.columns:
                            market_data['ask_prices'] = df['ask_price']
                        if 'bid_sizes' not in market_data and 'bid_size' in df.columns:
                            market_data['bid_sizes'] = df['bid_size']
                        if 'ask_sizes' not in market_data and 'ask_size' in df.columns:
                            market_data['ask_sizes'] = df['ask_size']
                        if 'trades' not in market_data:
                            trades_df = df[df['volume'] > 0] if 'volume' in df.columns else df
                            if len(trades_df) > 0:
                                market_data['trades'] = trades_df
                        
                        # Calculate returns for validation
                        if 'returns' not in market_data:
                            prices = df['last_price'] if 'last_price' in df.columns else df.get('close', df.iloc[:, 0])
                            returns = prices.pct_change().dropna()
                            market_data['returns'] = returns
                            forward_returns = returns.shift(-1).dropna()
                except Exception as e:
                    print(f'  ⚠️  Could not load {symbol} from QDB: {e}')
            
            if market_data:
                Path(RESEARCH_OUTPUT).mkdir(parents=True, exist_ok=True)
                framework = CompleteResearchFramework()
                results = framework.run_complete_research_pipeline(market_data, forward_returns)
                framework.export_results(f'{RESEARCH_OUTPUT}/research_backtest_{datetime.now().strftime(\"%Y%m%d_%H%M%S\")}.json')
                print(f'\\n✓ Research results exported to {RESEARCH_OUTPUT}')
            else:
                print('  ⚠️  No market data available for research')
        else:
            print('  ⚠️  QDB not enabled or not available, skipping research')
    except ImportError as e:
        print(f'  ⚠️  Research Framework not available: {e}')
    except Exception as e:
        print(f'  ⚠️  Research Framework error: {e}')

symbols = '$SYMBOLS'.split(',')
run_backtest(symbols=symbols, capital=$CAPITAL)
" 2>&1 | tee "$LOGFILE"
            else
                python3 -c "
import sys
sys.path.insert(0, '.')
ENABLE_QDB = '$ENABLE_QDB'.lower() == 'true'
QDB_PATH = '$QDB_PATH'
QDB_OPTIMIZED = '$QDB_OPTIMIZED'.lower() == 'true'
if ENABLE_QDB:
    try:
        if QDB_OPTIMIZED:
            try:
                from Data.qdb.improved_optimized_indexer import ImprovedOptimizedIndexer
                indexer = ImprovedOptimizedIndexer(base_path=QDB_PATH)
                class OptimizedQDBWrapper:
                    def __init__(self, indexer):
                        self.indexer = indexer
                    def load(self, symbol, start=None, end=None):
                        from datetime import datetime
                        import pandas as pd
                        start_time = pd.to_datetime(start) if start else None
                        end_time = pd.to_datetime(end) if end else None
                        return self.indexer.load_parallel(symbol, start_time, end_time)
                qdb = OptimizedQDBWrapper(indexer)
            except ImportError:
                from Data.qdb import create_qdb
                qdb = create_qdb(base_path=QDB_PATH)
        else:
            from Data.qdb import create_qdb
            qdb = create_qdb(base_path=QDB_PATH)
    except:
        ENABLE_QDB = False
from Environment.backtester.simple_backtester import run_backtest

# Research Framework integration
ENABLE_RESEARCH = '$ENABLE_RESEARCH'.lower() == 'true'
RESEARCH_MODE = '$RESEARCH_MODE'
RESEARCH_OUTPUT = '$RESEARCH_OUTPUT'

if ENABLE_RESEARCH:
    try:
        from Research import CompleteResearchFramework
        from pathlib import Path
        from datetime import datetime
        import pandas as pd
        
        print('\\n' + '='*80)
        print('Running Research Framework (Backtest Mode)...')
        print('='*80)
        
        # Load market data from QDB if available
        if ENABLE_QDB and 'qdb' in locals():
            market_data = {}
            forward_returns = None
            
            for symbol in symbols:
                try:
                    df = qdb.load(symbol=symbol, start=None, end=None)
                    if len(df) > 0:
                        if 'prices' not in market_data:
                            market_data['prices'] = df['last_price'] if 'last_price' in df.columns else df.get('close', df.iloc[:, 0])
                        if 'bid_prices' not in market_data and 'bid_price' in df.columns:
                            market_data['bid_prices'] = df['bid_price']
                        if 'ask_prices' not in market_data and 'ask_price' in df.columns:
                            market_data['ask_prices'] = df['ask_price']
                        if 'bid_sizes' not in market_data and 'bid_size' in df.columns:
                            market_data['bid_sizes'] = df['bid_size']
                        if 'ask_sizes' not in market_data and 'ask_size' in df.columns:
                            market_data['ask_sizes'] = df['ask_size']
                        if 'trades' not in market_data:
                            trades_df = df[df['volume'] > 0] if 'volume' in df.columns else df
                            if len(trades_df) > 0:
                                market_data['trades'] = trades_df
                        
                        # Calculate returns for validation
                        if 'returns' not in market_data:
                            prices = df['last_price'] if 'last_price' in df.columns else df.get('close', df.iloc[:, 0])
                            returns = prices.pct_change().dropna()
                            market_data['returns'] = returns
                            forward_returns = returns.shift(-1).dropna()
                except Exception as e:
                    print(f'  ⚠️  Could not load {symbol} from QDB: {e}')
            
            if market_data:
                Path(RESEARCH_OUTPUT).mkdir(parents=True, exist_ok=True)
                framework = CompleteResearchFramework()
                results = framework.run_complete_research_pipeline(market_data, forward_returns)
                framework.export_results(f'{RESEARCH_OUTPUT}/research_backtest_{datetime.now().strftime(\"%Y%m%d_%H%M%S\")}.json')
                print(f'\\n✓ Research results exported to {RESEARCH_OUTPUT}')
            else:
                print('  ⚠️  No market data available for research')
        else:
            print('  ⚠️  QDB not enabled or not available, skipping research')
    except ImportError as e:
        print(f'  ⚠️  Research Framework not available: {e}')
    except Exception as e:
        print(f'  ⚠️  Research Framework error: {e}')

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
        cat > /tmp/run_live_trading.py << PYTHON_SCRIPT
import asyncio
import sys
import os
from datetime import datetime

sys.path.insert(0, os.getcwd())

from Execution.trading.trading_engine import RealTimeTradingEngine
from Data.connectors.alpaca_connector import AlpacaConnector
from Execution.strategies.strategy_registry import get_strategy

# Import strategy adapters to register strategies
try:
    from Execution.strategies.strategy_adapters import MomentumStrategyAdapter, MeanReversionStrategyAdapter
    print("✓ Strategy adapters loaded")
except ImportError as e:
    print(f"⚠️  Strategy adapters not available: {e}")

# Select connector based on CONNECTOR environment variable
CONNECTOR_TYPE = os.environ.get('CONNECTOR', 'alpaca').lower()
print(f"Using connector: {CONNECTOR_TYPE}")

if CONNECTOR_TYPE == 'yahoo':
    try:
        from Data.connectors.yahoo_connector import YahooFinanceConnector
        print("✓ Yahoo Finance connector loaded")
    except ImportError as e:
        print(f"❌ Yahoo Finance connector not available: {e}")
        print("Please install: pip install yfinance")
        sys.exit(1)
elif CONNECTOR_TYPE == 'binance':
    try:
        from Data.connectors.binance_connector import BinanceConnector
        print("✓ Binance connector loaded")
    except ImportError as e:
        print(f"❌ Binance connector not available: {e}")
        print("Please install: pip install python-binance")
        sys.exit(1)
elif CONNECTOR_TYPE == 'polygon':
    try:
        from Data.connectors.polygon_connector import PolygonConnector
        print("✓ Polygon.io connector loaded")
    except ImportError as e:
        print(f"❌ Polygon.io connector not available: {e}")
        print("Please install: pip install websockets")
        sys.exit(1)
elif CONNECTOR_TYPE == 'alphavantage':
    try:
        from Data.connectors.alphavantage_connector import AlphaVantageConnector
        print("✓ Alpha Vantage connector loaded")
    except ImportError as e:
        print(f"❌ Alpha Vantage connector not available: {e}")
        sys.exit(1)
elif CONNECTOR_TYPE == 'coinbase':
    try:
        from Data.connectors.coinbase_connector import CoinbaseProConnector
        print("✓ Coinbase Pro connector loaded")
    except ImportError as e:
        print(f"❌ Coinbase Pro connector not available: {e}")
        print("Please install: pip install websockets")
        sys.exit(1)
elif CONNECTOR_TYPE == 'iexcloud':
    try:
        from Data.connectors.iexcloud_connector import IEXCloudConnector
        print("✓ IEX Cloud connector loaded")
    except ImportError as e:
        print(f"❌ IEX Cloud connector not available: {e}")
        print("Please install: pip install websockets")
        sys.exit(1)
else:  # Default to Alpaca
    try:
        from Data.connectors.alpaca_connector import AlpacaConnector
        print("✓ Alpaca connector loaded")
    except ImportError as e:
        print(f"❌ Alpaca connector not available: {e}")
        sys.exit(1)

# QDB integration
ENABLE_QDB = os.environ.get('ENABLE_QDB', 'true').lower() == 'true'
QDB_PATH = os.environ.get('QDB_PATH', './Data/datasets/qdb')
QDB_DATA_VERSION = os.environ.get('QDB_DATA_VERSION', 'qdb_' + datetime.now().strftime('%Y%m%d'))
QDB_BUFFER_SIZE = int(os.environ.get('QDB_BUFFER_SIZE', '1000'))
QDB_OPTIMIZED = os.environ.get('QDB_OPTIMIZED', 'false').lower() == 'true'

if ENABLE_QDB:
    try:
        from Data.qdb import create_qdb
        from Data.qdb.ingestion import RealtimeCollector
        
        # Use optimized indexer if requested
        if QDB_OPTIMIZED:
            try:
                from Data.qdb.improved_optimized_indexer import ImprovedOptimizedIndexer
                from Data.qdb.cache import DataCache, CacheConfig
                from Data.qdb.versioning import DataVersioning
                from pathlib import Path
                import pandas as pd
                
                base_path = Path(QDB_PATH)
                base_path.mkdir(parents=True, exist_ok=True)
                
                class OptimizedQDB:
                    def __init__(self):
                        self.indexer = ImprovedOptimizedIndexer(base_path=str(base_path))
                        self.cache = DataCache(config=CacheConfig())
                        self.versioning = DataVersioning(base_path=str(base_path))
                        self.base_path = base_path
                    
                    def store(self, symbol, df, **kwargs):
                        from Data.qdb.indexer import DataIndexer
                        original_indexer = DataIndexer(base_path=str(base_path))
                        file_path = original_indexer.add_data(symbol, df, kwargs.get('data_version', '1.0'))
                        self.indexer.add_data(symbol, df.index.min(), df.index.max(), file_path)
                        return file_path
                    
                    def load(self, symbol, start=None, end=None, use_cache=True):
                        start_time = pd.to_datetime(start) if start else None
                        end_time = pd.to_datetime(end) if end else None
                        return self.indexer.load_parallel(symbol, start_time, end_time)
                    
                    def sample(self, symbol, window=1000, start=None, end=None):
                        df = self.load(symbol, start, end)
                        if len(df) > window:
                            return df.sample(n=window).sort_index()
                        return df
                
                qdb = OptimizedQDB()
                print(f"✓ QDB initialized with OPTIMIZED indexer: {QDB_PATH}")
                print(f"  Performance: O(log n) indexing, parallel loading")
            except ImportError:
                print("⚠️  Optimized indexer not available, using standard QDB")
                qdb = create_qdb(base_path=QDB_PATH)
                print(f"✓ QDB initialized: {QDB_PATH}")
        else:
            qdb = create_qdb(base_path=QDB_PATH)
            print(f"✓ QDB initialized: {QDB_PATH}")
    except Exception as e:
        print(f"⚠️  QDB initialization failed: {e}")
        ENABLE_QDB = False

async def main():
    symbols = os.environ.get('SYMBOLS', 'AAPL').split(',')
    capital = float(os.environ.get('CAPITAL', '10000'))
    strategy_names = os.environ.get('STRATEGIES', 'momentum').split(',')
    interval = float(os.environ.get('UPDATE_INTERVAL', '5'))

    print("⚠️  LIVE TRADING - USING REAL MONEY")
    print(f"Symbols: {symbols}")
    print(f"Capital: \${capital:,.2f}")
    if ENABLE_QDB:
        print(f"QDB: Enabled (version: {QDB_DATA_VERSION})")
        if QDB_OPTIMIZED:
            print(f"QDB Optimized: Enabled (O(log n) indexing)")
    print("")

    # Initialize connector (LIVE mode)
    connector = AlpacaConnector(
        api_key=os.environ['ALPACA_API_KEY'],
        api_secret=os.environ['ALPACA_API_SECRET'],
        paper=False  # LIVE TRADING
    )

    # Initialize QDB collector if enabled
    collector = None
    if ENABLE_QDB:
        try:
            collector = RealtimeCollector(
                connector=connector,
                qdb=qdb,
                buffer_size=QDB_BUFFER_SIZE
            )
        except Exception as e:
            print(f"⚠️  QDB collector failed: {e}")

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

    # Start QDB collector if enabled
    if collector:
        await collector.start(symbols)

    try:
        await engine.start()
    finally:
        if collector:
            await collector.stop()

if __name__ == "__main__":
    asyncio.run(main())
PYTHON_SCRIPT

        export SYMBOLS="$SYMBOLS"
        export CAPITAL="$CAPITAL"
        export STRATEGIES="$STRATEGIES"
        export UPDATE_INTERVAL="$UPDATE_INTERVAL"
        export ENABLE_QDB="$ENABLE_QDB"
        export QDB_PATH="$QDB_PATH"
        export QDB_DATA_VERSION="$QDB_DATA_VERSION"
        export QDB_BUFFER_SIZE="$QDB_BUFFER_SIZE"
        export QDB_OPTIMIZED="$QDB_OPTIMIZED"
        export ENABLE_RESEARCH="$ENABLE_RESEARCH"
        export RESEARCH_MODE="$RESEARCH_MODE"
        export RESEARCH_OUTPUT="$RESEARCH_OUTPUT"

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
            # Export QDB config for demo
            export ENABLE_QDB="$ENABLE_QDB"
            export QDB_PATH="$QDB_PATH"
            export QDB_DATA_VERSION="$QDB_DATA_VERSION"
            export QDB_OPTIMIZED="$QDB_OPTIMIZED"
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

    complete-flow)
        echo -e "${GREEN}============================================================${NC}"
        echo -e "${GREEN}Starting COMPLETE TRADING FLOW${NC}"
        echo -e "${GREEN}(EDA + Strategy Comparison + Risk + Position Management)${NC}"
        echo -e "${GREEN}============================================================${NC}"
        echo ""

        echo -e "${BLUE}Running complete trading flow...${NC}"
        echo ""

        if [ "$ENABLE_LOGGING" = true ]; then
            LOGFILE="$LOGS_DIR/complete_flow_$(date +%Y%m%d_%H%M%S).log"
            python3 -c "
import sys
sys.path.insert(0, '.')
from Execution.engine.integrated_trading_flow import IntegratedTradingFlow
from Execution.risk_control.portfolio_manager import RiskModel
from Execution.engine.complete_trading_flow import create_sample_strategies
from Execution.engine.pipeline import create_sample_data
force_slippage_impl = '${SLIPPAGE_IMPL}'
force_slippage_impl = force_slippage_impl if force_slippage_impl else None
import pandas as pd

# QDB integration
ENABLE_QDB = '$ENABLE_QDB'.lower() == 'true'
QDB_PATH = '$QDB_PATH'
QDB_DATA_VERSION = '$QDB_DATA_VERSION'
QDB_OPTIMIZED = '$QDB_OPTIMIZED'.lower() == 'true'

if ENABLE_QDB:
    try:
        if QDB_OPTIMIZED:
            try:
                from Data.qdb.improved_optimized_indexer import ImprovedOptimizedIndexer
                indexer = ImprovedOptimizedIndexer(base_path=QDB_PATH)
                print(f'✓ QDB initialized with OPTIMIZED indexer for complete flow: {QDB_PATH}')
                print(f'  Data version: {QDB_DATA_VERSION}')
                print(f'  Performance: O(log n) indexing, parallel loading')
                class OptimizedQDBWrapper:
                    def __init__(self, indexer):
                        self.indexer = indexer
                    def load(self, symbol, start=None, end=None):
                        from datetime import datetime
                        import pandas as pd
                        start_time = pd.to_datetime(start) if start else None
                        end_time = pd.to_datetime(end) if end else None
                        return self.indexer.load_parallel(symbol, start_time, end_time)
                qdb = OptimizedQDBWrapper(indexer)
            except ImportError:
                print('⚠️  Optimized indexer not available, using standard QDB')
                from Data.qdb import create_qdb
                qdb = create_qdb(base_path=QDB_PATH)
                print(f'✓ QDB initialized for complete flow: {QDB_PATH}')
                print(f'  Data version: {QDB_DATA_VERSION}')
        else:
            from Data.qdb import create_qdb
            qdb = create_qdb(base_path=QDB_PATH)
            print(f'✓ QDB initialized for complete flow: {QDB_PATH}')
            print(f'  Data version: {QDB_DATA_VERSION}')
    except Exception as e:
        print(f'⚠️  QDB initialization failed: {e}')
        ENABLE_QDB = False

# Parse risk model
risk_model_map = {
    'equal_weight': RiskModel.EQUAL_WEIGHT,
    'inverse_volatility': RiskModel.INVERSE_VOLATILITY,
    'mean_variance': RiskModel.MEAN_VARIANCE,
    'risk_parity': RiskModel.RISK_PARITY,
    'black_litterman': RiskModel.BLACK_LITTERMAN,
    'hrp': RiskModel.HIERARCHICAL_RISK_PARITY
}
risk_model = risk_model_map.get('$RISK_MODEL', RiskModel.RISK_PARITY)

# Create flow
flow = IntegratedTradingFlow(
    initial_capital=$CAPITAL,
    risk_model=risk_model,
    monte_carlo_paths=$MONTE_CARLO_PATHS
)

# Create sample data
data = create_sample_data(n_records=1000)
data['close'] = data['price']

# 使用所有策略进行对比（默认行为）
strategies = None
print('Using all available strategies for comprehensive comparison...')

# Get symbols
symbols = '$SYMBOLS'.split(',')

# Research Framework integration
ENABLE_RESEARCH = '$ENABLE_RESEARCH'.lower() == 'true'
RESEARCH_MODE = '$RESEARCH_MODE'
RESEARCH_OUTPUT = '$RESEARCH_OUTPUT'

if ENABLE_RESEARCH:
    try:
        from Research import CompleteResearchFramework
        from pathlib import Path
        from datetime import datetime
        
        print('\\n' + '='*80)
        print('Running Research Framework (Complete Flow Mode)...')
        print('='*80)
        
        # Prepare market data from sample data or QDB
        market_data = {}
        forward_returns = None
        
        if ENABLE_QDB and 'qdb' in locals():
            # Try to load from QDB first
            for symbol in symbols[:3]:
                try:
                    df = qdb.load(symbol=symbol, start=None, end=None)
                    if len(df) > 0:
                        if 'prices' not in market_data:
                            market_data['prices'] = df['last_price'] if 'last_price' in df.columns else df.get('close', df.iloc[:, 0])
                        if 'bid_prices' not in market_data and 'bid_price' in df.columns:
                            market_data['bid_prices'] = df['bid_price']
                        if 'ask_prices' not in market_data and 'ask_price' in df.columns:
                            market_data['ask_prices'] = df['ask_price']
                        if 'bid_sizes' not in market_data and 'bid_size' in df.columns:
                            market_data['bid_sizes'] = df['bid_size']
                        if 'ask_sizes' not in market_data and 'ask_size' in df.columns:
                            market_data['ask_sizes'] = df['ask_size']
                        if 'trades' not in market_data:
                            trades_df = df[df['volume'] > 0] if 'volume' in df.columns else df
                            if len(trades_df) > 0:
                                market_data['trades'] = trades_df
                        
                        if 'returns' not in market_data:
                            prices = df['last_price'] if 'last_price' in df.columns else df.get('close', df.iloc[:, 0])
                            returns = prices.pct_change().dropna()
                            market_data['returns'] = returns
                            forward_returns = returns.shift(-1).dropna()
                except Exception as e:
                    print(f'  ⚠️  Could not load {symbol} from QDB: {e}')
        
        # Fallback to sample data if QDB not available
        if not market_data and 'data' in locals():
            market_data['prices'] = data['close'] if 'close' in data.columns else data['price']
            market_data['returns'] = market_data['prices'].pct_change().dropna()
            forward_returns = market_data['returns'].shift(-1).dropna()
        
        if market_data:
            Path(RESEARCH_OUTPUT).mkdir(parents=True, exist_ok=True)
            framework = CompleteResearchFramework()
            results = framework.run_complete_research_pipeline(market_data, forward_returns)
            framework.export_results(f'{RESEARCH_OUTPUT}/research_complete_flow_{datetime.now().strftime(\"%Y%m%d_%H%M%S\")}.json')
            print(f'\\n✓ Research results exported to {RESEARCH_OUTPUT}')
        else:
            print('  ⚠️  No market data available for research')
    except ImportError as e:
        print(f'  ⚠️  Research Framework not available: {e}')
    except Exception as e:
        print(f'  ⚠️  Research Framework error: {e}')

# Run complete flow
result = flow.execute_complete_flow_with_position_management(
    data=data,
    strategies=strategies,
    symbols=symbols[:3] if len(symbols) > 3 else symbols,
    force_slippage_impl=force_slippage_impl
)

print('\\nComplete flow finished successfully!')
print('\\nAll reports saved to results/ directory')
" 2>&1 | tee "$LOGFILE"
        else
            python3 -c "
import sys
sys.path.insert(0, '.')
from Execution.engine.integrated_trading_flow import IntegratedTradingFlow
from Execution.risk_control.portfolio_manager import RiskModel
from Execution.engine.complete_trading_flow import create_sample_strategies
from Execution.engine.pipeline import create_sample_data
force_slippage_impl = '${SLIPPAGE_IMPL}'
force_slippage_impl = force_slippage_impl if force_slippage_impl else None

# QDB integration
ENABLE_QDB = '$ENABLE_QDB'.lower() == 'true'
QDB_PATH = '$QDB_PATH'
QDB_OPTIMIZED = '$QDB_OPTIMIZED'.lower() == 'true'
if ENABLE_QDB:
    try:
        if QDB_OPTIMIZED:
            try:
                from Data.qdb.improved_optimized_indexer import ImprovedOptimizedIndexer
                indexer = ImprovedOptimizedIndexer(base_path=QDB_PATH)
                class OptimizedQDBWrapper:
                    def __init__(self, indexer):
                        self.indexer = indexer
                    def load(self, symbol, start=None, end=None):
                        from datetime import datetime
                        import pandas as pd
                        start_time = pd.to_datetime(start) if start else None
                        end_time = pd.to_datetime(end) if end else None
                        return self.indexer.load_parallel(symbol, start_time, end_time)
                qdb = OptimizedQDBWrapper(indexer)
            except ImportError:
                from Data.qdb import create_qdb
                qdb = create_qdb(base_path=QDB_PATH)
        else:
            from Data.qdb import create_qdb
            qdb = create_qdb(base_path=QDB_PATH)
    except:
        ENABLE_QDB = False

risk_model_map = {
    'equal_weight': RiskModel.EQUAL_WEIGHT,
    'inverse_volatility': RiskModel.INVERSE_VOLATILITY,
    'mean_variance': RiskModel.MEAN_VARIANCE,
    'risk_parity': RiskModel.RISK_PARITY,
    'black_litterman': RiskModel.BLACK_LITTERMAN,
    'hrp': RiskModel.HIERARCHICAL_RISK_PARITY
}
risk_model = risk_model_map.get('$RISK_MODEL', RiskModel.RISK_PARITY)

flow = IntegratedTradingFlow(
    initial_capital=$CAPITAL,
    risk_model=risk_model,
    monte_carlo_paths=$MONTE_CARLO_PATHS
)

data = create_sample_data(n_records=1000)
data['close'] = data['price']

# 使用所有策略进行对比（默认行为）
strategies = None
print('Using all available strategies for comprehensive comparison...')

symbols = '$SYMBOLS'.split(',')

result = flow.execute_complete_flow_with_position_management(
    data=data,
    strategies=strategies,
    symbols=symbols[:3] if len(symbols) > 3 else symbols,
    force_slippage_impl=force_slippage_impl
)

print('\\nComplete flow finished successfully!')
print('\\nAll reports saved to results/ directory')
"
        fi
        ;;

    benchmark-slippage)
        echo -e "${GREEN}============================================================${NC}"
        echo -e "${GREEN}Running SLIPPAGE PERFORMANCE BENCHMARK${NC}"
        echo -e "${GREEN}============================================================${NC}"
        echo ""

        if [ -f "Monitoring/benchmarks/benchmark_slippage.py" ]; then
            if [ "$ENABLE_LOGGING" = true ]; then
                LOGFILE="$LOGS_DIR/slippage_benchmark_$(date +%Y%m%d_%H%M%S).log"
                python3 Monitoring/benchmarks/benchmark_slippage.py 2>&1 | tee "$LOGFILE"
            else
                python3 Monitoring/benchmarks/benchmark_slippage.py
            fi
        else
            echo -e "${RED}Error: benchmark_slippage.py not found${NC}"
            exit 1
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
