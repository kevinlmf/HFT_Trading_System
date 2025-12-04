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
RESULTS_DIR="./results"

# Auto-load QDB data configuration
AUTO_LOAD_DATA=true  # Automatically download historical data to QDB if missing (default: true)

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
        --auto-load-data)
            AUTO_LOAD_DATA=true
            shift
            ;;
        --no-auto-load-data)
            AUTO_LOAD_DATA=false
            shift
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
  --auto-load-data
                 Automatically download historical data to QDB if missing (default: enabled)
  --no-auto-load-data
                 Disable automatic data loading
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
        
        # Check if QDB module exists (updated to new QDB package layout)
        # Old path was Data/qdb/qdb.py; now we use the top-level QDB package.
        if [ ! -f "QDB/qdb.py" ]; then
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
    from QDB import create_qdb
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

# Check and auto-load QDB data if needed (for complete-flow mode)
check_and_load_qdb_data() {
    if [ "$MODE" != "complete-flow" ] || [ "$ENABLE_QDB" != true ] || [ "$AUTO_LOAD_DATA" != true ]; then
        return 0
    fi
    
    if [ -z "$SYMBOLS" ]; then
        return 0
    fi
    
    echo -e "${BLUE}Checking QDB data availability...${NC}"
    
    # Check if data exists for the requested symbols
    result=$(python3 -c "
import sys
sys.path.insert(0, '.')
try:
    from QDB import create_qdb
    qdb = create_qdb(base_path='$QDB_PATH')
    symbols = '$SYMBOLS'.split(',')
    missing_symbols = []
    
    for symbol in symbols:
        symbol = symbol.strip()
        if not symbol:
            continue
        try:
            df = qdb.load(symbol=symbol, start=None, end=None)
            if df is None or len(df) == 0:
                missing_symbols.append(symbol)
        except Exception:
            missing_symbols.append(symbol)
    
    if missing_symbols:
        print('MISSING:' + ','.join(missing_symbols))
    else:
        print('OK')
except Exception as e:
    print(f'ERROR:{e}')
" 2>/dev/null)
    
    if [[ "$result" == "OK" ]]; then
        echo -e "${GREEN}✓ QDB data available for all symbols${NC}"
        return 0
    elif [[ "$result" == MISSING:* ]]; then
        missing_symbols="${result#MISSING:}"
        echo -e "${YELLOW}⚠️  QDB data missing for: $missing_symbols${NC}"
        echo -e "${BLUE}Auto-loading historical data from Yahoo Finance...${NC}"
        
        # Check if load script exists
        if [ ! -f "Market_Data/load_historical_data_to_qdb.py" ]; then
            echo -e "${YELLOW}⚠️  Data loading script not found at Market_Data/load_historical_data_to_qdb.py${NC}"
            echo -e "${YELLOW}   Please run manually: python Market_Data/load_historical_data_to_qdb.py --symbols $missing_symbols --days 365${NC}"
            return 1
        fi
        
        # Auto-load data (download last 365 days by default)
        echo -e "${BLUE}Downloading last 365 days of daily data...${NC}"
        python3 Market_Data/load_historical_data_to_qdb.py \
            --symbols "$missing_symbols" \
            --days 365 \
            --interval 1d \
            --qdb-path "$QDB_PATH" \
            --data-version "$QDB_DATA_VERSION" 2>&1 | grep -E "(✓|✗|⚠️|下载|存储|成功|失败|处理|正在)" || true
        
        # Verify data was loaded
        verify_result=$(python3 -c "
import sys
sys.path.insert(0, '.')
from QDB import create_qdb
qdb = create_qdb(base_path='$QDB_PATH')
symbols = '$missing_symbols'.split(',')
all_loaded = True
for symbol in symbols:
    symbol = symbol.strip()
    if not symbol:
        continue
    try:
        df = qdb.load(symbol=symbol, start=None, end=None)
        if df is None or len(df) == 0:
            all_loaded = False
            break
    except Exception:
        all_loaded = False
        break
sys.exit(0 if all_loaded else 1)
" 2>/dev/null)
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ Historical data loaded successfully${NC}"
        else
            echo -e "${YELLOW}⚠️  Some data may not have loaded correctly, continuing with sample data${NC}"
        fi
    elif [[ "$result" == ERROR:* ]]; then
        echo -e "${YELLOW}⚠️  Could not check QDB data: ${result#ERROR:}${NC}"
    fi
}

# Check and auto-load QDB data if needed
check_and_load_qdb_data

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
from Market_Data.alpaca_connector import AlpacaConnector
from Strategy_Construction.strategy_registry import get_strategy

# Import strategy adapters to register strategies
try:
    from Strategy_Construction.strategy_adapters import MomentumStrategyAdapter, MeanReversionStrategyAdapter
    print("✓ Strategy adapters loaded")
except ImportError as e:
    print(f"⚠️  Strategy adapters not available: {e}")

# Select connector based on CONNECTOR environment variable
CONNECTOR_TYPE = os.environ.get('CONNECTOR', 'alpaca').lower()
print(f"Using connector: {CONNECTOR_TYPE}")

if CONNECTOR_TYPE == 'yahoo':
    try:
        from Market_Data.yahoo_connector import YahooFinanceConnector
        print("✓ Yahoo Finance connector loaded")
    except ImportError as e:
        print(f"❌ Yahoo Finance connector not available: {e}")
        print("Please install: pip install yfinance")
        sys.exit(1)
elif CONNECTOR_TYPE == 'binance':
    try:
        from Market_Data.binance_connector import BinanceConnector
        print("✓ Binance connector loaded")
    except ImportError as e:
        print(f"❌ Binance connector not available: {e}")
        print("Please install: pip install python-binance")
        sys.exit(1)
elif CONNECTOR_TYPE == 'polygon':
    try:
        from Market_Data.polygon_connector import PolygonConnector
        print("✓ Polygon.io connector loaded")
    except ImportError as e:
        print(f"❌ Polygon.io connector not available: {e}")
        print("Please install: pip install websockets")
        sys.exit(1)
elif CONNECTOR_TYPE == 'alphavantage':
    try:
        from Market_Data.alphavantage_connector import AlphaVantageConnector
        print("✓ Alpha Vantage connector loaded")
    except ImportError as e:
        print(f"❌ Alpha Vantage connector not available: {e}")
        sys.exit(1)
elif CONNECTOR_TYPE == 'coinbase':
    try:
        from Market_Data.coinbase_connector import CoinbaseProConnector
        print("✓ Coinbase Pro connector loaded")
    except ImportError as e:
        print(f"❌ Coinbase Pro connector not available: {e}")
        print("Please install: pip install websockets")
        sys.exit(1)
elif CONNECTOR_TYPE == 'iexcloud':
    try:
        from Market_Data.iexcloud_connector import IEXCloudConnector
        print("✓ IEX Cloud connector loaded")
    except ImportError as e:
        print(f"❌ IEX Cloud connector not available: {e}")
        print("Please install: pip install websockets")
        sys.exit(1)
else:  # Default to Alpaca
    try:
        from Market_Data.alpaca_connector import AlpacaConnector
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
        from QDB import create_qdb
        from QDB.ingestion import RealtimeCollector
        
        # Use optimized indexer if requested
        if QDB_OPTIMIZED:
            try:
                from QDB.improved_optimized_indexer import ImprovedOptimizedIndexer
                from QDB.cache import DataCache, CacheConfig
                from QDB.versioning import DataVersioning
                
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
                        from QDB.indexer import DataIndexer
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
    
    # Ensure results directory exists
    results_dir = os.environ.get('RESULTS_DIR', './results')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(f'{results_dir}/strategies', exist_ok=True)
    os.makedirs(f'{results_dir}/research', exist_ok=True)
    print(f'Results will be saved to: {os.path.abspath(results_dir)}')

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
            from Microstructure_Analysis.microstructure_profiling import MicrostructureProfiler
            from Alpha_Modeling.factor_hypothesis import FactorHypothesisGenerator
            from Alpha_Modeling.statistical_validation import StatisticalValidator
            from Alpha_Modeling.ml_validation import MLValidator
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
                    # Run research pipeline manually
                    profiler = MicrostructureProfiler()
                    profile_results = profiler.analyze(market_data)
                    
                    generator = FactorHypothesisGenerator()
                    factors = generator.generate(profile_results)
                    
                    validator = StatisticalValidator()
                    validation_results = validator.validate(factors, forward_returns if forward_returns is not None else market_data.get('returns', pd.Series()))
                    
                    # Export results
                    import json
                    research_results = {
                        'profiling': profile_results,
                        'factors': [f.__dict__ for f in factors] if factors else [],
                        'validation': validation_results.__dict__ if hasattr(validation_results, '__dict__') else str(validation_results),
                        'timestamp': datetime.now().isoformat()
                    }
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    output_file = f"{RESEARCH_OUTPUT}/research_{timestamp}.json"
                    with open(output_file, 'w') as f:
                        json.dump(research_results, f, indent=2, default=str)
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
        export RESULTS_DIR="./results"
        export ENABLE_VALIDATION="true"  # Enable quick validation by default
        export VALIDATION_MIN_TICKS="50"
        export VALIDATION_TIMEOUT="0.5"  # 500ms timeout

        # Run trading engine
        if [ "$ENABLE_LOGGING" = true ]; then
            LOGFILE="$LOGS_DIR/paper_trading_$(date +%Y%m%d_%H%M%S).log"
            RESULTS_DIR="./results"
            mkdir -p "$RESULTS_DIR"
            export RESULTS_DIR="$RESULTS_DIR"
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
                from QDB.improved_optimized_indexer import ImprovedOptimizedIndexer
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
                from QDB import create_qdb
                qdb = create_qdb(base_path=QDB_PATH)
                print(f'✓ QDB initialized for backtest: {QDB_PATH}')
        else:
            from QDB import create_qdb
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
        from Microstructure_Analysis.microstructure_profiling import MicrostructureProfiler
from Alpha_Modeling.factor_hypothesis import FactorHypothesisGenerator
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
                # Run research pipeline manually
                from Microstructure_Analysis.microstructure_profiling import MicrostructureProfiler
                from Alpha_Modeling.factor_hypothesis import FactorHypothesisGenerator
                from Alpha_Modeling.statistical_validation import StatisticalValidator
                
                profiler = MicrostructureProfiler()
                profile_results = profiler.analyze(market_data)
                
                generator = FactorHypothesisGenerator()
                factors = generator.generate(profile_results)
                
                validator = StatisticalValidator()
                validation_results = validator.validate(factors, forward_returns if forward_returns is not None else market_data.get('returns', pd.Series()))
                
                # Export results
                import json
                research_results = {
                    'profiling': profile_results,
                    'factors': [f.__dict__ for f in factors] if factors else [],
                    'validation': validation_results.__dict__ if hasattr(validation_results, '__dict__') else str(validation_results),
                    'timestamp': datetime.now().isoformat()
                }
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_file = f'{RESEARCH_OUTPUT}/research_backtest_{timestamp}.json'
                with open(output_file, 'w') as f:
                    json.dump(research_results, f, indent=2, default=str)
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
                from QDB.improved_optimized_indexer import ImprovedOptimizedIndexer
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
                from QDB import create_qdb
                qdb = create_qdb(base_path=QDB_PATH)
        else:
            from QDB import create_qdb
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
        from Microstructure_Analysis.microstructure_profiling import MicrostructureProfiler
from Alpha_Modeling.factor_hypothesis import FactorHypothesisGenerator
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
                # Run research pipeline manually
                from Microstructure_Analysis.microstructure_profiling import MicrostructureProfiler
                from Alpha_Modeling.factor_hypothesis import FactorHypothesisGenerator
                from Alpha_Modeling.statistical_validation import StatisticalValidator
                
                profiler = MicrostructureProfiler()
                profile_results = profiler.analyze(market_data)
                
                generator = FactorHypothesisGenerator()
                factors = generator.generate(profile_results)
                
                validator = StatisticalValidator()
                validation_results = validator.validate(factors, forward_returns if forward_returns is not None else market_data.get('returns', pd.Series()))
                
                # Export results
                import json
                research_results = {
                    'profiling': profile_results,
                    'factors': [f.__dict__ for f in factors] if factors else [],
                    'validation': validation_results.__dict__ if hasattr(validation_results, '__dict__') else str(validation_results),
                    'timestamp': datetime.now().isoformat()
                }
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_file = f'{RESEARCH_OUTPUT}/research_backtest_{timestamp}.json'
                with open(output_file, 'w') as f:
                    json.dump(research_results, f, indent=2, default=str)
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
from Market_Data.alpaca_connector import AlpacaConnector
from Strategy_Construction.strategy_registry import get_strategy

# Import strategy adapters to register strategies
try:
    from Strategy_Construction.strategy_adapters import MomentumStrategyAdapter, MeanReversionStrategyAdapter
    print("✓ Strategy adapters loaded")
except ImportError as e:
    print(f"⚠️  Strategy adapters not available: {e}")

# Select connector based on CONNECTOR environment variable
CONNECTOR_TYPE = os.environ.get('CONNECTOR', 'alpaca').lower()
print(f"Using connector: {CONNECTOR_TYPE}")

if CONNECTOR_TYPE == 'yahoo':
    try:
        from Market_Data.yahoo_connector import YahooFinanceConnector
        print("✓ Yahoo Finance connector loaded")
    except ImportError as e:
        print(f"❌ Yahoo Finance connector not available: {e}")
        print("Please install: pip install yfinance")
        sys.exit(1)
elif CONNECTOR_TYPE == 'binance':
    try:
        from Market_Data.binance_connector import BinanceConnector
        print("✓ Binance connector loaded")
    except ImportError as e:
        print(f"❌ Binance connector not available: {e}")
        print("Please install: pip install python-binance")
        sys.exit(1)
elif CONNECTOR_TYPE == 'polygon':
    try:
        from Market_Data.polygon_connector import PolygonConnector
        print("✓ Polygon.io connector loaded")
    except ImportError as e:
        print(f"❌ Polygon.io connector not available: {e}")
        print("Please install: pip install websockets")
        sys.exit(1)
elif CONNECTOR_TYPE == 'alphavantage':
    try:
        from Market_Data.alphavantage_connector import AlphaVantageConnector
        print("✓ Alpha Vantage connector loaded")
    except ImportError as e:
        print(f"❌ Alpha Vantage connector not available: {e}")
        sys.exit(1)
elif CONNECTOR_TYPE == 'coinbase':
    try:
        from Market_Data.coinbase_connector import CoinbaseProConnector
        print("✓ Coinbase Pro connector loaded")
    except ImportError as e:
        print(f"❌ Coinbase Pro connector not available: {e}")
        print("Please install: pip install websockets")
        sys.exit(1)
elif CONNECTOR_TYPE == 'iexcloud':
    try:
        from Market_Data.iexcloud_connector import IEXCloudConnector
        print("✓ IEX Cloud connector loaded")
    except ImportError as e:
        print(f"❌ IEX Cloud connector not available: {e}")
        print("Please install: pip install websockets")
        sys.exit(1)
else:  # Default to Alpaca
    try:
        from Market_Data.alpaca_connector import AlpacaConnector
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
        from QDB import create_qdb
        from QDB.ingestion import RealtimeCollector
        
        # Use optimized indexer if requested
        if QDB_OPTIMIZED:
            try:
                from QDB.improved_optimized_indexer import ImprovedOptimizedIndexer
                from QDB.cache import DataCache, CacheConfig
                from QDB.versioning import DataVersioning
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
                        from QDB.indexer import DataIndexer
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
from Execution.engine.integrated_trading_flow import IntegratedTradingFlow, create_sample_strategies, create_sample_data
try:
    from Risk_Control.portfolio_manager import RiskModel
except ImportError:
    # Fallback if RiskModel not available
    from enum import Enum
    class RiskModel(Enum):
        EQUAL_WEIGHT = "equal_weight"
        INVERSE_VOLATILITY = "inverse_volatility"
        MEAN_VARIANCE = "mean_variance"
        RISK_PARITY = "risk_parity"
        BLACK_LITTERMAN = "black_litterman"
        HIERARCHICAL_RISK_PARITY = "hrp"
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
                from QDB.improved_optimized_indexer import ImprovedOptimizedIndexer
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
                from QDB import create_qdb
                qdb = create_qdb(base_path=QDB_PATH)
                print(f'✓ QDB initialized for complete flow: {QDB_PATH}')
                print(f'  Data version: {QDB_DATA_VERSION}')
        else:
            from QDB import create_qdb
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
        from Microstructure_Analysis.microstructure_profiling import MicrostructureProfiler
        from Alpha_Modeling.factor_hypothesis import FactorHypothesisGenerator
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
            # Run research pipeline manually
            from Microstructure_Analysis.microstructure_profiling import MicrostructureProfiler
            from Alpha_Modeling.factor_hypothesis import FactorHypothesisGenerator
            from Alpha_Modeling.statistical_validation import StatisticalValidator
            
            profiler = MicrostructureProfiler()
            profile_results = profiler.analyze(market_data)
            
            generator = FactorHypothesisGenerator()
            factors = generator.generate(profile_results)
            
            validator = StatisticalValidator()
            validation_results = validator.validate(factors, forward_returns if forward_returns is not None else market_data.get('returns', pd.Series()))
            
            # Export results
            import json
            research_results = {
                'profiling': profile_results,
                'factors': [f.__dict__ for f in factors] if factors else [],
                'validation': validation_results.__dict__ if hasattr(validation_results, '__dict__') else str(validation_results),
                'timestamp': datetime.now().isoformat()
            }
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f'{RESEARCH_OUTPUT}/research_complete_flow_{timestamp}.json'
            with open(output_file, 'w') as f:
                json.dump(research_results, f, indent=2, default=str)
            print(f'\\n✓ Research results exported to {RESEARCH_OUTPUT}')

            # Derive a lightweight factor attribution summary
            try:
                factors_list = research_results.get('factors', [])
                validation_str = research_results.get('validation', '{}')
                
                # Parse validation results (it's a string representation of dict)
                validation_dict = {}
                try:
                    if isinstance(validation_str, dict):
                        validation_dict = validation_str
                    elif isinstance(validation_str, str) and validation_str.startswith('{'):
                        # The validation string contains Python object reprs, we need to extract numeric values
                        # Use regex to extract key metrics from the string
                        import re
                        # Extract factor names and their metrics
                        for factor_name in [f.get('name') for f in factors_list if isinstance(f, dict)]:
                            if not factor_name:
                                continue
                            
                            # Initialize dict for this factor
                            if factor_name not in validation_dict:
                                validation_dict[factor_name] = {'regression': {}}
                            
                            # Directly search for sharpe_ratio, ic_mean, t_stat after the factor name
                            # Pattern: 'FactorName': {... sharpe_ratio=value ...}
                            try:
                                factor_section_pattern = f\"'{factor_name}':.*?(?='[A-Z]|$)\"
                                factor_section = re.search(factor_section_pattern, validation_str, re.DOTALL)
                                
                                if factor_section:
                                    section = factor_section.group(0)
                                    # Extract sharpe_ratio
                                    sharpe_match = re.search(r'sharpe_ratio=([0-9.e+-]+)', section)
                                    if sharpe_match:
                                        try:
                                            validation_dict[factor_name]['regression']['sharpe_ratio'] = float(sharpe_match.group(1))
                                        except (ValueError, KeyError):
                                            pass
                                    
                                    # Extract ic_mean
                                    ic_match = re.search(r'ic_mean=([0-9.e+-]+)', section)
                                    if ic_match:
                                        try:
                                            validation_dict[factor_name]['regression']['ic_mean'] = float(ic_match.group(1))
                                        except (ValueError, KeyError):
                                            pass
                                    
                                    # Extract t_stat
                                    t_stat_match = re.search(r't_stat=([0-9.e+-]+)', section)
                                    if t_stat_match:
                                        try:
                                            validation_dict[factor_name]['regression']['t_stat'] = float(t_stat_match.group(1))
                                        except (ValueError, KeyError):
                                            pass
                                    
                                    # Extract long_short_return
                                    ls_return_match = re.search(r'long_short_return=([0-9.e+-]+)', section)
                                    if ls_return_match:
                                        try:
                                            if 'long_short' not in validation_dict[factor_name]:
                                                validation_dict[factor_name]['long_short'] = {}
                                            validation_dict[factor_name]['long_short']['long_short_return'] = float(ls_return_match.group(1))
                                        except (ValueError, KeyError):
                                            pass
                            except Exception as parse_err:
                                # Skip this factor if parsing fails
                                pass
                except Exception as e:
                    pass
                
                top_factors = []

                for f_dict in factors_list:
                    if not isinstance(f_dict, dict):
                        continue
                    # Try common naming patterns for factor id/name
                    name = f_dict.get('name') or f_dict.get('id') or f_dict.get('factor_name') or 'unknown_factor'
                    
                    # Extract score from validation results
                    score = None
                    score_details = {}
                    
                    if name in validation_dict:
                        try:
                            val_result = validation_dict[name]
                            if isinstance(val_result, dict):
                                # Try to extract from regression results
                                reg = val_result.get('regression')
                                if reg:
                                    # Try to get attributes from RegressionResult object
                                    if hasattr(reg, 'sharpe_ratio'):
                                        score = reg.sharpe_ratio
                                        score_details['sharpe_ratio'] = reg.sharpe_ratio
                                    elif hasattr(reg, 'ic_mean'):
                                        score = reg.ic_mean
                                        score_details['ic_mean'] = reg.ic_mean
                                    elif hasattr(reg, 't_stat'):
                                        score = abs(reg.t_stat)
                                        score_details['t_stat'] = reg.t_stat
                                    
                                    # Also try to extract from dict representation
                                    if score is None and isinstance(reg, dict):
                                        score = reg.get('sharpe_ratio')
                                        if score is None:
                                            score = reg.get('ic_mean')
                                        if score is None and reg.get('t_stat') is not None:
                                            score = abs(reg.get('t_stat'))
                                        
                                        if reg.get('sharpe_ratio') is not None:
                                            score_details['sharpe_ratio'] = reg['sharpe_ratio']
                                        if reg.get('ic_mean') is not None:
                                            score_details['ic_mean'] = reg['ic_mean']
                                        if reg.get('t_stat') is not None:
                                            score_details['t_stat'] = reg['t_stat']
                                
                                # Also check long_short results
                                ls = val_result.get('long_short')
                                if ls and score is None:
                                    if hasattr(ls, 'long_short_return'):
                                        score = abs(ls.long_short_return)
                                        score_details['long_short_return'] = ls.long_short_return
                                    elif isinstance(ls, dict):
                                        score = abs(ls.get('long_short_return', 0))
                                        if ls.get('long_short_return'):
                                            score_details['long_short_return'] = ls['long_short_return']
                        except (KeyError, AttributeError, TypeError) as e:
                            # Skip this factor if accessing validation_dict fails
                            pass
                    
                    # Fallback: try to find score in factor dict itself
                    if score is None:
                        for key in ['ic', 'information_coefficient', 't_value', 't_stat', 'sharpe', 'score']:
                            if key in f_dict:
                                score = f_dict[key]
                                score_details[key] = score
                                break
                    
                    top_factors.append({
                        'name': name,
                        'score': score,
                        'score_details': score_details if score_details else None,
                        'category': f_dict.get('category', 'unknown'),
                        'expected_target': f_dict.get('expected_target', 'unknown'),
                        'raw': f_dict
                    })

                # Sort by score when available, keep top 20
                def _score_key(item):
                    s = item.get('score')
                    # Put scored factors first, then by score descending
                    if s is None:
                        return (False, 0.0)
                    try:
                        return (True, float(s))
                    except (ValueError, TypeError):
                        return (False, 0.0)

                top_factors_sorted = sorted(top_factors, key=_score_key, reverse=True)[:20]

                factor_attribution = {
                    'top_factors': top_factors_sorted,
                    'n_factors_total': len(top_factors),
                    'summary': {
                        'top_factor': top_factors_sorted[0]['name'] if top_factors_sorted and top_factors_sorted[0].get('score') is not None else None,
                        'factors_with_scores': sum(1 for f in top_factors if f.get('score') is not None),
                    },
                    'timestamp': datetime.now().isoformat()
                }

                fa_file = f'{RESEARCH_OUTPUT}/factor_attribution_{timestamp}.json'
                with open(fa_file, 'w') as f:
                    json.dump(factor_attribution, f, indent=2, default=str)
                print(f'✓ Factor attribution summary exported to {fa_file}')
                
                # Print summary
                if top_factors_sorted and top_factors_sorted[0].get('score') is not None:
                    top = top_factors_sorted[0]
                    print(f'  Top factor: {top["name"]} (score: {top["score"]:.4f})')
            except Exception as e:
                print(f'  ⚠️  Could not generate factor attribution summary: {e}')
                import traceback
                traceback.print_exc()
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

# Ensure results directory exists and save results
import os
import json
from datetime import datetime
results_dir = os.environ.get('RESULTS_DIR', './results')
os.makedirs(results_dir, exist_ok=True)
os.makedirs(f'{results_dir}/strategies', exist_ok=True)
os.makedirs(f'{results_dir}/research', exist_ok=True)
os.makedirs(f'{results_dir}/backtest', exist_ok=True)
os.makedirs(f'{results_dir}/performance', exist_ok=True)
os.makedirs(f'{results_dir}/hft_metrics', exist_ok=True)

# Save complete results
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
result_file = f'{results_dir}/performance/complete_flow_{timestamp}.json'
with open(result_file, 'w') as f:
    json.dump(result, f, indent=2, default=str)
print(f'\\n✓ Complete flow results saved to: {result_file}')

# Save strategy comparison results separately
if 'strategy_results' in result:
    strategy_file = f'{results_dir}/strategies/strategy_comparison_{timestamp}.json'
    strategy_data = {
        'strategies_tested': result.get('strategies_tested', []),
        'strategy_results': {k: {
            'total_return': v.get('total_return', 0) if isinstance(v, dict) else 0,
            'error': v.get('error', None) if isinstance(v, dict) and 'error' in v else None
        } for k, v in result.get('strategy_results', {}).items()},
        'risk_results': result.get('risk_results', {}),
        'best_strategy': result.get('best_strategy', None),
        'timestamp': result.get('timestamp', datetime.now().isoformat())
    }
    with open(strategy_file, 'w') as f:
        json.dump(strategy_data, f, indent=2, default=str)
    print(f'✓ Strategy comparison saved to: {strategy_file}')

# Save backtest results separately
if 'strategy_results' in result and 'risk_results' in result:
    backtest_file = f'{results_dir}/backtest/backtest_results_{timestamp}.json'
    backtest_data = {
        'data_info': result.get('data_info', {}),
        'strategies': result.get('strategies_tested', []),
        'backtest_results': {
            name: {
                'total_return': result['strategy_results'][name].get('total_return', 0) if name in result['strategy_results'] and isinstance(result['strategy_results'][name], dict) else 0,
                'risk_metrics': result['risk_results'].get(name, {}) if name in result['risk_results'] else {}
            }
            for name in result.get('strategies_tested', [])
            if name in result.get('strategy_results', {})
        },
        'best_strategy': result.get('best_strategy', None),
        'timestamp': result.get('timestamp', datetime.now().isoformat())
    }
    with open(backtest_file, 'w') as f:
        json.dump(backtest_data, f, indent=2, default=str)
    print(f'✓ Backtest results saved to: {backtest_file}')

# Save HFT metrics separately
if 'hft_metrics' in result and result['hft_metrics']:
    hft_file = f'{results_dir}/hft_metrics/hft_metrics_{timestamp}.json'
    with open(hft_file, 'w') as f:
        json.dump(result['hft_metrics'], f, indent=2, default=str)
    print(f'✓ HFT metrics saved to: {hft_file}')
    
    # Generate HFT reports (import hft_metrics directly to avoid Evaluation/__init__ dependencies)
    try:
        import importlib.util
        from pathlib import Path as _Path
        hft_metrics_path = _Path('Evaluation') / 'hft_metrics.py'
        if hft_metrics_path.exists():
            spec = importlib.util.spec_from_file_location('hft_metrics', str(hft_metrics_path))
            hft_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(hft_mod)
            HFTEvaluator = getattr(hft_mod, 'HFTEvaluator', None)
            HFTMetrics = getattr(hft_mod, 'HFTMetrics', None)
        else:
            HFTEvaluator = None
            HFTMetrics = None

        if HFTEvaluator is not None and HFTMetrics is not None:
            for strategy_name, metrics_dict in result['hft_metrics'].items():
                if not isinstance(metrics_dict, dict):
                    continue

                evaluator = HFTEvaluator()
                m = HFTMetrics()

                # Map persisted dict keys back to HFTMetrics fields
                m.hit_ratio = float(metrics_dict.get('hit_ratio', 0.0))
                m.latency_jitter = float(metrics_dict.get('latency_jitter_ms', 0.0))
                m.cancel_to_trade_ratio = float(metrics_dict.get('cancel_to_trade_ratio', 0.0))
                m.order_book_imbalance_importance = float(metrics_dict.get('order_book_imbalance_importance', 0.0))
                m.alpha_decay_ms = float(metrics_dict.get('alpha_decay_ms', 0.0))
                m.slippage_bps = float(metrics_dict.get('slippage_bps', 0.0))
                m.throughput_tps = float(metrics_dict.get('throughput_tps', 0.0))

                m.total_signals = int(metrics_dict.get('total_signals', 0) or 0)
                m.correct_signals = int(metrics_dict.get('correct_signals', 0) or 0)
                m.total_trades = int(metrics_dict.get('total_trades', 0) or 0)
                m.total_cancels = int(metrics_dict.get('total_cancels', 0) or 0)
                # We don't have raw latency/slippage samples in the JSON; lists stay empty.
                
                evaluator.metrics = m
                report = evaluator.generate_report(strategy_name)
                report_file = f'{results_dir}/hft_metrics/report_{strategy_name}_{timestamp}.txt'
                with open(report_file, 'w') as f:
                    f.write(report)
                print(f'✓ HFT report for {strategy_name} saved to: {report_file}')
        else:
            print('⚠️  HFTEvaluator/HFTMetrics not available, skipping HFT reports generation')
    except Exception as e:
        print(f'⚠️  Could not generate HFT reports: {e}')

print('\\nComplete flow finished successfully!')
print(f'\\nAll reports saved to: {os.path.abspath(results_dir)}')
print(f'  - Strategies: {results_dir}/strategies/')
print(f'  - Research: {results_dir}/research/')
print(f'  - Backtest: {results_dir}/backtest/')
print(f'  - Performance: {results_dir}/performance/')
print(f'  - HFT Metrics: {results_dir}/hft_metrics/')
" 2>&1 | tee "$LOGFILE"
        else
            python3 -c "
import sys
sys.path.insert(0, '.')
from Execution.engine.integrated_trading_flow import IntegratedTradingFlow, create_sample_strategies, create_sample_data
try:
    from Risk_Control.portfolio_manager import RiskModel
except ImportError:
    # Fallback if RiskModel not available
    from enum import Enum
    class RiskModel(Enum):
        EQUAL_WEIGHT = "equal_weight"
        INVERSE_VOLATILITY = "inverse_volatility"
        MEAN_VARIANCE = "mean_variance"
        RISK_PARITY = "risk_parity"
        BLACK_LITTERMAN = "black_litterman"
        HIERARCHICAL_RISK_PARITY = "hrp"
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
                from QDB.improved_optimized_indexer import ImprovedOptimizedIndexer
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
                from QDB import create_qdb
                qdb = create_qdb(base_path=QDB_PATH)
        else:
            from QDB import create_qdb
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

# Ensure results directory exists and save results
import os
import json
from datetime import datetime
results_dir = os.environ.get('RESULTS_DIR', './results')
os.makedirs(results_dir, exist_ok=True)
os.makedirs(f'{results_dir}/strategies', exist_ok=True)
os.makedirs(f'{results_dir}/research', exist_ok=True)
os.makedirs(f'{results_dir}/backtest', exist_ok=True)
os.makedirs(f'{results_dir}/performance', exist_ok=True)
os.makedirs(f'{results_dir}/hft_metrics', exist_ok=True)

# Save complete results
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
result_file = f'{results_dir}/performance/complete_flow_{timestamp}.json'
with open(result_file, 'w') as f:
    json.dump(result, f, indent=2, default=str)
print(f'\\n✓ Complete flow results saved to: {result_file}')

# Save strategy comparison results separately
if 'strategy_results' in result:
    strategy_file = f'{results_dir}/strategies/strategy_comparison_{timestamp}.json'
    strategy_data = {
        'strategies_tested': result.get('strategies_tested', []),
        'strategy_results': {k: {
            'total_return': v.get('total_return', 0) if isinstance(v, dict) else 0,
            'error': v.get('error', None) if isinstance(v, dict) and 'error' in v else None
        } for k, v in result.get('strategy_results', {}).items()},
        'risk_results': result.get('risk_results', {}),
        'best_strategy': result.get('best_strategy', None),
        'timestamp': result.get('timestamp', datetime.now().isoformat())
    }
    with open(strategy_file, 'w') as f:
        json.dump(strategy_data, f, indent=2, default=str)
    print(f'✓ Strategy comparison saved to: {strategy_file}')

# Save backtest results separately
if 'strategy_results' in result and 'risk_results' in result:
    backtest_file = f'{results_dir}/backtest/backtest_results_{timestamp}.json'
    backtest_data = {
        'data_info': result.get('data_info', {}),
        'strategies': result.get('strategies_tested', []),
        'backtest_results': {
            name: {
                'total_return': result['strategy_results'][name].get('total_return', 0) if name in result['strategy_results'] and isinstance(result['strategy_results'][name], dict) else 0,
                'risk_metrics': result['risk_results'].get(name, {}) if name in result['risk_results'] else {}
            }
            for name in result.get('strategies_tested', [])
            if name in result.get('strategy_results', {})
        },
        'best_strategy': result.get('best_strategy', None),
        'timestamp': result.get('timestamp', datetime.now().isoformat())
    }
    with open(backtest_file, 'w') as f:
        json.dump(backtest_data, f, indent=2, default=str)
    print(f'✓ Backtest results saved to: {backtest_file}')

# Save HFT metrics separately
if 'hft_metrics' in result and result['hft_metrics']:
    hft_file = f'{results_dir}/hft_metrics/hft_metrics_{timestamp}.json'
    with open(hft_file, 'w') as f:
        json.dump(result['hft_metrics'], f, indent=2, default=str)
    print(f'✓ HFT metrics saved to: {hft_file}')
    
    # Generate HFT reports (import hft_metrics directly to avoid Evaluation/__init__ dependencies)
    try:
        import importlib.util
        from pathlib import Path as _Path
        hft_metrics_path = _Path('Evaluation') / 'hft_metrics.py'
        if hft_metrics_path.exists():
            spec = importlib.util.spec_from_file_location('hft_metrics', str(hft_metrics_path))
            hft_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(hft_mod)
            HFTEvaluator = hft_mod.HFTEvaluator
        else:
            HFTEvaluator = None

        if HFTEvaluator is not None:
            for strategy_name, metrics_dict in result['hft_metrics'].items():
                if isinstance(metrics_dict, dict):
                    evaluator = HFTEvaluator()
                    # Create a simple object with the metrics
                    class MetricsObj:
                        def __init__(self, d):
                            for k, v in d.items():
                                setattr(self, k, v)
                        def to_dict(self):
                            return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
                    evaluator.metrics = MetricsObj(metrics_dict)
                    report = evaluator.generate_report(strategy_name)
                    report_file = f'{results_dir}/hft_metrics/report_{strategy_name}_{timestamp}.txt'
                    with open(report_file, 'w') as f:
                        f.write(report)
                    print(f'✓ HFT report for {strategy_name} saved to: {report_file}')
        else:
            print('⚠️  HFTEvaluator not available, skipping HFT reports generation')
    except Exception as e:
        print(f'⚠️  Could not generate HFT reports: {e}')

print('\\nComplete flow finished successfully!')
print(f'\\nAll reports saved to: {os.path.abspath(results_dir)}')
print(f'  - Strategies: {results_dir}/strategies/')
print(f'  - Research: {results_dir}/research/')
print(f'  - Backtest: {results_dir}/backtest/')
print(f'  - Performance: {results_dir}/performance/')
print(f'  - HFT Metrics: {results_dir}/hft_metrics/')
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
