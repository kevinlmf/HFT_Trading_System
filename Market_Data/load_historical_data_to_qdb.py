#!/usr/bin/env python3
"""
将历史数据加载到 QDB

从 Yahoo Finance 或 Alpaca 下载 AAPL/MSFT 等标的的历史数据，并写入 QDB。
之后 complete-flow 就可以使用真实数据而不是 sample data。

用法:
    python Market_Data/load_historical_data_to_qdb.py --symbols AAPL,MSFT --days 365
    python Market_Data/load_historical_data_to_qdb.py --symbols AAPL,MSFT --days 365 --source alpaca
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import argparse
import pandas as pd
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from QDB import create_qdb
from QDB.ingestion import HistoricalDownloader


def download_from_yahoo(symbol: str, days: int = 365, interval: str = '1d') -> pd.DataFrame:
    """
    从 Yahoo Finance 下载历史数据
    
    Args:
        symbol: 交易标的
        days: 下载多少天的数据
        interval: 数据间隔 ('1d', '1h', '5m', '1m')
    
    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    try:
        import yfinance as yf
    except ImportError:
        print("⚠️  yfinance 未安装，请运行: pip install yfinance")
        sys.exit(1)
    
    print(f"  正在从 Yahoo Finance 下载 {symbol} ({days} 天, {interval})...")
    
    # 计算日期范围
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # 下载数据
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, end=end_date, interval=interval)
    
    if df.empty:
        print(f"  ⚠️  未获取到 {symbol} 的数据")
        return pd.DataFrame()
    
    # 标准化列名和格式
    df = df.reset_index()
    
    # 重命名列以匹配 QDB 标准格式
    column_mapping = {
        'Date': 'timestamp',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume',
    }
    
    # 只保留需要的列
    available_cols = {k: v for k, v in column_mapping.items() if k in df.columns}
    df = df.rename(columns=available_cols)
    
    # 设置 timestamp 为索引
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    
    # 添加 QDB 需要的标准列
    if 'last_price' not in df.columns:
        df['last_price'] = df.get('close', df.iloc[:, 0] if len(df.columns) > 0 else 100.0)
    
    # 如果没有 bid/ask，用 close 价格模拟（对于日线数据这是合理的）
    if 'bid_price' not in df.columns:
        df['bid_price'] = df['last_price'] - 0.01  # 模拟 bid
    if 'ask_price' not in df.columns:
        df['ask_price'] = df['last_price'] + 0.01  # 模拟 ask
    if 'bid_size' not in df.columns:
        df['bid_size'] = 1000  # 默认值
    if 'ask_size' not in df.columns:
        df['ask_size'] = 1000  # 默认值
    
    # 添加 symbol 列
    df['symbol'] = symbol
    
    print(f"  ✓ 下载了 {len(df)} 条记录 ({df.index.min()} 到 {df.index.max()})")
    
    return df


def download_from_alpaca(symbol: str, days: int = 365, timeframe: str = '1Day') -> pd.DataFrame:
    """
    从 Alpaca 下载历史数据
    
    Args:
        symbol: 交易标的
        days: 下载多少天的数据
        timeframe: 时间框架 ('1Day', '1Hour', '5Min', '1Min')
    
    Returns:
        DataFrame with standard columns
    """
    try:
        from Market_Data.alpaca_connector import AlpacaConnector
    except ImportError:
        print("⚠️  Alpaca connector 未找到")
        return pd.DataFrame()
    
    # 检查环境变量
    api_key = os.environ.get('ALPACA_API_KEY')
    api_secret = os.environ.get('ALPACA_API_SECRET')
    
    if not api_key or not api_secret:
        print("⚠️  未设置 ALPACA_API_KEY 和 ALPACA_API_SECRET 环境变量")
        print("   使用 Yahoo Finance 作为备选...")
        return download_from_yahoo(symbol, days)
    
    print(f"  正在从 Alpaca 下载 {symbol} ({days} 天, {timeframe})...")
    
    try:
        connector = AlpacaConnector(api_key=api_key, api_secret=api_secret)
        # 注意：这里需要 AlpacaConnector 有 get_historical_data 方法
        # 如果没有，可以回退到 yahoo
        print("  ⚠️  Alpaca historical download 功能待实现，使用 Yahoo Finance...")
        return download_from_yahoo(symbol, days)
    except Exception as e:
        print(f"  ⚠️  Alpaca 下载失败: {e}")
        print("   回退到 Yahoo Finance...")
        return download_from_yahoo(symbol, days)


def main():
    parser = argparse.ArgumentParser(description='将历史数据加载到 QDB')
    parser.add_argument('--symbols', type=str, default='AAPL,MSFT',
                       help='要下载的标的，用逗号分隔 (默认: AAPL,MSFT)')
    parser.add_argument('--days', type=int, default=365,
                       help='下载多少天的历史数据 (默认: 365)')
    parser.add_argument('--source', type=str, default='yahoo',
                       choices=['yahoo', 'alpaca'],
                       help='数据源 (默认: yahoo)')
    parser.add_argument('--interval', type=str, default='1d',
                       choices=['1d', '1h', '5m', '1m'],
                       help='数据间隔 (默认: 1d，日线)')
    parser.add_argument('--qdb-path', type=str, default='./Data/datasets/qdb',
                       help='QDB 存储路径 (默认: ./Data/datasets/qdb)')
    parser.add_argument('--data-version', type=str, default=None,
                       help='数据版本标识 (默认: qdb_YYYYMMDD)')
    
    args = parser.parse_args()
    
    # 解析 symbols
    symbols = [s.strip().upper() for s in args.symbols.split(',')]
    
    # 生成数据版本
    if args.data_version is None:
        args.data_version = f"qdb_{datetime.now().strftime('%Y%m%d')}"
    
    print("=" * 80)
    print("历史数据加载到 QDB")
    print("=" * 80)
    print(f"标的: {', '.join(symbols)}")
    print(f"数据源: {args.source}")
    print(f"时间范围: 最近 {args.days} 天")
    print(f"数据间隔: {args.interval}")
    print(f"QDB 路径: {args.qdb_path}")
    print(f"数据版本: {args.data_version}")
    print("=" * 80)
    
    # 初始化 QDB
    print("\n初始化 QDB...")
    qdb = create_qdb(base_path=args.qdb_path, memmap=True)
    downloader = HistoricalDownloader(qdb)
    
    # 下载并存储每个标的的数据
    results = {}
    
    for symbol in symbols:
        print(f"\n处理 {symbol}...")
        
        try:
            # 下载数据
            if args.source == 'alpaca':
                df = download_from_alpaca(symbol, days=args.days, timeframe=args.interval)
            else:
                df = download_from_yahoo(symbol, days=args.days, interval=args.interval)
            
            if df.empty:
                print(f"  ✗ {symbol}: 未获取到数据")
                results[symbol] = None
                continue
            
            # 存储到 QDB
            print(f"  正在存储到 QDB...")
            file_path = downloader.download_from_dataframe(
                symbol=symbol,
                df=df,
                data_version=args.data_version,
                source_format="standard",
                description=f"Historical {args.interval} data for {symbol} (last {args.days} days)"
            )
            
            print(f"  ✓ {symbol}: 已存储到 {file_path}")
            results[symbol] = file_path
            
            # 验证数据可以加载
            print(f"  验证数据加载...")
            test_df = qdb.load(symbol=symbol, start=None, end=None)
            if test_df is not None and len(test_df) > 0:
                print(f"  ✓ {symbol}: 验证成功，可加载 {len(test_df)} 条记录")
            else:
                print(f"  ⚠️  {symbol}: 验证失败，数据可能未正确存储")
        
        except Exception as e:
            print(f"  ✗ {symbol}: 错误 - {e}")
            import traceback
            traceback.print_exc()
            results[symbol] = None
    
    # 总结
    print("\n" + "=" * 80)
    print("加载完成")
    print("=" * 80)
    
    success_count = sum(1 for v in results.values() if v is not None)
    print(f"成功: {success_count}/{len(symbols)}")
    
    for symbol, file_path in results.items():
        if file_path:
            print(f"  ✓ {symbol}: {file_path}")
        else:
            print(f"  ✗ {symbol}: 失败")
    
    if success_count > 0:
        print(f"\n✓ 数据已加载到 QDB")
        print(f"  现在运行 complete-flow 将使用真实数据:")
        print(f"  ./run_trading.sh complete-flow --symbols {','.join(symbols)}")
    else:
        print("\n⚠️  没有数据被成功加载，complete-flow 将继续使用 sample data")
    
    print("=" * 80)


if __name__ == "__main__":
    main()

