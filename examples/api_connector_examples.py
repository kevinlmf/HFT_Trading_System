"""
使用不同API连接器的示例

展示如何使用Yahoo Finance、Binance、Polygon.io等连接器
"""

import asyncio
import sys
import os
sys.path.insert(0, os.getcwd())

from Execution.trading.trading_engine import RealTimeTradingEngine
from Execution.strategies.strategy_registry import get_strategy

# 导入策略适配器
try:
    from Execution.strategies.strategy_adapters import MomentumStrategyAdapter, MeanReversionStrategyAdapter
except ImportError:
    pass


async def example_yahoo_finance():
    """使用Yahoo Finance连接器（免费，无需API密钥）"""
    print("=" * 70)
    print("示例1: Yahoo Finance连接器")
    print("=" * 70)
    
    try:
        from Market_Data.yahoo_connector import YahooFinanceConnector
        
        # 创建连接器（无需API密钥）
        connector = YahooFinanceConnector(update_interval=10.0)  # 每10秒更新
        
        # 创建策略
        strategies = {
            'momentum': get_strategy('momentum'),
            'mean_reversion': get_strategy('mean_reversion'),
        }
        
        # 创建交易引擎
        engine = RealTimeTradingEngine(
            connector=connector,
            strategies=strategies,
            initial_capital=100000.0,
            symbols=['AAPL', 'MSFT'],
            update_interval=10.0
        )
        
        print("✓ Yahoo Finance连接器已创建")
        print("注意: Yahoo Finance数据有15-20分钟延迟，适合测试")
        print("\n启动引擎...")
        print("(运行10秒后自动停止)")
        
        # 运行10秒
        await asyncio.sleep(10)
        await engine.stop()
        
    except ImportError:
        print("✗ 请先安装yfinance: pip install yfinance")
    except Exception as e:
        print(f"✗ 错误: {e}")


async def example_binance():
    """使用Binance连接器（加密货币）"""
    print("\n" + "=" * 70)
    print("示例2: Binance连接器（加密货币）")
    print("=" * 70)
    
    try:
        from Market_Data.binance_connector import BinanceConnector
        
        # 创建连接器（数据订阅不需要API密钥）
        connector = BinanceConnector(testnet=True)  # 使用测试网
        
        # 创建策略
        strategies = {
            'momentum': get_strategy('momentum'),
        }
        
        # 创建交易引擎
        engine = RealTimeTradingEngine(
            connector=connector,
            strategies=strategies,
            initial_capital=100000.0,
            symbols=['BTCUSDT', 'ETHUSDT'],  # Binance使用ticker格式
            update_interval=5.0
        )
        
        print("✓ Binance连接器已创建")
        print("注意: 使用测试网，不会产生真实交易")
        print("\n启动引擎...")
        print("(运行10秒后自动停止)")
        
        # 运行10秒
        await asyncio.sleep(10)
        await engine.stop()
        
    except ImportError:
        print("✗ 请先安装python-binance: pip install python-binance")
    except Exception as e:
        print(f"✗ 错误: {e}")


async def example_polygon():
    """使用Polygon.io连接器（需要API密钥）"""
    print("\n" + "=" * 70)
    print("示例3: Polygon.io连接器")
    print("=" * 70)
    
    try:
        from Market_Data.polygon_connector import PolygonConnector
        
        # 需要API密钥
        api_key = os.environ.get('POLYGON_API_KEY')
        if not api_key:
            print("✗ 请设置环境变量: export POLYGON_API_KEY='your_key'")
            print("  注册: https://polygon.io/")
            return
        
        # 创建连接器
        connector = PolygonConnector(api_key=api_key)
        
        # 创建策略
        strategies = {
            'momentum': get_strategy('momentum'),
            'mean_reversion': get_strategy('mean_reversion'),
        }
        
        # 创建交易引擎
        engine = RealTimeTradingEngine(
            connector=connector,
            strategies=strategies,
            initial_capital=100000.0,
            symbols=['AAPL', 'MSFT'],
            update_interval=5.0
        )
        
        print("✓ Polygon.io连接器已创建")
        print("\n启动引擎...")
        print("(运行10秒后自动停止)")
        
        # 运行10秒
        await asyncio.sleep(10)
        await engine.stop()
        
    except ImportError:
        print("✗ 请先安装websockets: pip install websockets")
    except Exception as e:
        print(f"✗ 错误: {e}")


async def main():
    """运行所有示例"""
    print("\n" + "=" * 70)
    print("API连接器使用示例")
    print("=" * 70)
    print("\n选择要运行的示例:")
    print("1. Yahoo Finance (免费，无需API密钥)")
    print("2. Binance (加密货币，测试网)")
    print("3. Polygon.io (需要API密钥)")
    print("4. 运行所有示例")
    
    choice = input("\n请输入选择 (1-4): ").strip()
    
    if choice == '1':
        await example_yahoo_finance()
    elif choice == '2':
        await example_binance()
    elif choice == '3':
        await example_polygon()
    elif choice == '4':
        await example_yahoo_finance()
        await example_binance()
        await example_polygon()
    else:
        print("无效选择")


if __name__ == "__main__":
    asyncio.run(main())












