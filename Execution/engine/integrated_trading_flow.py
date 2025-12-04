"""
Integrated Trading Flow
整合完整的交易流程：数据准备 -> 策略对比 -> 风险控制 -> 仓位管理 -> 执行
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Any
from pathlib import Path
import sys
import os
import importlib.util
from datetime import datetime

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from Risk_Control.portfolio_manager import RiskModel
except ImportError:
    # 如果导入失败，定义基本的 RiskModel
    from enum import Enum
    class RiskModel(Enum):
        EQUAL_WEIGHT = "equal_weight"
        INVERSE_VOLATILITY = "inverse_volatility"
        MEAN_VARIANCE = "mean_variance"
        RISK_PARITY = "risk_parity"
        BLACK_LITTERMAN = "black_litterman"
        HIERARCHICAL_RISK_PARITY = "hrp"

try:
    from Strategy_Construction.strategy_registry import get_strategy, list_strategies
except ImportError:
    def get_strategy(name: str):
        return None
    def list_strategies():
        return []

try:
    from Evaluation.strategy_benchmark import StrategyBenchmark
except ImportError:
    StrategyBenchmark = None

try:
    from Execution.engine.smart_executor import SmartExecutor
except ImportError:
    SmartExecutor = None

# 直接导入hft_metrics，避免通过__init__.py（可能有其他依赖问题）
try:
    import importlib.util
    hft_metrics_path = Path(__file__).parent.parent.parent / "Evaluation" / "hft_metrics.py"
    if hft_metrics_path.exists():
        spec = importlib.util.spec_from_file_location("hft_metrics", hft_metrics_path)
        hft_metrics_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hft_metrics_module)
        HFTEvaluator = hft_metrics_module.HFTEvaluator
        HFTMetrics = hft_metrics_module.HFTMetrics
    else:
        HFTEvaluator = None
        HFTMetrics = None
except Exception as e:
    # 如果导入失败，设置为None（不影响其他功能）
    HFTEvaluator = None
    HFTMetrics = None


def create_sample_strategies() -> Dict[str, Callable]:
    """
    创建示例策略，包括传统策略、ML、RL和LLM方法
    
    Returns:
        策略字典 {name: strategy_function}
    """
    strategies = {}
    
    # ========== 传统策略 ==========
    
    # 动量策略
    def momentum_strategy(data: pd.DataFrame, lookback: int = 20) -> pd.Series:
        """简单动量策略：如果过去N天上涨，买入信号"""
        if len(data) < lookback:
            return pd.Series([0] * len(data), index=data.index)
        prices = data['close'] if 'close' in data.columns else data.iloc[:, 0]
        returns = prices.pct_change(lookback)
        signals = (returns > 0.02).astype(int) - (returns < -0.02).astype(int)
        return signals.fillna(0)
    
    # 均值回归策略
    def mean_reversion_strategy(data: pd.DataFrame, lookback: int = 20) -> pd.Series:
        """均值回归策略：价格偏离均值时反向交易"""
        if len(data) < lookback:
            return pd.Series([0] * len(data), index=data.index)
        prices = data['close'] if 'close' in data.columns else data.iloc[:, 0]
        ma = prices.rolling(lookback).mean()
        std = prices.rolling(lookback).std()
        z_score = (prices - ma) / std
        signals = (-(z_score > 1.5).astype(int) + (z_score < -1.5).astype(int))
        return signals.fillna(0)
    
    strategies['momentum'] = momentum_strategy
    strategies['mean_reversion'] = mean_reversion_strategy
    
    # ========== ML 策略 ==========
    
    # Random Forest 策略
    def ml_random_forest_strategy(data: pd.DataFrame) -> pd.Series:
        """基于随机森林的ML策略"""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler
            import numpy as np
            
            if len(data) < 50:
                return pd.Series([0] * len(data), index=data.index)
            
            prices = data['close'] if 'close' in data.columns else data.iloc[:, 0]
            returns = prices.pct_change().fillna(0)
            
            # 特征工程
            features = []
            for lookback in [5, 10, 20]:
                features.append(returns.rolling(lookback).mean())
                features.append(returns.rolling(lookback).std())
                features.append(prices.rolling(lookback).mean() / prices - 1)
            
            feature_df = pd.concat(features, axis=1).fillna(0)
            
            # 创建标签（未来收益方向）
            forward_returns = returns.shift(-1).fillna(0)
            labels = (forward_returns > 0).astype(int) - (forward_returns < 0).astype(int)
            
            # 训练数据准备
            train_size = min(200, len(feature_df) // 2)
            if train_size < 20:
                return pd.Series([0] * len(data), index=data.index)
            
            X_train = feature_df.iloc[:train_size].values
            y_train = labels.iloc[:train_size].values
            
            # 训练模型
            model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            
            # 预测
            X_test = feature_df.iloc[train_size:].values
            if len(X_test) == 0:
                return pd.Series([0] * len(data), index=data.index)
            
            predictions = model.predict(X_test)
            
            # 组合信号
            signals = pd.Series([0] * train_size, index=feature_df.iloc[:train_size].index)
            signals = pd.concat([signals, pd.Series(predictions, index=feature_df.iloc[train_size:].index)])
            signals = signals.reindex(data.index, fill_value=0)
            
            return signals.fillna(0)
        except Exception as e:
            # 如果ML失败，返回零信号
            return pd.Series([0] * len(data), index=data.index)
    
    strategies['ml_random_forest'] = ml_random_forest_strategy
    
    # XGBoost 策略
    def ml_xgboost_strategy(data: pd.DataFrame) -> pd.Series:
        """基于XGBoost的ML策略"""
        try:
            import xgboost as xgb
            import numpy as np
            
            if len(data) < 50:
                return pd.Series([0] * len(data), index=data.index)
            
            prices = data['close'] if 'close' in data.columns else data.iloc[:, 0]
            returns = prices.pct_change().fillna(0)
            
            # 特征工程
            features = []
            for lookback in [5, 10, 20, 30]:
                features.append(returns.rolling(lookback).mean())
                features.append(returns.rolling(lookback).std())
                if 'volume' in data.columns:
                    features.append(data['volume'].rolling(lookback).mean() / data['volume'] - 1)
            
            feature_df = pd.concat(features, axis=1).fillna(0)
            
            # 创建标签
            forward_returns = returns.shift(-1).fillna(0)
            labels = (forward_returns > 0).astype(int)
            
            # 训练数据准备
            train_size = min(200, len(feature_df) // 2)
            if train_size < 20:
                return pd.Series([0] * len(data), index=data.index)
            
            X_train = feature_df.iloc[:train_size].values
            y_train = labels.iloc[:train_size].values
            
            # 训练模型
            model = xgb.XGBClassifier(n_estimators=50, max_depth=4, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            
            # 预测
            X_test = feature_df.iloc[train_size:].values
            if len(X_test) == 0:
                return pd.Series([0] * len(data), index=data.index)
            
            predictions = model.predict(X_test)
            probabilities = model.predict_proba(X_test)[:, 1]
            
            # 转换预测为信号（使用概率阈值）
            signals_raw = (probabilities > 0.6).astype(int) - (probabilities < 0.4).astype(int)
            
            # 组合信号
            signals = pd.Series([0] * train_size, index=feature_df.iloc[:train_size].index)
            signals = pd.concat([signals, pd.Series(signals_raw, index=feature_df.iloc[train_size:].index)])
            signals = signals.reindex(data.index, fill_value=0)
            
            return signals.fillna(0)
        except ImportError:
            # XGBoost未安装，返回零信号
            return pd.Series([0] * len(data), index=data.index)
        except Exception:
            return pd.Series([0] * len(data), index=data.index)
    
    try:
        import xgboost
        strategies['ml_xgboost'] = ml_xgboost_strategy
    except ImportError:
        pass
    
    # ========== RL 策略 ==========
    
    # 简单RL策略（基于策略梯度的简化版本）
    def rl_simple_strategy(data: pd.DataFrame) -> pd.Series:
        """基于强化学习的简化策略"""
        try:
            if len(data) < 50:
                return pd.Series([0] * len(data), index=data.index)
            
            prices = data['close'] if 'close' in data.columns else data.iloc[:, 0]
            returns = prices.pct_change().fillna(0)
            
            # 状态特征
            state_features = []
            for lookback in [5, 10, 20]:
                state_features.append(returns.rolling(lookback).mean())
                state_features.append(returns.rolling(lookback).std())
            
            state_df = pd.concat(state_features, axis=1).fillna(0)
            
            # 简单的RL策略：基于状态值函数的阈值决策
            # 这是一个简化的实现，实际RL需要训练过程
            signals = pd.Series([0] * len(data), index=data.index)
            
            for i in range(20, len(state_df)):
                state = state_df.iloc[i].values
                
                # 简单的策略：如果多个特征都为正，买入；都为负，卖出
                positive_features = np.sum(state > 0)
                negative_features = np.sum(state < 0)
                
                if positive_features >= len(state) * 0.6:
                    signals.iloc[i] = 1
                elif negative_features >= len(state) * 0.6:
                    signals.iloc[i] = -1
            
            return signals.fillna(0)
        except Exception:
            return pd.Series([0] * len(data), index=data.index)
    
    strategies['rl_simple'] = rl_simple_strategy
    
    # ========== LLM 策略 ==========
    
    # LLM增强策略（使用LLM分析市场情绪和模式）
    def llm_sentiment_strategy(data: pd.DataFrame) -> pd.Series:
        """基于LLM情绪分析的策略"""
        try:
            if len(data) < 30:
                return pd.Series([0] * len(data), index=data.index)
            
            prices = data['close'] if 'close' in data.columns else data.iloc[:, 0]
            returns = prices.pct_change().fillna(0)
            
            # 模拟LLM分析：基于价格模式识别
            # 实际LLM策略需要接入真实的LLM API（如GPT-4, Claude等）
            
            signals = pd.Series([0] * len(data), index=data.index)
            
            # 检测价格模式
            for i in range(20, len(prices)):
                recent_prices = prices.iloc[i-20:i]
                recent_returns = returns.iloc[i-20:i]
                
                # 模式1：上升趋势
                if recent_prices.iloc[-1] > recent_prices.iloc[0] * 1.02:
                    if recent_returns.mean() > 0:
                        signals.iloc[i] = 1  # 买入信号
                
                # 模式2：下降趋势
                elif recent_prices.iloc[-1] < recent_prices.iloc[0] * 0.98:
                    if recent_returns.mean() < 0:
                        signals.iloc[i] = -1  # 卖出信号
                
                # 模式3：波动加剧（可能的转折点）
                elif recent_returns.std() > returns.iloc[:i].std() * 1.5:
                    # 在波动加剧时减少交易
                    signals.iloc[i] = 0
            
            return signals.fillna(0)
        except Exception:
            return pd.Series([0] * len(data), index=data.index)
    
    strategies['llm_sentiment'] = llm_sentiment_strategy
    
    # LLM模式识别策略
    def llm_pattern_strategy(data: pd.DataFrame) -> pd.Series:
        """基于LLM模式识别的策略"""
        try:
            if len(data) < 40:
                return pd.Series([0] * len(data), index=data.index)
            
            prices = data['close'] if 'close' in data.columns else data.iloc[:, 0]
            returns = prices.pct_change().fillna(0)
            
            signals = pd.Series([0] * len(data), index=data.index)
            
            # 识别技术形态
            for i in range(30, len(prices)):
                window = prices.iloc[i-30:i]
                
                # 头肩顶/底形态检测（简化版）
                peaks = []
                troughs = []
                
                for j in range(1, len(window)-1):
                    if window.iloc[j] > window.iloc[j-1] and window.iloc[j] > window.iloc[j+1]:
                        peaks.append((j, window.iloc[j]))
                    elif window.iloc[j] < window.iloc[j-1] and window.iloc[j] < window.iloc[j+1]:
                        troughs.append((j, window.iloc[j]))
                
                # 如果检测到明显的上升模式
                if len(peaks) >= 2:
                    if peaks[-1][1] > peaks[0][1] * 1.01:
                        signals.iloc[i] = 1
                
                # 如果检测到明显的下降模式
                if len(troughs) >= 2:
                    if troughs[-1][1] < troughs[0][1] * 0.99:
                        signals.iloc[i] = -1
            
            return signals.fillna(0)
        except Exception:
            return pd.Series([0] * len(data), index=data.index)
    
    strategies['llm_pattern'] = llm_pattern_strategy
    
    return strategies


def create_sample_data(n_records: int = 1000) -> pd.DataFrame:
    """
    创建示例数据用于测试
    
    Args:
        n_records: 记录数量
        
    Returns:
        包含价格数据的 DataFrame
    """
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=n_records, freq='H')
    
    # 生成随机游走价格
    returns = np.random.randn(n_records) * 0.01
    prices = 100 * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'timestamp': dates,
        'price': prices,
        'close': prices,
        'open': prices * (1 + np.random.randn(n_records) * 0.001),
        'high': prices * (1 + np.abs(np.random.randn(n_records) * 0.002)),
        'low': prices * (1 - np.abs(np.random.randn(n_records) * 0.002)),
        'volume': np.random.randint(1000, 10000, n_records)
    })
    
    data.set_index('timestamp', inplace=True)
    return data


class IntegratedTradingFlow:
    """
    整合交易流程
    
    整合以下功能：
    1. 数据准备和清理
    2. 策略对比和评估
    3. 风险控制
    4. 仓位管理
    5. 智能执行
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        risk_model: RiskModel = RiskModel.RISK_PARITY,
        monte_carlo_paths: int = 100000,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ):
        """
        初始化整合交易流程
        
        Args:
            initial_capital: 初始资金
            risk_model: 风险模型
            monte_carlo_paths: Monte Carlo 模拟路径数
            risk_free_rate: 无风险利率
            periods_per_year: 每年交易周期数
        """
        self.initial_capital = initial_capital
        self.risk_model = risk_model
        self.monte_carlo_paths = monte_carlo_paths
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year
        
        # 初始化组件
        self.executor = SmartExecutor() if SmartExecutor else None
        self.benchmark = StrategyBenchmark() if StrategyBenchmark else None
        
        # 初始化HFT评估器（直接导入避免__init__依赖问题）
        if HFTEvaluator:
            try:
                self.hft_evaluator = HFTEvaluator()
                print(f"  ✓ HFT Metrics evaluator enabled")
            except Exception as e:
                print(f"  ⚠️  HFT Metrics evaluator initialization failed: {e}")
                self.hft_evaluator = None
        else:
            self.hft_evaluator = None
        
        print(f"✓ Integrated Trading Flow initialized")
        print(f"  Initial Capital: ${initial_capital:,.2f}")
        print(f"  Risk Model: {risk_model.value if hasattr(risk_model, 'value') else risk_model}")
        print(f"  Monte Carlo Paths: {monte_carlo_paths:,}")
    
    def execute_complete_flow_with_position_management(
        self,
        data: pd.DataFrame,
        strategies: Optional[Dict[str, Callable]] = None,
        symbols: Optional[List[str]] = None,
        force_slippage_impl: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        执行完整交易流程（包含仓位管理）
        
        Args:
            data: 市场数据
            strategies: 策略字典，如果为 None 则使用默认策略
            symbols: 交易标的列表
            force_slippage_impl: 强制使用的 slippage 实现
            
        Returns:
            包含所有结果的字典
        """
        print("\n" + "=" * 80)
        print("Executing Complete Trading Flow with Position Management")
        print("=" * 80)
        
        # 1. 准备数据
        print("\n[1/5] Preparing data...")
        if data is None or len(data) == 0:
            print("  ⚠️  No data provided, creating sample data")
            data = create_sample_data(n_records=1000)
        
        # 确保有 close 列
        if 'close' not in data.columns and 'price' in data.columns:
            data['close'] = data['price']
        elif 'close' not in data.columns:
            data['close'] = data.iloc[:, 0]
        
        print(f"  ✓ Data prepared: {len(data)} records")
        print(f"  ✓ Date range: {data.index[0]} to {data.index[-1]}")
        
        # 2. 准备策略（可选优化）
        print("\n[2/5] Preparing strategies...")
        if strategies is None:
            strategies = create_sample_strategies()
            print(f"  ✓ Using default strategies: {list(strategies.keys())}")
        else:
            print(f"  ✓ Using provided strategies: {list(strategies.keys())}")
        
        # 可选：应用HFT优化
        enable_optimization = os.environ.get('ENABLE_HFT_OPTIMIZATION', 'false').lower() == 'true'
        if enable_optimization:
            try:
                from Optimization.hft_optimizer import HFTOptimizer
                print("\n  🔧 Applying HFT optimizations...")
                optimizer = HFTOptimizer()
                optimized_strategies = {}
                for name, strategy_func in strategies.items():
                    print(f"    - Optimizing {name}...")
                    optimized_strategy, _ = optimizer.comprehensive_optimize(
                        strategy_func, data,
                        target_hit_ratio=0.55,
                        target_latency_ms=2.0,
                        target_throughput_tps=1000.0
                    )
                    optimized_strategies[name] = optimized_strategy
                strategies = optimized_strategies
                print("  ✓ HFT optimizations applied")
            except ImportError:
                print("  ⚠️  HFT optimizer not available, using original strategies")
            except Exception as e:
                print(f"  ⚠️  Optimization failed: {e}, using original strategies")
        
        # 3. 策略回测和对比（包含HFT指标）
        print("\n[3/5] Running strategy backtest and comparison...")
        strategy_results = {}
        hft_metrics_results = {}
        
        for name, strategy_func in strategies.items():
            try:
                print(f"  - Testing strategy: {name}")
                signals = strategy_func(data)
                
                # 计算简单收益
                if isinstance(signals, pd.Series):
                    returns = data['close'].pct_change()
                    strategy_returns = signals.shift(1) * returns
                    cumulative_returns = (1 + strategy_returns).cumprod()
                    total_return = cumulative_returns.iloc[-1] - 1 if len(cumulative_returns) > 0 else 0
                    
                    strategy_results[name] = {
                        'total_return': total_return,
                        'signals': signals,
                        'returns': strategy_returns,
                        'cumulative_returns': cumulative_returns
                    }
                    print(f"    ✓ Total Return: {total_return*100:.2f}%")
                    
                    # 计算HFT指标
                    if self.hft_evaluator:
                        print(f"    - Calculating HFT metrics...")
                        prices = data['close'] if 'close' in data.columns else data.iloc[:, 0]
                        
                        # 模拟执行时间（基于信号生成时间）
                        execution_times = [data.index[i] for i in range(len(data)) if signals.iloc[i] != 0][:1000]
                        
                        # 尝试获取订单簿数据
                        order_book_data = None
                        if 'bid_price' in data.columns and 'ask_price' in data.columns:
                            order_book_data = pd.DataFrame({
                                'bid_price': data.get('bid_price', prices),
                                'ask_price': data.get('ask_price', prices),
                                'bid_size': data.get('bid_size', pd.Series([1000] * len(data), index=data.index)),
                                'ask_size': data.get('ask_size', pd.Series([1000] * len(data), index=data.index))
                            })
                        
                        # 模拟交易和取消日志（简化版）
                        trade_log = []
                        cancel_log = []
                        for i, (idx, signal) in enumerate(signals.items()):
                            if signal != 0 and i < len(prices):
                                price = prices.iloc[i] if i < len(prices) else prices.iloc[-1]
                                trade_log.append({
                                    'execution_price': price * (1 + np.random.randn() * 0.0001),  # 模拟slippage
                                    'intended_price': price,
                                    'timestamp': idx
                                })
                                # 模拟一些取消订单
                                if np.random.rand() < 0.1:  # 10%的订单被取消
                                    cancel_log.append({'timestamp': idx})
                        
                        hft_metrics = self.hft_evaluator.evaluate_strategy(
                            signals=signals,
                            prices=prices,
                            execution_times=execution_times if execution_times else None,
                            order_book_data=order_book_data,
                            trade_log=trade_log if trade_log else None,
                            cancel_log=cancel_log if cancel_log else None
                        )
                        
                        hft_metrics_results[name] = hft_metrics
                        print(f"      ✓ Hit Ratio: {hft_metrics.hit_ratio*100:.2f}%")
                        print(f"      ✓ Latency Jitter: {hft_metrics.latency_jitter:.2f} ms")
                        print(f"      ✓ Alpha Decay: {hft_metrics.alpha_decay_ms:.2f} ms")
                        print(f"      ✓ Slippage: {hft_metrics.slippage_bps:.2f} bps")
                        print(f"      ✓ Throughput: {hft_metrics.throughput_tps:.2f} TPS")
            except Exception as e:
                print(f"    ✗ Error testing {name}: {e}")
                import traceback
                traceback.print_exc()
                strategy_results[name] = {'error': str(e)}
        
        # 4. 风险控制
        print("\n[4/5] Applying risk control...")
        risk_results = {}
        
        for name, result in strategy_results.items():
            if 'error' in result:
                continue
            try:
                returns = result.get('returns', pd.Series())
                if len(returns) > 0:
                    volatility = returns.std() * np.sqrt(self.periods_per_year)
                    sharpe = (returns.mean() * self.periods_per_year - self.risk_free_rate) / volatility if volatility > 0 else 0
                    
                    risk_results[name] = {
                        'volatility': volatility,
                        'sharpe_ratio': sharpe,
                        'max_drawdown': self._calculate_max_drawdown(result.get('cumulative_returns', pd.Series()))
                    }
                    print(f"  - {name}: Sharpe={sharpe:.2f}, Vol={volatility*100:.2f}%")
            except Exception as e:
                print(f"  ✗ Error calculating risk for {name}: {e}")
        
        # 5. 仓位管理和执行
        print("\n[5/5] Position management and execution...")
        position_results = {}
        
        # 选择最佳策略
        best_strategy = None
        best_sharpe = -np.inf
        
        for name, risk in risk_results.items():
            sharpe = risk.get('sharpe_ratio', -np.inf)
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_strategy = name
        
        if best_strategy:
            print(f"  ✓ Best strategy selected: {best_strategy} (Sharpe: {best_sharpe:.2f})")
            
            # 计算仓位
            if best_strategy in strategy_results:
                signals = strategy_results[best_strategy]['signals']
                positions = self._calculate_positions(signals, data)
                position_results[best_strategy] = {
                    'positions': positions,
                    'total_trades': (signals.diff() != 0).sum()
                }
                print(f"  ✓ Position management completed: {position_results[best_strategy]['total_trades']} trades")
        else:
            print("  ⚠️  No valid strategy found for position management")
        
        # 汇总结果
        result = {
            'data_info': {
                'n_records': len(data),
                'date_range': (str(data.index[0]), str(data.index[-1]))
            },
            'strategies_tested': list(strategies.keys()),
            'strategy_results': strategy_results,
            'risk_results': risk_results,
            'hft_metrics': {k: v.to_dict() if hasattr(v, 'to_dict') else v for k, v in hft_metrics_results.items()},
            'best_strategy': best_strategy,
            'position_results': position_results,
            'timestamp': datetime.now().isoformat()
        }
        
        print("\n" + "=" * 80)
        print("Complete Flow Finished Successfully")
        print("=" * 80)
        print(f"\nBest Strategy: {best_strategy}")
        if best_strategy and best_strategy in risk_results:
            risk = risk_results[best_strategy]
            print(f"  Sharpe Ratio: {risk.get('sharpe_ratio', 0):.2f}")
            print(f"  Volatility: {risk.get('volatility', 0)*100:.2f}%")
            print(f"  Max Drawdown: {risk.get('max_drawdown', 0)*100:.2f}%")
        
        # 打印HFT指标摘要
        if hft_metrics_results:
            print("\n" + "=" * 80)
            print("HFT Metrics Summary")
            print("=" * 80)
            for name, metrics in hft_metrics_results.items():
                if hasattr(metrics, 'hit_ratio'):
                    print(f"\n{name}:")
                    print(f"  Hit Ratio: {metrics.hit_ratio*100:.2f}%")
                    print(f"  Latency Jitter: {metrics.latency_jitter:.2f} ms")
                    print(f"  Cancel-to-Trade Ratio: {metrics.cancel_to_trade_ratio:.2f}")
                    print(f"  Alpha Decay: {metrics.alpha_decay_ms:.2f} ms")
                    print(f"  Slippage: {metrics.slippage_bps:.2f} bps")
                    print(f"  Throughput: {metrics.throughput_tps:.2f} TPS")
        
        return result
    
    def _calculate_max_drawdown(self, cumulative_returns: pd.Series) -> float:
        """计算最大回撤"""
        if len(cumulative_returns) == 0:
            return 0.0
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        return abs(drawdown.min())
    
    def _calculate_positions(self, signals: pd.Series, data: pd.DataFrame) -> pd.Series:
        """根据信号计算仓位"""
        # 简单实现：信号为1时满仓，-1时空仓，0时保持
        positions = signals.copy()
        positions[positions > 0] = 1.0  # 满仓
        positions[positions < 0] = -1.0  # 做空
        positions[positions == 0] = 0.0  # 空仓
        return positions.fillna(0)

