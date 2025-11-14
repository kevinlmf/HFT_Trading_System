"""
策略工厂 - 创建各种类型的交易策略
包括：Classical, ML, RL, HFT等
"""
import numpy as np
import pandas as pd
from typing import Dict, Callable, Optional, List
import warnings
warnings.filterwarnings('ignore')

# 尝试导入ML策略
try:
    from Execution.strategies.ml_based.ml_traditional import (
        RandomForestStrategy, XGBoostStrategy, LightGBMStrategy
    )
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# 尝试导入RL策略
try:
    from Execution.strategies.ml_based.rl_strategies import DQNStrategy, PPOStrategy
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False


class StrategyFactory:
    """策略工厂 - 创建各种交易策略"""
    
    @staticmethod
    def create_all_strategies(data: pd.DataFrame) -> Dict[str, Callable]:
        """
        创建所有可用的策略
        
        Args:
            data: 历史数据（用于训练ML/RL模型）
        
        Returns:
            策略字典 {name: strategy_function}
        """
        strategies = {}
        
        # 1. Classical Strategies
        strategies.update(StrategyFactory._create_classical_strategies())
        
        # 2. ML Strategies
        if ML_AVAILABLE:
            strategies.update(StrategyFactory._create_ml_strategies(data))
        else:
            print("⚠ ML strategies not available (install scikit-learn, xgboost)")
        
        # 3. RL Strategies (简化版，用于快速测试)
        strategies.update(StrategyFactory._create_simple_rl_strategies(data))
        
        # 4. HFT Strategies
        strategies.update(StrategyFactory._create_hft_strategies())
        
        # 5. Statistical Strategies
        strategies.update(StrategyFactory._create_statistical_strategies())
        
        return strategies
    
    @staticmethod
    def _create_classical_strategies() -> Dict[str, Callable]:
        """创建经典策略"""
        strategies = {}
        
        # 1. Momentum
        def momentum(data: pd.DataFrame):
            if len(data) < 20 or 'close' not in data.columns:
                return 0
            returns = data['close'].pct_change(20)
            if len(returns) == 0 or pd.isna(returns.iloc[-1]):
                return 0
            current_return = returns.iloc[-1]
            if current_return > 0.02:
                return 1
            elif current_return < -0.02:
                return -1
            return 0
        
        # 2. Mean Reversion
        def mean_reversion(data: pd.DataFrame):
            if len(data) < 20 or 'close' not in data.columns:
                return 0
            ma = data['close'].rolling(20).mean()
            if len(ma) == 0 or pd.isna(ma.iloc[-1]) or ma.iloc[-1] == 0:
                return 0
            current_price = data['close'].iloc[-1]
            deviation = (current_price - ma.iloc[-1]) / ma.iloc[-1]
            if deviation < -0.05:
                return 1
            elif deviation > 0.05:
                return -1
            return 0
        
        # 3. RSI Strategy
        def rsi_strategy(data: pd.DataFrame):
            if len(data) < 14 or 'close' not in data.columns:
                return 0
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / (loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            if len(rsi) == 0 or pd.isna(rsi.iloc[-1]):
                return 0
            current_rsi = rsi.iloc[-1]
            if current_rsi < 30:
                return 1  # 超卖，买入
            elif current_rsi > 70:
                return -1  # 超买，卖出
            return 0
        
        # 4. MACD Strategy
        def macd_strategy(data: pd.DataFrame):
            if len(data) < 26 or 'close' not in data.columns:
                return 0
            ema12 = data['close'].ewm(span=12, adjust=False).mean()
            ema26 = data['close'].ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9, adjust=False).mean()
            if len(macd) == 0 or len(signal) == 0:
                return 0
            if pd.isna(macd.iloc[-1]) or pd.isna(signal.iloc[-1]):
                return 0
            if macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-2] <= signal.iloc[-2]:
                return 1  # 金叉，买入
            elif macd.iloc[-1] < signal.iloc[-1] and macd.iloc[-2] >= signal.iloc[-2]:
                return -1  # 死叉，卖出
            return 0
        
        # 5. Bollinger Bands
        def bollinger_bands(data: pd.DataFrame):
            if len(data) < 20 or 'close' not in data.columns:
                return 0
            ma = data['close'].rolling(20).mean()
            std = data['close'].rolling(20).std()
            upper = ma + 2 * std
            lower = ma - 2 * std
            if len(ma) == 0 or pd.isna(ma.iloc[-1]):
                return 0
            current_price = data['close'].iloc[-1]
            if current_price < lower.iloc[-1]:
                return 1  # 价格低于下轨，买入
            elif current_price > upper.iloc[-1]:
                return -1  # 价格高于上轨，卖出
            return 0
        
        # 6. Moving Average Crossover
        def ma_crossover(data: pd.DataFrame):
            if len(data) < 50 or 'close' not in data.columns:
                return 0
            ma_short = data['close'].rolling(10).mean()
            ma_long = data['close'].rolling(50).mean()
            if len(ma_short) == 0 or len(ma_long) == 0:
                return 0
            if pd.isna(ma_short.iloc[-1]) or pd.isna(ma_long.iloc[-1]):
                return 0
            if ma_short.iloc[-1] > ma_long.iloc[-1] and ma_short.iloc[-2] <= ma_long.iloc[-2]:
                return 1  # 短期均线上穿长期，买入
            elif ma_short.iloc[-1] < ma_long.iloc[-1] and ma_short.iloc[-2] >= ma_long.iloc[-2]:
                return -1  # 短期均线下穿长期，卖出
            return 0
        
        strategies.update({
            'momentum': momentum,
            'mean_reversion': mean_reversion,
            'rsi': rsi_strategy,
            'macd': macd_strategy,
            'bollinger_bands': bollinger_bands,
            'ma_crossover': ma_crossover
        })
        
        return strategies
    
    @staticmethod
    def _create_ml_strategies(data: pd.DataFrame) -> Dict[str, Callable]:
        """创建ML策略"""
        strategies = {}
        
        if not ML_AVAILABLE or len(data) < 100:
            return strategies
        
        try:
            # 准备特征
            features = StrategyFactory._prepare_features(data)
            if features is None or len(features) < 50:
                return strategies
            
            # 1. Random Forest
            def rf_strategy(data: pd.DataFrame):
                try:
                    if len(data) < 50:
                        return 0
                    features = StrategyFactory._prepare_features(data)
                    if features is None or len(features) == 0:
                        return 0
                    # 使用简单的随机森林（简化版）
                    from sklearn.ensemble import RandomForestClassifier
                    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
                    X = features[:-1]
                    y = (features['close'].pct_change().shift(-1) > 0).astype(int)[:-1]
                    if len(X) < 20 or y.sum() == 0 or y.sum() == len(y):
                        return 0
                    model.fit(X, y)
                    pred = model.predict([features.iloc[-1].values])[0]
                    return 1 if pred == 1 else -1
                except:
                    return 0
            
            # 2. XGBoost (简化版)
            def xgb_strategy(data: pd.DataFrame):
                try:
                    if len(data) < 50:
                        return 0
                    features = StrategyFactory._prepare_features(data)
                    if features is None or len(features) == 0:
                        return 0
                    # 使用简单的XGBoost
                    try:
                        import xgboost as xgb
                        model = xgb.XGBClassifier(n_estimators=50, random_state=42, n_jobs=-1)
                        X = features[:-1]
                        y = (features['close'].pct_change().shift(-1) > 0).astype(int)[:-1]
                        if len(X) < 20:
                            return 0
                        model.fit(X, y)
                        pred = model.predict([features.iloc[-1].values])[0]
                        return 1 if pred == 1 else -1
                    except ImportError:
                        return 0
                except:
                    return 0
            
            strategies.update({
                'ml_random_forest': rf_strategy,
                'ml_xgboost': xgb_strategy
            })
        except Exception as e:
            print(f"  ⚠ Warning: ML strategies creation failed: {e}")
        
        return strategies
    
    @staticmethod
    def _create_simple_rl_strategies(data: pd.DataFrame) -> Dict[str, Callable]:
        """创建简化的RL策略（用于快速测试）"""
        strategies = {}
        
        # 简化的RL策略 - 基于Q-learning的简单实现
        def simple_rl_strategy(data: pd.DataFrame):
            """简化的RL策略 - 基于状态-动作值函数"""
            if len(data) < 30 or 'close' not in data.columns:
                return 0
            
            # 定义状态：价格变化、波动率、趋势
            returns = data['close'].pct_change(5)
            volatility = returns.rolling(10).std()
            trend = (data['close'].iloc[-1] - data['close'].iloc[-10]) / data['close'].iloc[-10]
            
            if len(returns) == 0 or pd.isna(returns.iloc[-1]):
                return 0
            
            # 简单的规则：结合多个信号
            signal = 0
            if returns.iloc[-1] > 0.01 and trend > 0.02:
                signal = 1  # 买入
            elif returns.iloc[-1] < -0.01 and trend < -0.02:
                signal = -1  # 卖出
            
            return signal
        
        # 基于策略梯度的简化策略
        def policy_gradient_strategy(data: pd.DataFrame):
            """简化的策略梯度策略"""
            if len(data) < 20 or 'close' not in data.columns:
                return 0
            
            # 计算多个技术指标
            ma_short = data['close'].rolling(5).mean()
            ma_long = data['close'].rolling(20).mean()
            rsi = StrategyFactory._calculate_rsi(data['close'])
            
            if len(ma_short) == 0 or pd.isna(ma_short.iloc[-1]):
                return 0
            
            # 综合信号
            signal = 0
            if ma_short.iloc[-1] > ma_long.iloc[-1] and rsi < 60:
                signal = 1
            elif ma_short.iloc[-1] < ma_long.iloc[-1] and rsi > 40:
                signal = -1
            
            return signal
        
        strategies.update({
            'rl_simple': simple_rl_strategy,
            'rl_policy_gradient': policy_gradient_strategy
        })
        
        return strategies
    
    @staticmethod
    def _create_hft_strategies() -> Dict[str, Callable]:
        """创建HFT策略"""
        strategies = {}
        
        # 1. Market Making
        def market_making(data: pd.DataFrame):
            if len(data) < 5 or 'close' not in data.columns:
                return 0
            # 简单的做市策略：在价格波动时提供流动性
            volatility = data['close'].pct_change().rolling(5).std()
            if len(volatility) == 0 or pd.isna(volatility.iloc[-1]):
                return 0
            # 高波动时减少仓位
            if volatility.iloc[-1] > 0.02:
                return 0
            # 低波动时持有
            return 0
        
        # 2. Order Flow Imbalance
        def order_flow_imbalance(data: pd.DataFrame):
            if len(data) < 10:
                return 0
            # 简化的订单流不平衡策略
            if 'quantity' in data.columns:
                recent_vol = data['quantity'].tail(10).mean()
                if recent_vol > data['quantity'].mean() * 1.2:
                    return 1  # 高成交量，买入
                elif recent_vol < data['quantity'].mean() * 0.8:
                    return -1  # 低成交量，卖出
            return 0
        
        strategies.update({
            'hft_market_making': market_making,
            'hft_order_flow': order_flow_imbalance
        })
        
        return strategies
    
    @staticmethod
    def _create_statistical_strategies() -> Dict[str, Callable]:
        """创建统计套利策略"""
        strategies = {}
        
        # 1. Pairs Trading (简化版)
        def pairs_trading(data: pd.DataFrame):
            if len(data) < 40:
                return 0
            # 简化的配对交易：基于价格与均值的偏离
            ma = data['close'].rolling(20).mean()
            std = data['close'].rolling(20).std()
            zscore = (data['close'] - ma) / (std + 1e-10)
            if len(zscore) == 0 or pd.isna(zscore.iloc[-1]):
                return 0
            if zscore.iloc[-1] < -2:
                return 1  # 价格过低，买入
            elif zscore.iloc[-1] > 2:
                return -1  # 价格过高，卖出
            return 0
        
        # 2. Mean Reversion (Statistical)
        def statistical_mean_reversion(data: pd.DataFrame):
            if len(data) < 30:
                return 0
            # 基于统计的均值回归
            returns = data['close'].pct_change()
            mean_return = returns.rolling(20).mean()
            std_return = returns.rolling(20).std()
            zscore = (returns - mean_return) / (std_return + 1e-10)
            if len(zscore) == 0 or pd.isna(zscore.iloc[-1]):
                return 0
            if zscore.iloc[-1] < -1.5:
                return 1  # 收益率过低，买入
            elif zscore.iloc[-1] > 1.5:
                return -1  # 收益率过高，卖出
            return 0
        
        strategies.update({
            'statistical_pairs': pairs_trading,
            'statistical_mean_reversion': statistical_mean_reversion
        })
        
        return strategies
    
    @staticmethod
    def _prepare_features(data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """准备ML特征"""
        if 'close' not in data.columns or len(data) < 50:
            return None
        
        features = pd.DataFrame(index=data.index)
        features['close'] = data['close']
        
        # 技术指标
        for period in [5, 10, 20]:
            features[f'ma_{period}'] = data['close'].rolling(period).mean()
            features[f'return_{period}'] = data['close'].pct_change(period)
        
        # 波动率
        features['volatility'] = data['close'].pct_change().rolling(10).std()
        
        # 填充缺失值
        features = features.fillna(method='bfill').fillna(0)
        
        return features
    
    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int = 14) -> float:
        """计算RSI"""
        if len(prices) < period:
            return 50.0
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if len(rsi) > 0 and not pd.isna(rsi.iloc[-1]) else 50.0
    
    @staticmethod
    def get_strategy_categories() -> Dict[str, List[str]]:
        """获取策略分类"""
        return {
            'Classical': ['momentum', 'mean_reversion', 'rsi', 'macd', 'bollinger_bands', 'ma_crossover'],
            'ML': ['ml_random_forest', 'ml_xgboost'],
            'RL': ['rl_simple', 'rl_policy_gradient'],
            'HFT': ['hft_market_making', 'hft_order_flow'],
            'Statistical': ['statistical_pairs', 'statistical_mean_reversion']
        }







