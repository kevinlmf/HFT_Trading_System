"""
Machine Learning Validation Framework

用机器学习方法验证因子有效性

验证方法：
1. 将有效因子作为RL/ML的输入
2. 看reward/return是否提升
3. 检查feature importance/SHAP解释
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from Alpha_Modeling.factor_hypothesis import FactorHypothesis
from Alpha_Modeling.statistical_validation import StatisticalValidator


@dataclass
class MLValidationResult:
    """机器学习验证结果"""
    model_type: str
    feature_importance: Dict[str, float]  # 特征重要性
    r_squared: float  # R²
    mse: float  # 均方误差
    sharpe_improvement: float  # Sharpe比率提升
    shap_values: Optional[np.ndarray] = None  # SHAP值（如果可用）


@dataclass
class RLValidationResult:
    """强化学习验证结果"""
    baseline_reward: float  # 基准奖励
    factor_reward: float  # 使用因子后的奖励
    reward_improvement: float  # 奖励提升
    feature_importance: Dict[str, float]  # 特征重要性


class MLValidator:
    """
    机器学习验证器
    
    验证因子在ML模型中的有效性
    """
    
    def __init__(self):
        self.validation_results = {}
    
    def validate_with_tree_models(
        self,
        factor_values: pd.Series,
        forward_returns: pd.Series,
        baseline_features: Optional[pd.DataFrame] = None,
        model_type: str = 'random_forest'
    ) -> MLValidationResult:
        """
        使用树模型验证因子
        
        Args:
            factor_values: 因子值序列
            forward_returns: 未来收益序列
            baseline_features: 基准特征（不含该因子）
            model_type: 模型类型（'random_forest' 或 'gradient_boosting'）
        
        Returns:
            ML验证结果
        """
        # 对齐数据
        aligned_data = pd.DataFrame({
            'factor': factor_values,
            'return': forward_returns
        }).dropna()
        
        if len(aligned_data) < 20:
            return MLValidationResult(
                model_type=model_type,
                feature_importance={},
                r_squared=0.0,
                mse=0.0,
                sharpe_improvement=0.0
            )
        
        # 准备特征
        if baseline_features is not None:
            # 合并基准特征和因子
            baseline_aligned = baseline_features.reindex(aligned_data.index).fillna(0)
            X = pd.concat([baseline_aligned, aligned_data[['factor']]], axis=1)
            feature_names = list(baseline_aligned.columns) + ['factor']
        else:
            X = aligned_data[['factor']]
            feature_names = ['factor']
        
        y = aligned_data['return'].values
        
        # 训练模型
        if model_type == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
        else:
            model = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
        
        model.fit(X, y)
        
        # 预测
        y_pred = model.predict(X)
        
        # 评估指标
        r_squared = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        
        # 特征重要性
        feature_importance = dict(zip(feature_names, model.feature_importances_))
        
        # Sharpe比率提升（简化：基于预测收益）
        baseline_sharpe = 0.0
        if baseline_features is not None:
            baseline_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
            baseline_model.fit(baseline_aligned, y)
            baseline_pred = baseline_model.predict(baseline_aligned)
            baseline_returns = baseline_pred
            baseline_sharpe = np.mean(baseline_returns) / np.std(baseline_returns) * np.sqrt(252) if np.std(baseline_returns) > 0 else 0.0
        
        factor_returns = y_pred
        factor_sharpe = np.mean(factor_returns) / np.std(factor_returns) * np.sqrt(252) if np.std(factor_returns) > 0 else 0.0
        
        sharpe_improvement = factor_sharpe - baseline_sharpe
        
        # SHAP值（如果可用）
        shap_values = None
        if SHAP_AVAILABLE and len(X) < 1000:  # SHAP计算较慢，限制样本数
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X.iloc[:100])  # 只计算前100个样本
            except:
                pass
        
        return MLValidationResult(
            model_type=model_type,
            feature_importance=feature_importance,
            r_squared=r_squared,
            mse=mse,
            sharpe_improvement=sharpe_improvement,
            shap_values=shap_values
        )
    
    def validate_with_rl(
        self,
        factor_values: pd.Series,
        forward_returns: pd.Series,
        baseline_features: Optional[pd.DataFrame] = None
    ) -> RLValidationResult:
        """
        使用强化学习验证因子（简化版）
        
        注意：这里使用简化的RL验证，实际应该使用完整的RL环境
        
        Args:
            factor_values: 因子值序列
            forward_returns: 未来收益序列
            baseline_features: 基准特征
        
        Returns:
            RL验证结果
        """
        # 对齐数据
        aligned_data = pd.DataFrame({
            'factor': factor_values,
            'return': forward_returns
        }).dropna()
        
        if len(aligned_data) < 20:
            return RLValidationResult(
                baseline_reward=0.0,
                factor_reward=0.0,
                reward_improvement=0.0,
                feature_importance={}
            )
        
        # 简化的RL验证：使用因子作为信号，计算策略收益
        # 实际RL验证应该使用完整的RL环境（如gym环境）
        
        # 基准奖励（不使用因子）
        baseline_reward = 0.0
        if baseline_features is not None:
            # 使用基准特征构建简单策略
            baseline_aligned = baseline_features.reindex(aligned_data.index).fillna(0)
            baseline_signal = baseline_aligned.mean(axis=1)
            baseline_returns = baseline_signal * aligned_data['return']
            baseline_reward = baseline_returns.sum()
        
        # 因子奖励（使用因子）
        factor_signal = aligned_data['factor']
        factor_returns = factor_signal * aligned_data['return']
        factor_reward = factor_returns.sum()
        
        reward_improvement = factor_reward - baseline_reward
        
        # 特征重要性（简化：基于相关性）
        feature_importance = {'factor': abs(factor_signal.corr(aligned_data['return']))}
        if baseline_features is not None:
            for col in baseline_aligned.columns:
                feature_importance[col] = abs(baseline_aligned[col].corr(aligned_data['return']))
        
        return RLValidationResult(
            baseline_reward=float(baseline_reward),
            factor_reward=float(factor_reward),
            reward_improvement=float(reward_improvement),
            feature_importance=feature_importance
        )
    
    def comprehensive_ml_validation(
        self,
        factor_values: pd.Series,
        forward_returns: pd.Series,
        hypothesis: FactorHypothesis,
        baseline_features: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        综合ML验证
        
        Args:
            factor_values: 因子值序列
            forward_returns: 未来收益序列
            hypothesis: 因子假设
            baseline_features: 基准特征
        
        Returns:
            综合ML验证结果
        """
        results = {}
        
        # 1. 树模型验证
        rf_result = self.validate_with_tree_models(
            factor_values, forward_returns, baseline_features, 'random_forest'
        )
        results['random_forest'] = rf_result
        
        gb_result = self.validate_with_tree_models(
            factor_values, forward_returns, baseline_features, 'gradient_boosting'
        )
        results['gradient_boosting'] = gb_result
        
        # 2. RL验证（简化版）
        rl_result = self.validate_with_rl(factor_values, forward_returns, baseline_features)
        results['rl'] = rl_result
        
        # 3. 验证结论
        is_valid = (
            rf_result.r_squared > 0.01 and  # R²足够大
            'factor' in rf_result.feature_importance and
            rf_result.feature_importance['factor'] > 0.1 and  # 因子重要性足够
            rf_result.sharpe_improvement > 0 and  # Sharpe提升为正
            rl_result.reward_improvement > 0  # RL奖励提升为正
        )
        
        results['is_valid'] = is_valid
        results['hypothesis'] = hypothesis
        
        self.validation_results[hypothesis.name] = results
        return results
    
    def print_ml_validation_results(self, factor_name: Optional[str] = None):
        """打印ML验证结果"""
        print("\n" + "=" * 80)
        print("Machine Learning Validation Results")
        print("=" * 80)
        
        factors_to_print = [factor_name] if factor_name else list(self.validation_results.keys())
        
        for factor_name in factors_to_print:
            if factor_name not in self.validation_results:
                continue
            
            results = self.validation_results[factor_name]
            hyp = results['hypothesis']
            
            print(f"\n{factor_name} ({hyp.category.value})")
            
            # Random Forest结果
            rf = results['random_forest']
            print(f"\nRandom Forest:")
            print(f"  R²: {rf.r_squared:.4f}")
            print(f"  MSE: {rf.mse:.4f}")
            print(f"  Sharpe Improvement: {rf.sharpe_improvement:.4f}")
            print(f"  Feature Importance:")
            for feat, imp in sorted(rf.feature_importance.items(), key=lambda x: x[1], reverse=True):
                print(f"    {feat}: {imp:.4f}")
            
            # Gradient Boosting结果
            gb = results['gradient_boosting']
            print(f"\nGradient Boosting:")
            print(f"  R²: {gb.r_squared:.4f}")
            print(f"  MSE: {gb.mse:.4f}")
            print(f"  Sharpe Improvement: {gb.sharpe_improvement:.4f}")
            
            # RL结果
            rl = results['rl']
            print(f"\nReinforcement Learning (Simplified):")
            print(f"  Baseline Reward: {rl.baseline_reward:.4f}")
            print(f"  Factor Reward: {rl.factor_reward:.4f}")
            print(f"  Reward Improvement: {rl.reward_improvement:.4f}")
            
            # 验证结论
            is_valid = results['is_valid']
            print(f"\n验证结论: {'✓ 因子在ML中有效' if is_valid else '✗ 因子在ML中无效'}")
            print("=" * 80)


# ========== 使用示例 ==========

if __name__ == "__main__":
    # 创建示例数据
    np.random.seed(42)
    n = 1000
    
    factor_values = pd.Series(np.random.randn(n).cumsum() * 0.01)
    forward_returns = 0.5 * factor_values + np.random.randn(n) * 0.02
    
    baseline_features = pd.DataFrame({
        'feature1': np.random.randn(n),
        'feature2': np.random.randn(n)
    })
    
    from Alpha_Modeling.factor_hypothesis import FactorHypothesis, FactorCategory
    
    hypothesis = FactorHypothesis(
        name="TestFactor",
        category=FactorCategory.ORDER_IMBALANCE,
        economic_motivation="测试因子",
        formula="TestFactor = X",
        expected_direction="正相关",
        expected_target="return",
        microstructure_basis="测试"
    )
    
    validator = MLValidator()
    results = validator.comprehensive_ml_validation(
        factor_values, forward_returns, hypothesis, baseline_features
    )
    
    validator.print_ml_validation_results("TestFactor")













