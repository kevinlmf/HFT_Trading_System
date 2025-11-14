"""
Statistical Validation Framework

用统计方法验证因子有效性

验证框架：
1. 单因子回归：r_{t+1} = α + β*f_t + ε_t
2. 对冲组合测试（long-short top-bottom quantiles）
3. 稳定性验证（滚动窗口、横截面）
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

from Research.factor_hypothesis import FactorHypothesis


@dataclass
class RegressionResult:
    """回归结果"""
    alpha: float  # 截距
    beta: float  # 因子系数
    beta_std: float  # 系数标准差
    t_stat: float  # t统计量
    p_value: float  # p值
    r_squared: float  # R²
    sharpe_ratio: float  # 策略Sharpe比率
    ic_mean: float  # 信息系数均值
    ic_std: float  # 信息系数标准差
    ic_ir: float  # 信息比率（IC均值/IC标准差）


@dataclass
class LongShortResult:
    """多空组合结果"""
    long_return: float  # 多头收益
    short_return: float  # 空头收益
    long_short_return: float  # 多空收益
    sharpe_ratio: float  # Sharpe比率
    max_drawdown: float  # 最大回撤
    win_rate: float  # 胜率
    turnover: float  # 换手率


@dataclass
class StabilityResult:
    """稳定性验证结果"""
    rolling_betas: pd.Series  # 滚动beta
    rolling_sharpe: pd.Series  # 滚动Sharpe
    cross_sectional_correlation: float  # 横截面相关性
    time_consistency: float  # 时间一致性（beta符号一致性）


class StatisticalValidator:
    """
    统计验证器
    
    验证因子假设的统计显著性、稳定性和有效性
    """
    
    def __init__(self):
        self.validation_results = {}
    
    def single_factor_regression(
        self,
        factor_values: pd.Series,
        forward_returns: pd.Series,
        risk_free_rate: float = 0.0
    ) -> RegressionResult:
        """
        单因子回归
        
        模型：r_{t+1} = α + β*f_t + ε_t
        
        Args:
            factor_values: 因子值序列
            forward_returns: 未来收益序列
            risk_free_rate: 无风险利率
        
        Returns:
            回归结果
        """
        # 对齐数据
        aligned_data = pd.DataFrame({
            'factor': factor_values,
            'return': forward_returns
        }).dropna()
        
        if len(aligned_data) < 10:
            return RegressionResult(
                alpha=0.0, beta=0.0, beta_std=0.0, t_stat=0.0, p_value=1.0,
                r_squared=0.0, sharpe_ratio=0.0, ic_mean=0.0, ic_std=0.0, ic_ir=0.0
            )
        
        X = aligned_data['factor'].values.reshape(-1, 1)
        y = aligned_data['return'].values
        
        # OLS回归
        model = LinearRegression()
        model.fit(X, y)
        
        alpha = model.intercept_
        beta = model.coef_[0]
        
        # 计算统计量
        y_pred = model.predict(X)
        residuals = y - y_pred
        n = len(y)
        mse = np.mean(residuals**2)
        
        # Beta的标准误
        x_mean = np.mean(X)
        x_var = np.var(X, ddof=1)
        beta_std = np.sqrt(mse / ((n - 2) * x_var))
        
        # t统计量和p值
        t_stat = beta / beta_std if beta_std > 0 else 0.0
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
        
        # R²
        r_squared = r2_score(y, y_pred)
        
        # Sharpe比率（基于因子策略收益）
        strategy_returns = beta * aligned_data['factor'].values
        excess_returns = strategy_returns - risk_free_rate
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0.0
        
        # 信息系数（IC）
        ic_values = aligned_data['factor'].corrwith(aligned_data['return'])
        ic_mean = ic_values.mean() if isinstance(ic_values, pd.Series) else ic_values
        ic_std = ic_values.std() if isinstance(ic_values, pd.Series) else 0.0
        ic_ir = ic_mean / ic_std if ic_std > 0 else 0.0
        
        return RegressionResult(
            alpha=float(alpha),
            beta=float(beta),
            beta_std=float(beta_std),
            t_stat=float(t_stat),
            p_value=float(p_value),
            r_squared=float(r_squared),
            sharpe_ratio=float(sharpe_ratio),
            ic_mean=float(ic_mean),
            ic_std=float(ic_std),
            ic_ir=float(ic_ir)
        )
    
    def long_short_portfolio_test(
        self,
        factor_values: pd.Series,
        forward_returns: pd.Series,
        quantile: float = 0.2,
        rebalance_freq: int = 1
    ) -> LongShortResult:
        """
        多空组合测试
        
        策略：做多top quantile，做空bottom quantile
        
        Args:
            factor_values: 因子值序列
            forward_returns: 未来收益序列
            quantile: 分位数阈值（默认0.2表示top/bottom 20%）
            rebalance_freq: 再平衡频率（每N期）
        
        Returns:
            多空组合结果
        """
        # 对齐数据
        aligned_data = pd.DataFrame({
            'factor': factor_values,
            'return': forward_returns
        }).dropna()
        
        if len(aligned_data) < 20:
            return LongShortResult(
                long_return=0.0, short_return=0.0, long_short_return=0.0,
                sharpe_ratio=0.0, max_drawdown=0.0, win_rate=0.0, turnover=0.0
            )
        
        # 计算分位数
        top_threshold = aligned_data['factor'].quantile(1 - quantile)
        bottom_threshold = aligned_data['factor'].quantile(quantile)
        
        # 构建多空组合
        long_mask = aligned_data['factor'] >= top_threshold
        short_mask = aligned_data['factor'] <= bottom_threshold
        
        long_returns = aligned_data[long_mask]['return'].values
        short_returns = aligned_data[short_mask]['return'].values
        
        long_return = np.mean(long_returns) if len(long_returns) > 0 else 0.0
        short_return = np.mean(short_returns) if len(short_returns) > 0 else 0.0
        long_short_return = long_return - short_return
        
        # 计算策略收益序列（简化版）
        strategy_returns = []
        for i in range(0, len(aligned_data), rebalance_freq):
            chunk = aligned_data.iloc[i:i+rebalance_freq]
            if len(chunk) == 0:
                continue
            
            chunk_top = chunk[chunk['factor'] >= chunk['factor'].quantile(1 - quantile)]
            chunk_bottom = chunk[chunk['factor'] <= chunk['factor'].quantile(quantile)]
            
            chunk_long = chunk_top['return'].mean() if len(chunk_top) > 0 else 0.0
            chunk_short = chunk_bottom['return'].mean() if len(chunk_bottom) > 0 else 0.0
            strategy_returns.append(chunk_long - chunk_short)
        
        strategy_returns = np.array(strategy_returns)
        
        # Sharpe比率
        sharpe_ratio = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252) if np.std(strategy_returns) > 0 else 0.0
        
        # 最大回撤
        cumulative = np.cumprod(1 + strategy_returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0.0
        
        # 胜率
        win_rate = np.sum(strategy_returns > 0) / len(strategy_returns) if len(strategy_returns) > 0 else 0.0
        
        # 换手率（简化：基于再平衡频率）
        turnover = 1.0 / rebalance_freq
        
        return LongShortResult(
            long_return=float(long_return),
            short_return=float(short_return),
            long_short_return=float(long_short_return),
            sharpe_ratio=float(sharpe_ratio),
            max_drawdown=float(max_drawdown),
            win_rate=float(win_rate),
            turnover=float(turnover)
        )
    
    def stability_validation(
        self,
        factor_values: pd.Series,
        forward_returns: pd.Series,
        window_size: int = 252,
        step_size: int = 21
    ) -> StabilityResult:
        """
        稳定性验证
        
        在不同时间窗口内滚动检验因子的稳定性
        
        Args:
            factor_values: 因子值序列
            forward_returns: 未来收益序列
            window_size: 滚动窗口大小
            step_size: 步长
        
        Returns:
            稳定性验证结果
        """
        aligned_data = pd.DataFrame({
            'factor': factor_values,
            'return': forward_returns
        }).dropna()
        
        rolling_betas = []
        rolling_sharpe = []
        
        for i in range(0, len(aligned_data) - window_size, step_size):
            window_data = aligned_data.iloc[i:i+window_size]
            
            if len(window_data) < 10:
                continue
            
            # 回归
            X = window_data['factor'].values.reshape(-1, 1)
            y = window_data['return'].values
            
            model = LinearRegression()
            model.fit(X, y)
            beta = model.coef_[0]
            rolling_betas.append(beta)
            
            # Sharpe比率
            strategy_returns = beta * window_data['factor'].values
            sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252) if np.std(strategy_returns) > 0 else 0.0
            rolling_sharpe.append(sharpe)
        
        rolling_betas = pd.Series(rolling_betas)
        rolling_sharpe = pd.Series(rolling_sharpe)
        
        # 时间一致性（beta符号一致性）
        if len(rolling_betas) > 0:
            positive_betas = np.sum(rolling_betas > 0)
            negative_betas = np.sum(rolling_betas < 0)
            time_consistency = max(positive_betas, negative_betas) / len(rolling_betas)
        else:
            time_consistency = 0.0
        
        # 横截面相关性（简化：使用不同子窗口的相关性）
        cross_sectional_corr = 0.0
        if len(rolling_betas) > 1:
            # 计算不同窗口beta之间的相关性
            cross_sectional_corr = rolling_betas.autocorr(lag=1)
        
        return StabilityResult(
            rolling_betas=rolling_betas,
            rolling_sharpe=rolling_sharpe,
            cross_sectional_correlation=float(cross_sectional_corr),
            time_consistency=float(time_consistency)
        )
    
    def comprehensive_validation(
        self,
        factor_values: pd.Series,
        forward_returns: pd.Series,
        hypothesis: FactorHypothesis
    ) -> Dict[str, Any]:
        """
        综合验证
        
        包含所有验证方法
        
        Args:
            factor_values: 因子值序列
            forward_returns: 未来收益序列
            hypothesis: 因子假设
        
        Returns:
            综合验证结果
        """
        results = {}
        
        # 1. 单因子回归
        regression_result = self.single_factor_regression(factor_values, forward_returns)
        results['regression'] = regression_result
        
        # 2. 多空组合测试
        long_short_result = self.long_short_portfolio_test(factor_values, forward_returns)
        results['long_short'] = long_short_result
        
        # 3. 稳定性验证
        stability_result = self.stability_validation(factor_values, forward_returns)
        results['stability'] = stability_result
        
        # 4. 验证结论
        is_valid = (
            regression_result.p_value < 0.05 and  # 统计显著
            abs(regression_result.beta) > 0.01 and  # 系数足够大
            regression_result.sharpe_ratio > 0.5 and  # Sharpe比率合理
            long_short_result.long_short_return > 0 and  # 多空收益为正
            stability_result.time_consistency > 0.6  # 时间一致性
        )
        
        results['is_valid'] = is_valid
        results['hypothesis'] = hypothesis
        
        self.validation_results[hypothesis.name] = results
        return results
    
    def print_validation_results(self, factor_name: Optional[str] = None):
        """打印验证结果"""
        print("\n" + "=" * 80)
        print("Statistical Validation Results")
        print("=" * 80)
        
        factors_to_print = [factor_name] if factor_name else list(self.validation_results.keys())
        
        for factor_name in factors_to_print:
            if factor_name not in self.validation_results:
                continue
            
            results = self.validation_results[factor_name]
            hyp = results['hypothesis']
            
            print(f"\n{factor_name} ({hyp.category.value})")
            print(f"经济学动机: {hyp.economic_motivation}")
            print(f"预期方向: {hyp.expected_direction}")
            
            # 回归结果
            reg = results['regression']
            print(f"\n单因子回归:")
            print(f"  Beta: {reg.beta:.4f} (std: {reg.beta_std:.4f})")
            print(f"  t-stat: {reg.t_stat:.2f}, p-value: {reg.p_value:.4f}")
            print(f"  R²: {reg.r_squared:.4f}")
            print(f"  Sharpe: {reg.sharpe_ratio:.2f}")
            print(f"  IC: {reg.ic_mean:.4f} (IR: {reg.ic_ir:.2f})")
            
            # 多空组合结果
            ls = results['long_short']
            print(f"\n多空组合测试:")
            print(f"  Long Return: {ls.long_return:.4f}")
            print(f"  Short Return: {ls.short_return:.4f}")
            print(f"  Long-Short Return: {ls.long_short_return:.4f}")
            print(f"  Sharpe: {ls.sharpe_ratio:.2f}")
            print(f"  Max Drawdown: {ls.max_drawdown:.4f}")
            print(f"  Win Rate: {ls.win_rate:.2%}")
            
            # 稳定性结果
            stab = results['stability']
            print(f"\n稳定性验证:")
            print(f"  Time Consistency: {stab.time_consistency:.2%}")
            print(f"  Cross-Sectional Correlation: {stab.cross_sectional_correlation:.4f}")
            print(f"  Rolling Beta Mean: {stab.rolling_betas.mean():.4f}")
            print(f"  Rolling Sharpe Mean: {stab.rolling_sharpe.mean():.2f}")
            
            # 验证结论
            is_valid = results['is_valid']
            print(f"\n验证结论: {'✓ 因子有效' if is_valid else '✗ 因子无效'}")
            print("=" * 80)


# ========== 使用示例 ==========

if __name__ == "__main__":
    # 创建示例数据
    np.random.seed(42)
    n = 1000
    
    # 生成因子值（模拟Order Imbalance）
    factor_values = pd.Series(np.random.randn(n).cumsum() * 0.01)
    
    # 生成未来收益（与因子正相关）
    true_beta = 0.5
    forward_returns = true_beta * factor_values + np.random.randn(n) * 0.02
    
    # 创建因子假设
    from Research.factor_hypothesis import FactorHypothesis, FactorCategory
    
    hypothesis = FactorHypothesis(
        name="TestFactor",
        category=FactorCategory.ORDER_IMBALANCE,
        economic_motivation="测试因子",
        formula="TestFactor = X",
        expected_direction="正相关",
        expected_target="return",
        microstructure_basis="测试"
    )
    
    # 验证
    validator = StatisticalValidator()
    results = validator.comprehensive_validation(factor_values, forward_returns, hypothesis)
    
    # 打印结果
    validator.print_validation_results("TestFactor")













