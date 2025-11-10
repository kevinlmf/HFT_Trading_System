"""
Intelligent Trading Pipeline
完整的智能交易处理流程：EDA -> 数据清理 -> 智能执行
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple, Any
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Data.eda.data_analyzer import DataAnalyzer, DataProfile, DataQuality
from Execution.engine.smart_executor import SmartExecutor


class TradingPipeline:
    """
    智能交易处理管道
    
    流程：
    1. EDA - 探索性数据分析
    2. 数据清理 - 根据EDA结果清理数据
    3. 智能执行 - 根据数据规模选择最优实现
    4. 结果验证 - 验证结果质量
    """
    
    def __init__(self):
        self.analyzer = DataAnalyzer()
        self.executor = SmartExecutor()
        self.pipeline_history: List[Dict] = []
    
    def process(self,
               data: pd.DataFrame,
               data_type: str = "orders",
               auto_clean: bool = True,
               force_implementation: Optional[str] = None) -> Dict:
        """
        完整处理流程
        
        Args:
            data: 输入数据
            data_type: 数据类型
            auto_clean: 是否自动清理数据
            force_implementation: 强制使用指定实现
        
        Returns:
            处理结果字典
        """
        pipeline_step = {
            'data_type': data_type,
            'original_size': len(data),
            'steps': []
        }
        
        # Step 1: EDA
        print("\n" + "="*80)
        print("Step 1: Exploratory Data Analysis (EDA)")
        print("="*80)
        profile = self.analyzer.analyze(data, data_name=data_type)
        self.analyzer.print_report(profile, data_type)
        pipeline_step['steps'].append({
            'step': 'EDA',
            'profile': profile
        })
        
        # Step 2: 数据清理（如果需要）
        cleaned_data = data.copy()
        if auto_clean and profile.quality != DataQuality.EXCELLENT:
            print("\n" + "="*80)
            print("Step 2: Data Cleaning")
            print("="*80)
            cleaned_data = self._clean_data(data, profile)
            print(f"清理前: {len(data):,} 条记录")
            print(f"清理后: {len(cleaned_data):,} 条记录")
            print(f"清理率: {(1 - len(cleaned_data)/len(data))*100:.2f}%")
            
            # 重新分析清理后的数据
            profile = self.analyzer.analyze(cleaned_data, data_name=f"{data_type}_cleaned")
            pipeline_step['steps'].append({
                'step': 'Cleaning',
                'cleaned_size': len(cleaned_data),
                'profile': profile
            })
        
        # Step 3: 智能执行
        print("\n" + "="*80)
        print("Step 3: Smart Execution")
        print("="*80)
        
        # 提取slippage计算所需的数据
        execution_result = self._execute_slippage(cleaned_data, profile, force_implementation)
        pipeline_step['steps'].append({
            'step': 'Execution',
            'result': execution_result
        })
        
        # Step 4: 结果验证
        print("\n" + "="*80)
        print("Step 4: Result Validation")
        print("="*80)
        validation = self._validate_results(execution_result)
        pipeline_step['steps'].append({
            'step': 'Validation',
            'validation': validation
        })
        
        pipeline_step['final_size'] = len(cleaned_data)
        pipeline_step['success'] = validation['is_valid']
        
        self.pipeline_history.append(pipeline_step)
        
        # 打印总结
        self._print_summary(pipeline_step, execution_result)
        
        return {
            'profile': profile,
            'cleaned_data': cleaned_data,
            'execution_result': execution_result,
            'validation': validation,
            'pipeline_info': pipeline_step
        }
    
    def _clean_data(self, data: pd.DataFrame, profile: DataProfile) -> pd.DataFrame:
        """根据EDA结果清理数据"""
        cleaned = data.copy()
        
        # 移除重复值
        if profile.duplicate_ratio > 0:
            cleaned = cleaned.drop_duplicates()
        
        # 处理缺失值
        if profile.missing_ratio > 0:
            # 对于数值列，使用中位数填充
            numeric_cols = cleaned.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if cleaned[col].isnull().sum() > 0:
                    cleaned[col].fillna(cleaned[col].median(), inplace=True)
            
            # 对于分类型列，使用众数填充
            categorical_cols = cleaned.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                if cleaned[col].isnull().sum() > 0:
                    mode_value = cleaned[col].mode()
                    if len(mode_value) > 0:
                        cleaned[col].fillna(mode_value[0], inplace=True)
        
        # 处理异常值（使用IQR方法）
        if profile.outlier_ratio > 0:
            numeric_cols = cleaned.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                Q1 = cleaned[col].quantile(0.25)
                Q3 = cleaned[col].quantile(0.75)
                IQR = Q3 - Q1
                if IQR > 0:
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    # 将异常值替换为边界值
                    cleaned[col] = cleaned[col].clip(lower=lower_bound, upper=upper_bound)
        
        return cleaned
    
    def _execute_slippage(self,
                          data: pd.DataFrame,
                          profile: DataProfile,
                          force_implementation: Optional[str] = None) -> Dict:
        """执行slippage计算"""
        # 提取所需列
        required_cols = ['price', 'quantity', 'mid_price', 'side']
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            raise ValueError(f"缺少必需的列: {missing_cols}")
        
        prices = data['price'].values
        quantities = data['quantity'].values
        mid_prices = data['mid_price'].values
        sides = data['side'].values if 'side' in data.columns else np.ones(len(data))
        
        # 使用智能执行器
        slippage_costs, exec_info = self.executor.execute_slippage_calculation(
            prices, quantities, mid_prices, sides,
            profile=profile,
            force_implementation=force_implementation
        )
        
        print(f"实现方式: {exec_info['implementation']}")
        print(f"执行时间: {exec_info['execution_time']*1000:.2f} ms")
        print(f"吞吐量: {exec_info['throughput']:,.0f} orders/sec")
        
        return {
            'slippage_costs': slippage_costs,
            'execution_info': exec_info,
            'n_orders': len(prices)
        }
    
    def _validate_results(self, execution_result: Dict) -> Dict:
        """验证结果质量"""
        slippage_costs = execution_result['slippage_costs']
        
        validation = {
            'is_valid': True,
            'warnings': [],
            'errors': []
        }
        
        # 检查是否有NaN
        if np.isnan(slippage_costs).any():
            validation['is_valid'] = False
            validation['errors'].append("结果中包含NaN值")
        
        # 检查是否有负值
        if (slippage_costs < 0).any():
            validation['warnings'].append("结果中包含负值（slippage应为非负）")
        
        # 检查是否有异常大的值
        if len(slippage_costs) > 0:
            q99 = np.percentile(slippage_costs, 99)
            q01 = np.percentile(slippage_costs, 1)
            if q99 > q01 * 1000:  # 99分位数超过1分位数的1000倍
                validation['warnings'].append("结果中存在异常大的值")
        
        # 检查结果数量
        if len(slippage_costs) != execution_result['n_orders']:
            validation['is_valid'] = False
            validation['errors'].append(f"结果数量不匹配: 期望 {execution_result['n_orders']}, 实际 {len(slippage_costs)}")
        
        if validation['is_valid']:
            print("✓ 结果验证通过")
        else:
            print("✗ 结果验证失败")
            for error in validation['errors']:
                print(f"  错误: {error}")
        
        if validation['warnings']:
            for warning in validation['warnings']:
                print(f"  警告: {warning}")
        
        return validation
    
    def _print_summary(self, pipeline_step: Dict, execution_result: Dict):
        """打印处理总结"""
        print("\n" + "="*80)
        print("Pipeline Summary")
        print("="*80)
        print(f"原始数据: {pipeline_step['original_size']:,} 条记录")
        print(f"最终数据: {pipeline_step['final_size']:,} 条记录")
        print(f"处理状态: {'成功' if pipeline_step['success'] else '失败'}")
        
        exec_info = execution_result['execution_info']
        print(f"\n执行信息:")
        print(f"  实现方式: {exec_info['implementation']}")
        print(f"  处理订单数: {exec_info['n_orders']:,}")
        print(f"  执行时间: {exec_info['execution_time']*1000:.2f} ms")
        print(f"  吞吐量: {exec_info['throughput']:,.0f} orders/sec")
        
        stats = self.executor.get_stats()
        print(f"\n累计统计:")
        print(f"  Python调用: {stats['python_calls']} ({stats['python_ratio']*100:.1f}%)")
        print(f"  C++调用: {stats['cpp_calls']} ({stats['cpp_ratio']*100:.1f}%)")
        print(f"  CUDA调用: {stats['cuda_calls']} ({stats['cuda_ratio']*100:.1f}%)")
        print(f"  总订单数: {stats['total_orders']:,}")
        print("="*80)


def create_sample_data(n_records: int = 10000, seed: int = 42) -> pd.DataFrame:
    """
    创建用于测试的示例交易数据。

    生成带有温和趋势和受控波动率的价格序列，避免完全随机导致的极端噪声，
    让策略回测与风险指标更加稳定。
    """
    rng = np.random.default_rng(seed)
    dt = 1 / 252  # 假设一年 252 个交易日
    annual_mu = 0.08
    annual_sigma = 0.18

    prices = [100.0]
    for _ in range(1, n_records):
        drift = (annual_mu - 0.5 * annual_sigma**2) * dt
        shock = annual_sigma * np.sqrt(dt) * rng.standard_normal()
        next_price = prices[-1] * np.exp(drift + shock)
        prices.append(next_price)

    prices = np.clip(prices, 20.0, 300.0)

    quantities = rng.uniform(10.0, 1000.0, size=n_records)
    mid_spread = rng.normal(0.0, 0.05, size=n_records)

    data = pd.DataFrame({
        'price': prices,
        'quantity': quantities,
        'mid_price': np.array(prices) + mid_spread,
        'side': rng.choice([1, -1], size=n_records, p=[0.52, 0.48])
    })

    return data


if __name__ == "__main__":
    # 示例使用
    pipeline = TradingPipeline()
    
    # 创建测试数据
    data = create_sample_data(n_records=50000)
    
    # 运行完整流程
    result = pipeline.process(data, data_type="orders", auto_clean=True)
    
    print("\n处理完成！")



