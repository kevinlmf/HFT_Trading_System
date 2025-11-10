"""
Exploratory Data Analysis (EDA) Module
分析数据质量、规模和特征，为后续处理选择最优实现方式
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


class DataQuality(Enum):
    """数据质量等级"""
    EXCELLENT = "excellent"  # 数据完整，质量高
    GOOD = "good"            # 数据基本完整，少量缺失
    FAIR = "fair"            # 数据有缺失，需要处理
    POOR = "poor"            # 数据质量差，不建议使用


class DataScale(Enum):
    """数据规模等级"""
    SMALL = "small"          # < 10K 订单/记录
    MEDIUM = "medium"        # 10K - 100K
    LARGE = "large"         # 100K - 1M
    VERY_LARGE = "very_large"  # > 1M


@dataclass
class DataProfile:
    """数据特征分析结果"""
    # 基本信息
    n_records: int
    n_features: int
    memory_size_mb: float
    
    # 数据质量
    quality: DataQuality
    missing_ratio: float
    duplicate_ratio: float
    outlier_ratio: float
    
    # 数据特征
    scale: DataScale
    is_time_series: bool
    has_numeric: bool
    has_categorical: bool
    
    # 统计信息
    numeric_stats: Optional[Dict] = None
    categorical_stats: Optional[Dict] = None
    
    # 建议
    recommended_implementation: str = "python_vectorized"
    processing_time_estimate: float = 0.0  # 秒


class DataAnalyzer:
    """
    数据分析和特征提取器
    
    功能：
    1. 分析数据质量（缺失值、重复值、异常值）
    2. 评估数据规模
    3. 检测数据特征（时间序列、数值型、分类型）
    4. 推荐最优实现方式（Python/C++/CUDA）
    """
    
    def __init__(self):
        self.profiles: Dict[str, DataProfile] = {}
    
    def analyze(self, 
                data: pd.DataFrame,
                data_name: str = "data",
                target_column: Optional[str] = None) -> DataProfile:
        """
        全面分析数据
        
        Args:
            data: 输入数据DataFrame
            data_name: 数据名称
            target_column: 目标列名（用于特殊分析）
        
        Returns:
            DataProfile对象
        """
        n_records = len(data)
        n_features = len(data.columns)
        memory_size_mb = data.memory_usage(deep=True).sum() / 1024 / 1024
        
        # 数据质量分析
        quality, missing_ratio, duplicate_ratio, outlier_ratio = self._analyze_quality(data)
        
        # 数据规模评估
        scale = self._assess_scale(n_records)
        
        # 数据特征检测
        is_time_series = self._is_time_series(data)
        has_numeric, has_categorical = self._detect_data_types(data)
        
        # 统计信息
        numeric_stats = self._get_numeric_stats(data) if has_numeric else None
        categorical_stats = self._get_categorical_stats(data) if has_categorical else None
        
        # 推荐实现方式
        recommended_impl = self._recommend_implementation(
            scale, n_records, memory_size_mb, quality
        )
        
        # 估算处理时间
        processing_time = self._estimate_processing_time(
            n_records, recommended_impl
        )
        
        profile = DataProfile(
            n_records=n_records,
            n_features=n_features,
            memory_size_mb=memory_size_mb,
            quality=quality,
            missing_ratio=missing_ratio,
            duplicate_ratio=duplicate_ratio,
            outlier_ratio=outlier_ratio,
            scale=scale,
            is_time_series=is_time_series,
            has_numeric=has_numeric,
            has_categorical=has_categorical,
            numeric_stats=numeric_stats,
            categorical_stats=categorical_stats,
            recommended_implementation=recommended_impl,
            processing_time_estimate=processing_time
        )
        
        self.profiles[data_name] = profile
        return profile
    
    def _analyze_quality(self, data: pd.DataFrame) -> Tuple[DataQuality, float, float, float]:
        """分析数据质量"""
        n_total = len(data)
        
        # 缺失值分析
        missing_count = data.isnull().sum().sum()
        missing_ratio = missing_count / (n_total * len(data.columns))
        
        # 重复值分析
        duplicate_count = data.duplicated().sum()
        duplicate_ratio = duplicate_count / n_total if n_total > 0 else 0
        
        # 异常值分析（仅对数值列）
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        outlier_ratio = 0.0
        if len(numeric_cols) > 0:
            outlier_counts = []
            for col in numeric_cols:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                if IQR > 0:
                    outliers = ((data[col] < (Q1 - 1.5 * IQR)) | 
                               (data[col] > (Q3 + 1.5 * IQR))).sum()
                    outlier_counts.append(outliers)
            if outlier_counts:
                outlier_ratio = sum(outlier_counts) / (n_total * len(numeric_cols))
        
        # 质量评级
        if missing_ratio < 0.01 and duplicate_ratio < 0.01 and outlier_ratio < 0.05:
            quality = DataQuality.EXCELLENT
        elif missing_ratio < 0.05 and duplicate_ratio < 0.05 and outlier_ratio < 0.10:
            quality = DataQuality.GOOD
        elif missing_ratio < 0.20 and duplicate_ratio < 0.10:
            quality = DataQuality.FAIR
        else:
            quality = DataQuality.POOR
        
        return quality, missing_ratio, duplicate_ratio, outlier_ratio
    
    def _assess_scale(self, n_records: int) -> DataScale:
        """评估数据规模"""
        if n_records < 10000:
            return DataScale.SMALL
        elif n_records < 100000:
            return DataScale.MEDIUM
        elif n_records < 1000000:
            return DataScale.LARGE
        else:
            return DataScale.VERY_LARGE
    
    def _is_time_series(self, data: pd.DataFrame) -> bool:
        """检测是否为时间序列数据"""
        # 检查索引是否为时间类型
        if isinstance(data.index, pd.DatetimeIndex):
            return True
        
        # 检查是否有时间相关的列
        time_keywords = ['time', 'date', 'timestamp', 'datetime']
        for col in data.columns:
            if any(keyword in col.lower() for keyword in time_keywords):
                if pd.api.types.is_datetime64_any_dtype(data[col]):
                    return True
        
        return False
    
    def _detect_data_types(self, data: pd.DataFrame) -> Tuple[bool, bool]:
        """检测数据类型"""
        has_numeric = len(data.select_dtypes(include=[np.number]).columns) > 0
        has_categorical = len(data.select_dtypes(include=['object', 'category']).columns) > 0
        return has_numeric, has_categorical
    
    def _get_numeric_stats(self, data: pd.DataFrame) -> Dict:
        """获取数值型统计信息"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return {}
        
        stats = {}
        for col in numeric_cols:
            stats[col] = {
                'mean': data[col].mean(),
                'std': data[col].std(),
                'min': data[col].min(),
                'max': data[col].max(),
                'median': data[col].median(),
                'skewness': data[col].skew(),
                'kurtosis': data[col].kurtosis()
            }
        return stats
    
    def _get_categorical_stats(self, data: pd.DataFrame) -> Dict:
        """获取分类型统计信息"""
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) == 0:
            return {}
        
        stats = {}
        for col in categorical_cols:
            value_counts = data[col].value_counts()
            stats[col] = {
                'n_unique': data[col].nunique(),
                'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
                'frequency': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                'entropy': self._calculate_entropy(value_counts)
            }
        return stats
    
    def _calculate_entropy(self, value_counts: pd.Series) -> float:
        """计算熵（信息量）"""
        probabilities = value_counts / value_counts.sum()
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return entropy
    
    def _recommend_implementation(self,
                                 scale: DataScale,
                                 n_records: int,
                                 memory_size_mb: float,
                                 quality: DataQuality) -> str:
        """
        根据数据特征推荐最优实现方式
        
        决策逻辑：
        - Small (<10K): Python向量化（简单快速）
        - Medium (10K-100K): Python向量化或C++（根据质量）
        - Large (100K-1M): C++（性能优先）
        - Very Large (>1M): CUDA（GPU加速）
        """
        # 如果数据质量差，先清理，使用Python
        if quality == DataQuality.POOR:
            return "python_vectorized"  # 先用Python清理
        
        if scale == DataScale.SMALL:
            return "python_vectorized"
        elif scale == DataScale.MEDIUM:
            # 中等规模，根据内存和记录数决定
            if memory_size_mb < 100 and n_records < 50000:
                return "python_vectorized"
            else:
                return "cpp"  # 推荐C++以获得更好性能
        elif scale == DataScale.LARGE:
            return "cpp"  # 大规模使用C++
        else:  # VERY_LARGE
            return "cuda"  # 超大规模使用CUDA
    
    def _estimate_processing_time(self, n_records: int, implementation: str) -> float:
        """估算处理时间（秒）"""
        # 基于基准测试结果的经验值
        if implementation == "python_vectorized":
            # 约62M orders/sec
            return n_records / 62_000_000
        elif implementation == "cpp":
            # 约100M orders/sec（估计）
            return n_records / 100_000_000
        elif implementation == "cuda":
            # 约200M orders/sec（估计）
            return n_records / 200_000_000
        else:
            # Python循环，约1M orders/sec
            return n_records / 1_000_000
    
    def print_report(self, profile: DataProfile, data_name: str = "Data"):
        """打印分析报告"""
        print("=" * 80)
        print(f"{data_name} Analysis Report")
        print("=" * 80)
        print(f"\n基本信息:")
        print(f"  记录数: {profile.n_records:,}")
        print(f"  特征数: {profile.n_features}")
        print(f"  内存占用: {profile.memory_size_mb:.2f} MB")
        
        print(f"\n数据质量:")
        print(f"  质量等级: {profile.quality.value.upper()}")
        print(f"  缺失值比例: {profile.missing_ratio*100:.2f}%")
        print(f"  重复值比例: {profile.duplicate_ratio*100:.2f}%")
        print(f"  异常值比例: {profile.outlier_ratio*100:.2f}%")
        
        print(f"\n数据特征:")
        print(f"  规模等级: {profile.scale.value.upper()}")
        print(f"  时间序列: {'是' if profile.is_time_series else '否'}")
        print(f"  数值型特征: {'是' if profile.has_numeric else '否'}")
        print(f"  分类型特征: {'是' if profile.has_categorical else '否'}")
        
        print(f"\n推荐方案:")
        print(f"  实现方式: {profile.recommended_implementation}")
        print(f"  预计处理时间: {profile.processing_time_estimate:.3f} 秒")
        
        if profile.numeric_stats:
            print(f"\n数值型统计 (前3列):")
            for i, (col, stats) in enumerate(list(profile.numeric_stats.items())[:3]):
                print(f"  {col}:")
                print(f"    均值: {stats['mean']:.4f}, 标准差: {stats['std']:.4f}")
                print(f"    范围: [{stats['min']:.4f}, {stats['max']:.4f}]")
        
        print("=" * 80)


def analyze_trading_data(data: pd.DataFrame, 
                         data_type: str = "orders") -> DataProfile:
    """
    专门用于交易数据的分析函数
    
    Args:
        data: 交易数据DataFrame
        data_type: 数据类型 ('orders', 'market_data', 'fills', etc.)
    
    Returns:
        DataProfile对象
    """
    analyzer = DataAnalyzer()
    profile = analyzer.analyze(data, data_name=data_type)
    analyzer.print_report(profile, data_type)
    return profile






