"""
策略注册系统

用于管理和动态加载所有策略
"""

from typing import Dict, Type, List, Optional
from .base_strategy import BaseStrategy, StrategyCategory


class StrategyRegistry:
    """策略注册中心"""

    _strategies: Dict[str, Type[BaseStrategy]] = {}
    _category_index: Dict[StrategyCategory, List[str]] = {
        category: [] for category in StrategyCategory
    }

    @classmethod
    def register(cls, strategy_class: Type[BaseStrategy]) -> Type[BaseStrategy]:
        """
        注册策略

        使用装饰器注册：
        @StrategyRegistry.register
        class MyStrategy(BaseStrategy):
            ...
        """
        if not hasattr(strategy_class, 'metadata'):
            raise ValueError(f"Strategy {strategy_class.__name__} must have 'metadata' attribute")

        strategy_name = strategy_class.__name__
        cls._strategies[strategy_name] = strategy_class

        # 添加到分类索引
        metadata = strategy_class.metadata
        if hasattr(metadata, 'category'):
            cls._category_index[metadata.category].append(strategy_name)

        print(f"✓ Registered strategy: {strategy_name}")
        return strategy_class

    @classmethod
    def get_strategy(cls, name: str) -> Optional[Type[BaseStrategy]]:
        """根据名称获取策略类"""
        return cls._strategies.get(name)

    @classmethod
    def list_strategies(cls, category: Optional[StrategyCategory] = None) -> List[str]:
        """列出所有策略或某一类别的策略"""
        if category is None:
            return list(cls._strategies.keys())
        return cls._category_index.get(category, [])

    @classmethod
    def get_all_categories(cls) -> Dict[StrategyCategory, List[str]]:
        """获取所有分类及其策略"""
        return {
            cat: strategies for cat, strategies in cls._category_index.items()
            if len(strategies) > 0
        }

    @classmethod
    def create_strategy(cls, name: str, config: Dict = None) -> Optional[BaseStrategy]:
        """根据名称创建策略实例"""
        strategy_class = cls.get_strategy(name)
        if strategy_class is None:
            raise ValueError(f"Strategy '{name}' not found in registry")

        # 假设策略类有 metadata 类属性
        if hasattr(strategy_class, 'metadata'):
            return strategy_class(strategy_class.metadata, config)
        else:
            raise ValueError(f"Strategy '{name}' missing metadata")

    @classmethod
    def print_registry(cls) -> None:
        """打印所有已注册策略"""
        print("\n" + "="*80)
        print("策略注册表 (Strategy Registry)")
        print("="*80)

        for category, strategies in cls._category_index.items():
            if len(strategies) > 0:
                print(f"\n【{category.value}】")
                for strategy_name in strategies:
                    strategy_class = cls._strategies[strategy_name]
                    if hasattr(strategy_class, 'metadata'):
                        metadata = strategy_class.metadata
                        print(f"  • {strategy_name}")
                        print(f"    描述: {metadata.description}")
                        print(f"    频率: {metadata.target_frequency}")
                    else:
                        print(f"  • {strategy_name}")

        print("\n" + "="*80)
        print(f"总计: {len(cls._strategies)} 个策略")
        print("="*80 + "\n")


# Convenience functions for direct access
def get_strategy(name: str, config: Dict = None) -> BaseStrategy:
    """
    获取并创建策略实例

    Args:
        name: 策略名称
        config: 策略配置（可选）

    Returns:
        策略实例
    """
    return StrategyRegistry.create_strategy(name, config)


def list_strategies(category: Optional[StrategyCategory] = None) -> List[str]:
    """列出所有策略或某一类别的策略"""
    return StrategyRegistry.list_strategies(category)
