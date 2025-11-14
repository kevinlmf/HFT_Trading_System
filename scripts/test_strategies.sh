#!/bin/bash
# 快速测试策略是否已注册

cd /Users/mengfanlong/Downloads/System/Projects/Quant/HFT/HFT_System

echo "测试策略注册..."
echo ""

python3 << 'EOF'
import sys
sys.path.insert(0, '.')

# 导入策略适配器（这会注册策略）
from Execution.strategies.strategy_adapters import MomentumStrategyAdapter, MeanReversionStrategyAdapter
from Execution.strategies.strategy_registry import get_strategy, StrategyRegistry

print("已注册的策略:")
strategies = StrategyRegistry.list_strategies()
for s in strategies:
    print(f"  - {s}")

print("\n测试别名查找:")
test_names = ['momentum', 'mean_reversion', 'MomentumStrategyAdapter']

for name in test_names:
    try:
        strategy = get_strategy(name)
        if strategy:
            print(f"  ✓ '{name}' -> {strategy.__class__.__name__}")
        else:
            print(f"  ✗ '{name}' -> Not found")
    except Exception as e:
        print(f"  ✗ '{name}' -> Error: {e}")

print("\n✅ 策略注册测试完成！")
EOF













