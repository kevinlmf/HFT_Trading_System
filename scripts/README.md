# 🚀 快速性能测试 - 使用说明

## ⚠️ 重要：需要先切换到项目目录

```bash
# 1. 切换到项目目录
cd /Users/mengfanlong/Downloads/System/Projects/Quant/HFT/HFT_System

# 2. 然后运行脚本
./scripts/quick_performance_test.sh
```

## 或者使用绝对路径

```bash
# 从任何目录运行
/Users/mengfanlong/Downloads/System/Projects/Quant/HFT/HFT_System/scripts/quick_performance_test.sh
```

## 或者使用Python直接运行

```bash
# 从任何目录运行（如果设置了PYTHONPATH）
cd /Users/mengfanlong/Downloads/System/Projects/Quant/HFT/HFT_System
python3 scripts/benchmark_qdb.py --test loading
```

## 快速测试（一键命令）

```bash
# 复制粘贴这个命令（包含cd）
cd /Users/mengfanlong/Downloads/System/Projects/Quant/HFT/HFT_System && ./scripts/quick_performance_test.sh
```

## 如果还是找不到文件

```bash
# 检查文件是否存在
ls -la /Users/mengfanlong/Downloads/System/Projects/Quant/HFT/HFT_System/scripts/quick_performance_test.sh

# 检查权限
chmod +x /Users/mengfanlong/Downloads/System/Projects/Quant/HFT/HFT_System/scripts/quick_performance_test.sh
```













