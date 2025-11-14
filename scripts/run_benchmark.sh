#!/bin/bash
# QDB性能测试脚本
# 快速运行性能测试并查看结果

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo ""
echo -e "${CYAN}============================================================${NC}"
echo -e "${CYAN}     QDB性能基准测试${NC}"
echo -e "${CYAN}============================================================${NC}"
echo ""

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 not found${NC}"
    exit 1
fi

# 运行基准测试
python3 scripts/benchmark_qdb.py --test all

echo ""
echo -e "${GREEN}测试完成！${NC}"
echo ""













