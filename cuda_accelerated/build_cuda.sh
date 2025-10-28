#!/bin/bash
# CUDA-Accelerated Backtesting & Monte Carlo - Build Script
# Usage: ./build_cuda.sh [--install] [--test] [--clean]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default options
INSTALL=false
RUN_TESTS=false
CLEAN=false
BUILD_TYPE="Release"

# Parse arguments
for arg in "$@"; do
    case $arg in
        --install)
            INSTALL=true
            shift
            ;;
        --test)
            RUN_TESTS=true
            shift
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --install    Install Python modules after building"
            echo "  --test       Run tests after building"
            echo "  --clean      Clean build directory before building"
            echo "  --debug      Build in debug mode (default: Release)"
            echo "  --help       Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $arg${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "CUDA-Accelerated Build Script"
echo "========================================"
echo ""

# Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"

# Check NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}Error: nvidia-smi not found. Is NVIDIA driver installed?${NC}"
    exit 1
fi

echo "GPU Information:"
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -n 1

# Check CUDA
if ! command -v nvcc &> /dev/null; then
    echo -e "${RED}Error: nvcc not found. Is CUDA Toolkit installed?${NC}"
    echo "Install from: https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
echo "CUDA Version: $CUDA_VERSION"

# Check CMake
if ! command -v cmake &> /dev/null; then
    echo -e "${RED}Error: cmake not found. Please install CMake 3.18+${NC}"
    exit 1
fi

CMAKE_VERSION=$(cmake --version | head -n 1 | sed -n 's/.*version \([0-9]\+\.[0-9]\+\).*/\1/p')
echo "CMake Version: $CMAKE_VERSION"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 not found${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | sed -n 's/Python \([0-9]\+\.[0-9]\+\).*/\1/p')
echo "Python Version: $PYTHON_VERSION"

# Check CuPy
echo ""
echo -e "${YELLOW}Checking Python dependencies...${NC}"

if ! python3 -c "import cupy" 2>/dev/null; then
    echo -e "${YELLOW}Warning: CuPy not installed. Python API will use simulation mode.${NC}"
    echo "Install with: pip install cupy-cuda${CUDA_VERSION/./}x"
else
    CUPY_VERSION=$(python3 -c "import cupy; print(cupy.__version__)")
    echo "CuPy Version: $CUPY_VERSION"
fi

# Check pybind11
if ! python3 -c "import pybind11" 2>/dev/null; then
    echo -e "${RED}Error: pybind11 not installed${NC}"
    echo "Install with: pip install pybind11"
    exit 1
fi

echo ""
echo -e "${GREEN}All prerequisites satisfied!${NC}"
echo ""

# Clean if requested
if [ "$CLEAN" = true ]; then
    echo -e "${YELLOW}Cleaning build directory...${NC}"
    rm -rf build
    echo "Clean complete"
    echo ""
fi

# Create build directory
echo -e "${YELLOW}Creating build directory...${NC}"
mkdir -p build
cd build

# Configure with CMake
echo ""
echo -e "${YELLOW}Configuring with CMake...${NC}"
cmake .. \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89" \
    -DBUILD_TESTS=$RUN_TESTS \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

if [ $? -ne 0 ]; then
    echo -e "${RED}CMake configuration failed${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}Configuration successful!${NC}"
echo ""

# Build
echo -e "${YELLOW}Building CUDA kernels...${NC}"
echo "Build type: $BUILD_TYPE"
echo ""

NUM_CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
echo "Using $NUM_CORES parallel jobs"
echo ""

make -j$NUM_CORES

if [ $? -ne 0 ]; then
    echo -e "${RED}Build failed${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}Build successful!${NC}"
echo ""

# Run tests if requested
if [ "$RUN_TESTS" = true ]; then
    echo -e "${YELLOW}Running tests...${NC}"
    ctest --output-on-failure

    if [ $? -ne 0 ]; then
        echo -e "${RED}Tests failed${NC}"
        exit 1
    fi

    echo -e "${GREEN}All tests passed!${NC}"
    echo ""
fi

# Install if requested
if [ "$INSTALL" = true ]; then
    echo -e "${YELLOW}Installing Python modules...${NC}"
    make install

    if [ $? -ne 0 ]; then
        echo -e "${RED}Installation failed${NC}"
        exit 1
    fi

    echo ""
    echo -e "${GREEN}Installation successful!${NC}"
    echo ""
fi

# Go back to original directory
cd ..

# Verify installation
echo -e "${YELLOW}Verifying installation...${NC}"

if python3 -c "import sys; sys.path.insert(0, 'python'); import cuda_backtest; print('cuda_backtest loaded successfully')" 2>/dev/null; then
    echo -e "${GREEN}Python API verified!${NC}"
else
    echo -e "${YELLOW}Warning: Python API not fully functional. This is OK if CUDA kernels not compiled.${NC}"
fi

echo ""
echo "========================================"
echo -e "${GREEN}Build complete!${NC}"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Test the installation:"
echo "   python3 python/cuda_backtest.py"
echo ""
echo "2. Run examples:"
echo "   python3 examples/example_backtest.py"
echo "   python3 examples/example_monte_carlo.py"
echo ""
echo "3. Read documentation:"
echo "   cat README.md"
echo ""

# Print build summary
echo "Build Summary:"
echo "  Build Type: $BUILD_TYPE"
echo "  CUDA Version: $CUDA_VERSION"
echo "  Install: $INSTALL"
echo "  Tests: $RUN_TESTS"
echo ""
