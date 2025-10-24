#!/bin/bash
# HFT Trading System C++ Module Build Script

set -e  # Exit on error

echo "========================================="
echo "Building HFT C++ Trading Module"
echo "========================================="

# Navigate to cpp_core directory
cd "$(dirname "$0")/cpp_core"

echo ""
echo "Step 1: Cleaning previous build..."
rm -rf build
rm -f *.so

echo ""
echo "Step 2: Building with setuptools..."
python setup.py build_ext --inplace

echo ""
echo "Step 3: Verifying the build..."
if [ -f "cpp_trading2.cpython-*.so" ]; then
    echo "✓ C++ module built successfully: $(ls cpp_trading2.cpython-*.so)"
else
    echo "✗ Build failed: .so file not found"
    exit 1
fi

echo ""
echo "Step 4: Testing import..."
cd ../..
python -c "import sys; sys.path.insert(0, 'cpp_core'); import cpp_trading2; print('✓ Module import successful')"

echo ""
echo "========================================="
echo "Build completed successfully!"
echo "========================================="
