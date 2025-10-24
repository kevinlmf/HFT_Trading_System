# HFT Trading System - Build Guide

## Overview
This guide explains how to build and verify the HFT Trading System C++ components.

## Build Process

### Quick Build
Run the automated build script:
```bash
./build_cpp.sh
```

This script will:
1. Clean previous build artifacts
2. Build the C++ module using setuptools
3. Verify the build was successful
4. Test module import

### Manual Build
If you prefer to build manually:

```bash
cd cpp_core
python setup.py build_ext --inplace
cd ..
```

### Verify Build
Run the test suite to verify everything works:
```bash
python test_build.py
```

## Common Issues & Solutions

### Issue 1: "ModuleNotFoundError: No module named 'cpp_trading2'"

**Solution:** Make sure the module is built and add the cpp_core directory to your Python path:
```python
import sys
sys.path.insert(0, 'cpp_core')
import cpp_trading2
```

### Issue 2: "make: *** No rule to make target '#'"

**Cause:** This error appears when running `make` with comments or invalid characters in the command line.

**Solution:** Use the provided `build_cpp.sh` script instead of running make directly.

### Issue 3: Build fails with "no such file or directory: 'src/order_executor.cpp'"

**Cause:** The setup.py file references a source file that doesn't exist.

**Solution:** This has been fixed in the current setup.py. The file only includes:
- bindings/all_bindings.cpp
- src/data_feed.cpp
- src/order.cpp

### Issue 4: Encoding errors with Chinese/English mixed text

**Solution:** The text has been cleaned up in strategy_comparison.py. All print statements now use English.

## Module Structure

### C++ Module (cpp_trading2)
Built from:
- `bindings/all_bindings.cpp` - Python bindings
- `src/data_feed.cpp` - Data feed implementation
- `src/order.cpp` - Order management

Available classes:
- `Order` - Order representation
- `DataFeed` - Market data feed
- `FastOrderBook` - High-performance order book
- `OrderExecutor` - Order execution engine
- `HFTOrder`, `Fill` - Trading primitives
- `OrderType`, `OrderSide`, `HFTOrderType` - Enums

### Python Strategies
- `strategy.momentum_strategy` - Momentum-based trading
- `strategy.pairs_trading` - Statistical arbitrage
- `strategy.mean_variance` - Mean-variance optimization

## Performance Notes

The C++ module is compiled with:
- `-O3` optimization
- C++17 standard
- Platform-specific optimizations

## Troubleshooting

If you encounter any build issues:

1. **Check Python version**: Ensure you're using Python 3.10+
2. **Verify dependencies**: Make sure pybind11 is installed
   ```bash
   pip install pybind11
   ```
3. **Clean build**: Remove all build artifacts and rebuild
   ```bash
   cd cpp_core
   rm -rf build *.so
   python setup.py build_ext --inplace
   ```
4. **Check compiler**: Ensure clang/gcc is available
   ```bash
   which clang
   ```

## Development Workflow

1. **Modify C++ code** in `cpp_core/src/` or `cpp_core/include/`
2. **Rebuild** using `./build_cpp.sh`
3. **Test** using `python test_build.py`
4. **Use** in your Python code:
   ```python
   import sys
   sys.path.insert(0, 'cpp_core')
   import cpp_trading2

   # Use the module
   order = cpp_trading2.Order()
   feed = cpp_trading2.DataFeed()
   ```

## Additional Resources

- See `README.md` for overall system documentation
- Check `cpp_core/bindings/` for Python binding definitions
- Review `cpp_core/include/` for C++ API documentation

## Support

If you continue to experience issues, please check:
1. The error message carefully
2. Python version compatibility
3. Required dependencies are installed
4. File paths are correct
