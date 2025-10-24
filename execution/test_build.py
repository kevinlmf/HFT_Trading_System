#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script to verify the HFT Trading System build
"""
import sys
import os

# Add cpp_core to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cpp_core'))

def test_cpp_module():
    """Test the C++ module import and basic functionality"""
    print("=" * 60)
    print("Testing C++ Module Import")
    print("=" * 60)

    try:
        import cpp_trading2
        print("✓ cpp_trading2 module imported successfully")

        # List available classes/functions
        members = [m for m in dir(cpp_trading2) if not m.startswith('_')]
        print(f"✓ Available members: {', '.join(members)}")

        # Test Order creation
        try:
            order = cpp_trading2.Order()
            print("✓ Order class instantiated successfully")
        except Exception as e:
            print(f"✗ Order instantiation failed: {e}")

        # Test DataFeed
        try:
            feed = cpp_trading2.DataFeed()
            print("✓ DataFeed class instantiated successfully")
        except Exception as e:
            print(f"✗ DataFeed instantiation failed: {e}")

        return True

    except ImportError as e:
        print(f"✗ Failed to import cpp_trading2: {e}")
        return False

def test_python_strategies():
    """Test Python strategy modules"""
    print("\n" + "=" * 60)
    print("Testing Python Strategy Modules")
    print("=" * 60)

    success = True

    # Test classical strategies
    try:
        from strategy.classical import MomentumStrategy
        strategy = MomentumStrategy()
        print("✓ MomentumStrategy imported and instantiated")
    except Exception as e:
        print(f"✗ MomentumStrategy failed: {e}")
        success = False

    try:
        from strategy.classical import PairsTradingStrategy
        strategy = PairsTradingStrategy()
        print("✓ PairsTradingStrategy imported and instantiated")
    except Exception as e:
        print(f"✗ PairsTradingStrategy failed: {e}")
        success = False

    try:
        from strategy.classical import MeanVarianceStrategy
        strategy = MeanVarianceStrategy()
        print("✓ MeanVarianceStrategy imported and instantiated")
    except Exception as e:
        print(f"✗ MeanVarianceStrategy failed: {e}")
        success = False

    # Test HFT strategies
    try:
        from strategy.hft_strategies import MarketMakingStrategy
        strategy = MarketMakingStrategy()
        print("✓ MarketMakingStrategy imported and instantiated")
    except Exception as e:
        print(f"✗ MarketMakingStrategy failed: {e}")
        success = False

    return success

def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("HFT Trading System Build Verification")
    print("=" * 60)
    print()

    results = []

    # Test C++ module
    results.append(("C++ Module", test_cpp_module()))

    # Test Python strategies
    results.append(("Python Strategies", test_python_strategies()))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name}: {status}")

    all_passed = all(r[1] for r in results)
    if all_passed:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
