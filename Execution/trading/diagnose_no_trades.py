"""
Diagnostic script to identify why no trades are executing
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

def diagnose_trading_engine():
    """Diagnose why trades are not executing"""
    
    print("="*80)
    print("Trading Engine Diagnostic")
    print("="*80)
    
    print("\n1. Checking Confidence Thresholds...")
    print("   Current thresholds:")
    print("     - Strategy Router: >= 0.2 (line 463)")
    print("     - Signal Confidence: >= 0.25 (line 588)")
    print("     - Signal Strength: > 0.05 (line 594/597)")
    print("\n   Recommendation: Lower these thresholds for more trades")
    
    print("\n2. Checking Data Requirements...")
    print("   Minimum data points: 10 (line 521)")
    print("   Strategy lookback: Usually 20+ periods")
    print("\n   Recommendation: Ensure sufficient data collection")
    
    print("\n3. Checking Validation Settings...")
    print("   Default: enable_validation=True")
    print("   Validation timeout: 500ms")
    print("   Minimum ticks for validation: 50")
    print("\n   Recommendation: Set enable_validation=False for testing")
    
    print("\n4. Common Issues:")
    print("   - Yahoo Finance connector has 15-20min delay (not suitable for HFT)")
    print("   - Data collection too slow (increase update frequency)")
    print("   - Thresholds too conservative (lower confidence/stength requirements)")
    print("   - Strategy not generating signals (check strategy implementation)")
    
    print("\n" + "="*80)
    print("Recommended Fixes")
    print("="*80)
    print("\nOption 1: Lower thresholds in trading_engine.py")
    print("Option 2: Use real-time connector (Alpaca, Polygon)")
    print("Option 3: Disable validation for faster execution")
    print("Option 4: Increase update frequency")

if __name__ == "__main__":
    diagnose_trading_engine()

