#!/usr/bin/env python3
"""
Test database connection and analyst loading functionality.
"""
import sys
import os
import asyncio

# Add the peak_detection directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_database_connection():
    """Test database connection and analyst query"""
    try:
        from config import get_analysts_from_database, get_async_database_uri
        
        print("Testing database connection...")
        
        # Test database URI
        try:
            db_uri = get_async_database_uri()
            print(f"Database URI configured: {db_uri[:50]}...")
        except Exception as e:
            print(f"Database URI error: {e}")
            return False
        
        # Test analyst query
        analysts = await get_analysts_from_database()
        print(f"Query executed successfully")
        print(f"   Found {len(analysts)} analysts: {analysts[:5] if len(analysts) > 5 else analysts}")
        
        return True
        
    except Exception as e:
        print(f"Database test failed: {e}")
        return False

def test_threshold_loading():
    """Test threshold loading without event loop"""
    try:
        from config import load_strategy_thresholds
        
        print("\nTesting threshold loading (no event loop)...")
        thresholds = load_strategy_thresholds()
        
        print(f"Thresholds loaded: {len(thresholds)} strategies")
        
        # Show sample thresholds
        sample_count = 0
        for strategy, threshold in thresholds.items():
            if sample_count < 3:
                print(f"   {strategy}: {threshold}")
                sample_count += 1
            else:
                break
        
        return True
        
    except Exception as e:
        print(f"Threshold loading failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("Testing database connectivity and analyst loading...\n")
    
    # Test database connection
    db_success = await test_database_connection()
    
    # Test threshold loading
    threshold_success = test_threshold_loading()
    
    if db_success and threshold_success:
        print("\nAll tests passed! Database connection works.")
    else:
        print("\nSome tests failed. Check database configuration.")

if __name__ == "__main__":
    asyncio.run(main())
