#!/usr/bin/env python3
"""
Test script for the electricity demand forecasting evaluation framework
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluate_gifteval_models import ElectricityDemandEvaluator


def test_data_loading():
    """Test data loading functionality"""
    print("Testing data loading...")
    
    evaluator = ElectricityDemandEvaluator()
    
    # Test demand data loading (will create sample data)
    demand_data = evaluator.load_demand_data()
    print(f"✓ Demand data loaded: {len(demand_data)} records")
    
    # Test weather data loading (will create sample data)
    weather_data = evaluator.load_weather_data()
    print(f"✓ Weather data loaded: {len(weather_data)} records")
    
    # Test data splits
    (train_data, val_data, test_data), (train_features, val_features, test_features) = evaluator.prepare_data_splits()
    
    print(f"✓ Data splits prepared:")
    print(f"  Train: {len(train_data)} records")
    print(f"  Validation: {len(val_data)} records")
    print(f"  Test: {len(test_data)} records")
    print(f"  Features: {train_features.shape[1]} features")
    
    return True


def test_tide_model():
    """Test TiDE model evaluation"""
    print("\nTesting TiDE model...")
    
    try:
        evaluator = ElectricityDemandEvaluator()
        
        # Load and prepare data
        evaluator.load_demand_data()
        (train_data, val_data, test_data), (train_features, val_features, test_features) = evaluator.prepare_data_splits()
        
        # Test TiDE model
        results = evaluator._evaluate_tide_model(
            train_data, val_data, test_data,
            train_features, val_features, test_features
        )
        
        print(f"✓ TiDE model evaluated successfully:")
        for metric, value in results.items():
            print(f"  {metric}: {value:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ TiDE model test failed: {e}")
        return False


def test_full_evaluation():
    """Test full evaluation framework"""
    print("\nTesting full evaluation framework...")
    
    try:
        evaluator = ElectricityDemandEvaluator()
        
        # Define models to test
        models_to_test = ["TiDE"]  # Start with just TiDE
        
        # Run evaluation
        results = evaluator.run_evaluation(models_to_test)
        
        print(f"✓ Full evaluation completed:")
        for model_name, model_results in results.items():
            print(f"  {model_name}:")
            for metric, value in model_results.items():
                print(f"    {metric}: {value:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Full evaluation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("ELECTRICITY DEMAND FORECASTING EVALUATION TESTS")
    print("=" * 60)
    
    tests = [
        ("Data Loading", test_data_loading),
        ("TiDE Model", test_tide_model),
        ("Full Evaluation", test_full_evaluation)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        
        try:
            success = test_func()
            results[test_name] = "PASS" if success else "FAIL"
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results[test_name] = "FAIL"
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result == "PASS" else "✗ FAIL"
        print(f"{test_name}: {status}")
    
    all_passed = all(result == "PASS" for result in results.values())
    
    if all_passed:
        print("\n🎉 All tests passed! The evaluation framework is ready to use.")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
