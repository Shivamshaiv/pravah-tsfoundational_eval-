#!/usr/bin/env python3
"""
Run comprehensive model evaluation on electricity demand data
"""

import sys
import argparse
from evaluate_gifteval_models import ElectricityDemandEvaluator


def run_baseline_evaluation():
    """Run evaluation with baseline models only"""
    print("=" * 80)
    print("RUNNING BASELINE MODELS EVALUATION")
    print("=" * 80)
    
    evaluator = ElectricityDemandEvaluator()
    evaluator.load_demand_data()
    
    models = ["Naive", "SeasonalNaive", "MovingAverage"]
    results = evaluator.run_evaluation(models)
    
    return results


def run_tide_evaluation():
    """Run evaluation with TiDE model"""
    print("=" * 80)
    print("RUNNING TIDE MODEL EVALUATION")
    print("=" * 80)
    
    evaluator = ElectricityDemandEvaluator()
    evaluator.load_demand_data()
    
    models = ["TiDE"]
    results = evaluator.run_evaluation(models)
    
    return results


def run_full_evaluation():
    """Run full evaluation with all models"""
    print("=" * 80)
    print("RUNNING FULL MODEL EVALUATION")
    print("=" * 80)
    
    evaluator = ElectricityDemandEvaluator()
    evaluator.load_demand_data()
    
    models = [
        "Naive",
        "SeasonalNaive",
        "MovingAverage",
        "TiDE"
    ]
    
    results = evaluator.run_evaluation(models)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Run electricity demand forecasting evaluation')
    parser.add_argument('--mode', type=str, default='baseline',
                       choices=['baseline', 'tide', 'full'],
                       help='Evaluation mode: baseline (fast), tide (slow), or full (all models)')
    
    args = parser.parse_args()
    
    if args.mode == 'baseline':
        results = run_baseline_evaluation()
    elif args.mode == 'tide':
        results = run_tide_evaluation()
    else:
        results = run_full_evaluation()
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE!")
    print("=" * 80)
    print(f"Results saved to: results/evaluation_results.csv")
    print(f"Results saved to: results/evaluation_results.json")
    
    return results


if __name__ == "__main__":
    results = main()
