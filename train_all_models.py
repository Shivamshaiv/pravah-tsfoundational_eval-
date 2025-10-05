#!/usr/bin/env python3
"""
Train All Models and Generate Report

This script:
1. Runs evaluation on all implemented models
2. Saves results to CSV and JSON
3. Generates a comprehensive markdown report
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from evaluate_gifteval_models import main as run_evaluation
from generate_report import generate_markdown_report


def main():
    """
    Main training and evaluation pipeline
    """
    print("=" * 80)
    print("ELECTRICITY DEMAND FORECASTING - MODEL EVALUATION")
    print("=" * 80)
    print()
    
    print("Starting model evaluation...")
    print()
    
    # Step 1: Run evaluation on all models
    print("Step 1: Running evaluation on all models...")
    print("-" * 80)
    
    try:
        results = run_evaluation()
        print()
        print("Evaluation completed successfully!")
        print()
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 2: Generate markdown report
    print("Step 2: Generating markdown report...")
    print("-" * 80)
    
    try:
        report_path = generate_markdown_report()
        print()
        print(f"Report generated successfully: {report_path}")
        print()
    except Exception as e:
        print(f"Error generating report: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Final summary
    print("=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print()
    print("Output files:")
    print("  - results/evaluation_results.csv")
    print("  - results/evaluation_results.json")
    print("  - results/EVALUATION_REPORT.md")
    print()
    print("To view the report:")
    print("  cat results/EVALUATION_REPORT.md")
    print()


if __name__ == "__main__":
    main()
