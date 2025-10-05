#!/usr/bin/env python3
"""
Generate Markdown Report from Evaluation Results
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime


def generate_markdown_report(results_csv: str = "results/evaluation_results.csv",
                            results_json: str = "results/evaluation_results.json",
                            output_file: str = "results/EVALUATION_REPORT.md"):
    """
    Generate a comprehensive markdown report from evaluation results
    """
    
    # Load results
    results_df = pd.read_csv(results_csv, index_col=0)
    
    with open(results_json, 'r') as f:
        results_dict = json.load(f)
    
    # Sort by MAPE (best first)
    results_df = results_df.sort_values('mape')
    
    # Generate report
    report = []
    
    # Header
    report.append("# Electricity Demand Forecasting - Model Evaluation Report")
    report.append("")
    report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    report.append("---")
    report.append("")
    
    # Executive Summary
    report.append("## Executive Summary")
    report.append("")
    report.append("This report presents the evaluation results of multiple time series forecasting models")
    report.append("on electricity demand data from 2016-2024.")
    report.append("")
    
    # Dataset Information
    report.append("### Dataset Information")
    report.append("")
    report.append("- **Data Source:** Electricity demand (15-minute intervals)")
    report.append("- **Training Period:** 2016-2022")
    report.append("- **Validation Period:** 2023")
    report.append("- **Test Period:** 2024")
    report.append("- **Forecast Horizon:** 48 hours")
    report.append("- **Evaluation Stride:** 24 hours")
    report.append("")
    
    # Evaluation Metrics
    report.append("### Evaluation Metrics")
    report.append("")
    report.append("- **MAPE (Mean Absolute Percentage Error):** Lower is better")
    report.append("- **MAE (Mean Absolute Error):** Lower is better")
    report.append("- **RMSE (Root Mean Squared Error):** Lower is better")
    report.append("- **MASE (Mean Absolute Scaled Error):** Lower is better (< 1.0 means better than naive baseline)")
    report.append("- **CRPS (Continuous Ranked Probability Score):** Lower is better")
    report.append("")
    report.append("---")
    report.append("")
    
    # Overall Results
    report.append("## Overall Results")
    report.append("")
    report.append("### Performance Ranking (by MAPE)")
    report.append("")
    
    # Create ranking table
    report.append("| Rank | Model | MAPE (%) | MAE | RMSE | MASE | CRPS |")
    report.append("|------|-------|----------|-----|------|------|------|")
    
    for rank, (model_name, row) in enumerate(results_df.iterrows(), 1):
        mape = row.get('mape', float('nan'))
        mae = row.get('mae', float('nan'))
        rmse = row.get('rmse', float('nan'))
        mase = row.get('mase', float('nan'))
        crps = row.get('crps', float('nan'))
        
        # Format values
        mape_str = f"{mape:.2f}" if pd.notna(mape) and mape != float('inf') else "N/A"
        mae_str = f"{mae:.2f}" if pd.notna(mae) and mae != float('inf') else "N/A"
        rmse_str = f"{rmse:.2f}" if pd.notna(rmse) and rmse != float('inf') else "N/A"
        mase_str = f"{mase:.4f}" if pd.notna(mase) and mase != float('inf') else "N/A"
        crps_str = f"{crps:.2f}" if pd.notna(crps) and crps != float('inf') else "N/A"
        
        # Add medal emoji for top 3
        if rank == 1:
            model_display = f"**{model_name}** 🥇"
        elif rank == 2:
            model_display = f"**{model_name}** 🥈"
        elif rank == 3:
            model_display = f"**{model_name}** 🥉"
        else:
            model_display = model_name
        
        report.append(f"| {rank} | {model_display} | {mape_str} | {mae_str} | {rmse_str} | {mase_str} | {crps_str} |")
    
    report.append("")
    report.append("---")
    report.append("")
    
    # Model Categories
    report.append("## Model Performance by Category")
    report.append("")
    
    # Baseline Models
    baseline_models = ["Naive", "SeasonalNaive", "MovingAverage"]
    baseline_results = results_df[results_df.index.isin(baseline_models)]
    
    if not baseline_results.empty:
        report.append("### Baseline Models")
        report.append("")
        report.append("| Model | MAPE (%) | MASE |")
        report.append("|-------|----------|------|")
        
        for model_name, row in baseline_results.iterrows():
            mape = row.get('mape', float('nan'))
            mase = row.get('mase', float('nan'))
            mape_str = f"{mape:.2f}" if pd.notna(mape) and mape != float('inf') else "N/A"
            mase_str = f"{mase:.4f}" if pd.notna(mase) and mase != float('inf') else "N/A"
            report.append(f"| {model_name} | {mape_str} | {mase_str} |")
        
        report.append("")
    
    # Deep Learning Models
    dl_models = ["TiDE", "PatchTST", "TFT", "NBEATS", "DLinear", "NHiTS"]
    dl_results = results_df[results_df.index.isin(dl_models)]
    
    if not dl_results.empty:
        report.append("### Deep Learning Models")
        report.append("")
        report.append("| Model | MAPE (%) | MAE | RMSE | MASE |")
        report.append("|-------|----------|-----|------|------|")
        
        for model_name, row in dl_results.iterrows():
            mape = row.get('mape', float('nan'))
            mae = row.get('mae', float('nan'))
            rmse = row.get('rmse', float('nan'))
            mase = row.get('mase', float('nan'))
            
            mape_str = f"{mape:.2f}" if pd.notna(mape) and mape != float('inf') else "N/A"
            mae_str = f"{mae:.2f}" if pd.notna(mae) and mae != float('inf') else "N/A"
            rmse_str = f"{rmse:.2f}" if pd.notna(rmse) and rmse != float('inf') else "N/A"
            mase_str = f"{mase:.4f}" if pd.notna(mase) and mase != float('inf') else "N/A"
            
            report.append(f"| {model_name} | {mape_str} | {mae_str} | {rmse_str} | {mase_str} |")
        
        report.append("")
    
    # Foundation Models
    foundation_models = ["TimesFM", "Moirai", "Chronos"]
    foundation_results = results_df[results_df.index.isin(foundation_models)]
    
    if not foundation_results.empty:
        report.append("### Foundation Models (Zero-Shot)")
        report.append("")
        report.append("| Model | MAPE (%) | MAE | RMSE | MASE |")
        report.append("|-------|----------|-----|------|------|")
        
        for model_name, row in foundation_results.iterrows():
            mape = row.get('mape', float('nan'))
            mae = row.get('mae', float('nan'))
            rmse = row.get('rmse', float('nan'))
            mase = row.get('mase', float('nan'))
            
            mape_str = f"{mape:.2f}" if pd.notna(mape) and mape != float('inf') else "N/A"
            mae_str = f"{mae:.2f}" if pd.notna(mae) and mae != float('inf') else "N/A"
            rmse_str = f"{rmse:.2f}" if pd.notna(rmse) and rmse != float('inf') else "N/A"
            mase_str = f"{mase:.4f}" if pd.notna(mase) and mase != float('inf') else "N/A"
            
            report.append(f"| {model_name} | {mape_str} | {mae_str} | {rmse_str} | {mase_str} |")
        
        report.append("")
    
    report.append("---")
    report.append("")
    
    # Key Findings
    report.append("## Key Findings")
    report.append("")
    
    # Best model
    best_model = results_df.index[0]
    best_mape = results_df.iloc[0]['mape']
    
    report.append(f"1. **Best Overall Model:** {best_model} with MAPE of {best_mape:.2f}%")
    report.append("")
    
    # Best baseline
    if not baseline_results.empty:
        best_baseline = baseline_results['mape'].idxmin()
        best_baseline_mape = baseline_results.loc[best_baseline, 'mape']
        report.append(f"2. **Best Baseline Model:** {best_baseline} with MAPE of {best_baseline_mape:.2f}%")
        report.append("")
    
    # MASE analysis
    models_better_than_naive = results_df[results_df['mase'] < 1.0]
    if not models_better_than_naive.empty:
        report.append(f"3. **Models Better Than Naive Baseline (MASE < 1.0):** {len(models_better_than_naive)} models")
        report.append("")
        for model_name in models_better_than_naive.index:
            mase = models_better_than_naive.loc[model_name, 'mase']
            report.append(f"   - {model_name}: MASE = {mase:.4f}")
        report.append("")
    
    # Performance improvement
    if not baseline_results.empty and not dl_results.empty:
        baseline_best_mape = baseline_results['mape'].min()
        dl_best_mape = dl_results['mape'].min()
        improvement = ((baseline_best_mape - dl_best_mape) / baseline_best_mape) * 100
        
        if improvement > 0:
            report.append(f"4. **Deep Learning Improvement:** {improvement:.1f}% MAPE reduction over best baseline")
        report.append("")
    
    report.append("---")
    report.append("")
    
    # Recommendations
    report.append("## Recommendations")
    report.append("")
    report.append(f"1. **Production Deployment:** Consider deploying {best_model} for operational forecasting")
    report.append("")
    report.append("2. **Model Ensemble:** Combine top 3 models for improved robustness")
    report.append("")
    report.append("3. **Further Optimization:**")
    report.append("   - Hyperparameter tuning for top-performing models")
    report.append("   - Feature engineering (additional weather variables, holidays)")
    report.append("   - Model-specific optimizations")
    report.append("")
    
    report.append("---")
    report.append("")
    
    # Technical Details
    report.append("## Technical Details")
    report.append("")
    report.append("### Model Configurations")
    report.append("")
    report.append("- **Input Length:** 192 steps (48 hours)")
    report.append("- **Output Length:** 192 steps (48 hours)")
    report.append("- **Features:** Weather data (temperature, humidity, wind speed) + Calendar features")
    report.append("- **Training:** Early stopping with validation monitoring")
    report.append("- **Hardware:** GPU acceleration where available")
    report.append("")
    
    report.append("### Data Preprocessing")
    report.append("")
    report.append("- **Scaling:** MinMaxScaler for all features")
    report.append("- **Missing Values:** Linear interpolation")
    report.append("- **Timezone:** IST (Indian Standard Time)")
    report.append("")
    
    report.append("---")
    report.append("")
    
    # Footer
    report.append("## Appendix")
    report.append("")
    report.append("### Full Results (JSON)")
    report.append("")
    report.append("```json")
    report.append(json.dumps(results_dict, indent=2))
    report.append("```")
    report.append("")
    
    # Write report
    output_path = Path(output_file)
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"Markdown report generated: {output_path}")
    return output_path


if __name__ == "__main__":
    generate_markdown_report()
