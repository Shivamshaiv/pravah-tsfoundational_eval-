#!/usr/bin/env python3
"""
Script to check if data has been uploaded and verify its format
"""

import os
import sys
from pathlib import Path
import pandas as pd

def check_data_upload():
    """Check if data has been uploaded"""
    print("=" * 60)
    print("DATA UPLOAD CHECK")
    print("=" * 60)
    
    data_dir = Path("data")
    
    # Check if data directory exists
    if not data_dir.exists():
        print("✗ Data directory does not exist")
        print("  Run: mkdir -p data")
        return False
    
    print(f"✓ Data directory exists: {data_dir.absolute()}")
    
    # Check for uploaded zip file
    zip_file = Path("data_upload.zip")
    if zip_file.exists():
        size_mb = zip_file.stat().st_size / (1024 * 1024)
        print(f"✓ Zip file found: {zip_file} ({size_mb:.2f} MB)")
    else:
        print("✗ Zip file not found: data_upload.zip")
        print("  Please upload your data file and rename it to 'data_upload.zip'")
        print("  See UPLOAD_INSTRUCTIONS.md for details")
    
    # List files in data directory
    print(f"\nFiles in {data_dir}:")
    files = list(data_dir.glob("*"))
    
    if not files:
        print("  (empty)")
        return False
    
    for f in sorted(files):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  - {f.name} ({size_mb:.2f} MB)")
        elif f.is_dir():
            print(f"  - {f.name}/ (directory)")
    
    # Look for CSV files
    csv_files = list(data_dir.glob("**/*.csv"))
    
    print(f"\nCSV files found: {len(csv_files)}")
    for csv in csv_files:
        rel_path = csv.relative_to(data_dir)
        size_mb = csv.stat().st_size / (1024 * 1024)
        print(f"  - {rel_path} ({size_mb:.2f} MB)")
    
    # Check for expected files
    expected_files = [
        "combined_demand_timeseries.csv",
        "merged_timeseries.csv",
        "timeseries.csv"
    ]
    
    print("\nLooking for expected demand timeseries files:")
    found_main_file = None
    
    for expected in expected_files:
        file_path = data_dir / expected
        if file_path.exists():
            print(f"  ✓ Found: {expected}")
            found_main_file = file_path
            break
        else:
            # Check in subdirectories
            matches = list(data_dir.glob(f"**/{expected}"))
            if matches:
                print(f"  ✓ Found: {matches[0].relative_to(data_dir)}")
                found_main_file = matches[0]
                break
    
    if not found_main_file:
        print("  ✗ Main timeseries file not found")
        print("  Available CSV files:")
        for csv in csv_files[:5]:  # Show first 5
            print(f"    - {csv.relative_to(data_dir)}")
        return False
    
    # Verify the main file
    print(f"\n{'=' * 60}")
    print(f"VERIFYING MAIN DATA FILE: {found_main_file.name}")
    print("=" * 60)
    
    try:
        df = pd.read_csv(found_main_file, nrows=5)
        
        print(f"✓ File can be read")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Shape (first 5 rows): {df.shape}")
        
        print(f"\nFirst few rows:")
        print(df.to_string())
        
        # Check for datetime column
        datetime_cols = [col for col in df.columns if any(x in col.lower() for x in ['date', 'time', 'ds', 'timestamp'])]
        if datetime_cols:
            print(f"\n✓ Found datetime column(s): {datetime_cols}")
        else:
            print(f"\n⚠ No obvious datetime column found")
        
        # Check for demand column
        demand_cols = [col for col in df.columns if any(x in col.lower() for x in ['demand', 'load', 'value', 'y'])]
        if demand_cols:
            print(f"✓ Found demand column(s): {demand_cols}")
        else:
            print(f"⚠ No obvious demand column found")
        
        # Count total rows
        print(f"\nCounting total rows (this may take a moment)...")
        total_rows = sum(1 for _ in open(found_main_file)) - 1  # Subtract header
        print(f"✓ Total rows: {total_rows:,}")
        
        # Estimate time range (if we can parse dates)
        if datetime_cols:
            try:
                df_full = pd.read_csv(found_main_file, usecols=[datetime_cols[0]], parse_dates=[datetime_cols[0]])
                date_col = datetime_cols[0]
                print(f"\nDate range:")
                print(f"  Start: {df_full[date_col].min()}")
                print(f"  End: {df_full[date_col].max()}")
                
                # Check frequency
                if len(df_full) > 1:
                    time_diff = (df_full[date_col].iloc[1] - df_full[date_col].iloc[0])
                    print(f"  Frequency (first interval): {time_diff}")
            except Exception as e:
                print(f"  Could not parse dates: {e}")
        
        print(f"\n{'=' * 60}")
        print("✓ DATA VERIFICATION SUCCESSFUL!")
        print("=" * 60)
        print("\nYou can now run the evaluation:")
        print("  python evaluate_gifteval_models.py")
        
        return True
        
    except Exception as e:
        print(f"✗ Error reading file: {e}")
        return False


if __name__ == "__main__":
    success = check_data_upload()
    sys.exit(0 if success else 1)
