#!/bin/bash
# Script to extract and organize electricity demand data

set -e

echo "=================================================="
echo "Electricity Demand Data Extraction Script"
echo "=================================================="

# Check if zip file exists
if [ ! -f "data_upload.zip" ]; then
    echo "Error: data_upload.zip not found in current directory"
    echo ""
    echo "Please upload your data file using one of these methods:"
    echo "1. SCP: scp 'C:\Users\patel\Downloads\drive-download-20251005T044203Z-1-002.zip' patel@<vm-ip>:/home/patel/tsfoundational_eval/data_upload.zip"
    echo "2. SFTP: Use an SFTP client like FileZilla or WinSCP"
    echo "3. VS Code: Drag and drop the file in the file explorer"
    echo ""
    exit 1
fi

# Create data directory if it doesn't exist
mkdir -p data
echo "✓ Data directory created/verified"

# Extract the zip file
echo "Extracting data..."
unzip -o data_upload.zip -d data/

# List extracted files
echo ""
echo "Extracted files:"
ls -lh data/

# Look for CSV files
echo ""
echo "CSV files found:"
find data/ -name "*.csv" -type f

# Look for the main demand timeseries file
echo ""
echo "Looking for main demand timeseries file..."
if [ -f "data/combined_demand_timeseries.csv" ]; then
    echo "✓ Found: combined_demand_timeseries.csv"
    wc -l data/combined_demand_timeseries.csv
elif [ -f "data/merged_timeseries.csv" ]; then
    echo "✓ Found: merged_timeseries.csv"
    wc -l data/merged_timeseries.csv
else
    echo "Main timeseries file not found with expected name."
    echo "Available CSV files:"
    find data/ -name "*.csv" -type f
fi

echo ""
echo "=================================================="
echo "Data extraction complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. Verify the data files are correct"
echo "2. Run: python evaluate_gifteval_models.py"
