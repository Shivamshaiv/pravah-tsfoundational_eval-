# Data Upload Instructions

## Uploading Data from Windows to Linux VM

You have the data file at: `C:\Users\patel\Downloads\drive-download-20251005T044203Z-1-002.zip`

Here are **3 methods** to upload it to the VM:

---

## Method 1: Using VS Code (Easiest)

Since you're already using Cursor/VS Code:

1. Open the file explorer in VS Code (left sidebar)
2. Navigate to `/home/patel/tsfoundational_eval/`
3. **Drag and drop** the zip file from Windows Explorer into the VS Code file explorer
4. Rename it to `data_upload.zip` (right-click → Rename)
5. Run the extraction script:
   ```bash
   cd /home/patel/tsfoundational_eval
   ./upload_and_extract_data.sh
   ```

---

## Method 2: Using SCP from Windows PowerShell/CMD

1. Open PowerShell or Command Prompt on Windows
2. Run this command (replace `<vm-ip>` with your VM's IP address):
   ```powershell
   scp "C:\Users\patel\Downloads\drive-download-20251005T044203Z-1-002.zip" patel@<vm-ip>:/home/patel/tsfoundational_eval/data_upload.zip
   ```
3. Enter your password when prompted
4. SSH into the VM and run:
   ```bash
   cd /home/patel/tsfoundational_eval
   ./upload_and_extract_data.sh
   ```

---

## Method 3: Using WinSCP (GUI Tool)

1. Download WinSCP: https://winscp.net/eng/download.php
2. Connect to your VM:
   - Host name: `<vm-ip>`
   - User name: `patel`
   - Password: `<your-password>`
3. Navigate to `/home/patel/tsfoundational_eval/`
4. Drag and drop the zip file from left (Windows) to right (Linux)
5. Rename to `data_upload.zip`
6. In your terminal, run:
   ```bash
   cd /home/patel/tsfoundational_eval
   ./upload_and_extract_data.sh
   ```

---

## Method 4: Using Python HTTP Server (if VM has internet access)

If you can't use the above methods, you can set up a temporary HTTP server:

### On Windows (in the Downloads folder):
```powershell
cd C:\Users\patel\Downloads
python -m http.server 8000
```

### On Linux VM:
```bash
cd /home/patel/tsfoundational_eval
# Replace <your-windows-ip> with your Windows machine's IP
wget http://<your-windows-ip>:8000/drive-download-20251005T044203Z-1-002.zip -O data_upload.zip
./upload_and_extract_data.sh
```

---

## After Upload: Verify the Data

Once uploaded and extracted, verify the data:

```bash
cd /home/patel/tsfoundational_eval
ls -lh data/
head -20 data/*.csv
```

Expected files:
- `combined_demand_timeseries.csv` or `merged_timeseries.csv`
- Possibly weather data files
- Other supporting data files

---

## Troubleshooting

### If the extraction script fails:

Manual extraction:
```bash
cd /home/patel/tsfoundational_eval
mkdir -p data
unzip -o data_upload.zip -d data/
ls -lh data/
```

### If you get "command not found: unzip":

Install unzip:
```bash
sudo apt-get update
sudo apt-get install unzip
```

### Check what's in the zip file without extracting:

```bash
unzip -l data_upload.zip
```

---

## Next Steps

After successful extraction:

1. **Verify the data format:**
   ```bash
   cd /home/patel/tsfoundational_eval
   source .venv/bin/activate
   python -c "import pandas as pd; df = pd.read_csv('data/combined_demand_timeseries.csv'); print(df.head()); print(df.info())"
   ```

2. **Update the evaluation script** to use the actual data file name

3. **Run the evaluation:**
   ```bash
   python evaluate_gifteval_models.py
   ```

---

## Need Help?

If you encounter issues, let me know:
- What method you're using
- Any error messages you see
- Output of `ls -lh data/` after extraction
