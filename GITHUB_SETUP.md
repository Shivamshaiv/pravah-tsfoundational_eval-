# GitHub Setup Instructions

## Push to GitHub from VM

After creating your GitHub repository, run these commands:

```bash
cd /home/patel/tsfoundational_eval

# Add your GitHub repository as remote (replace with your URL)
git remote add origin https://github.com/YOUR_USERNAME/tsfoundational_eval.git

# Rename branch to main
git branch -M main

# Push to GitHub
git push -u origin main
```

## Alternative: Use SSH (if you have SSH keys set up)

```bash
git remote add origin git@github.com:YOUR_USERNAME/tsfoundational_eval.git
git branch -M main
git push -u origin main
```

## Clone to Local Machine

Once pushed to GitHub, on your local machine:

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/tsfoundational_eval.git

# Navigate to the directory
cd tsfoundational_eval

# Set up virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Alternative Download Methods

### Option 1: Download as ZIP from GitHub
After pushing, you can download directly from GitHub:
1. Go to your repository page
2. Click the green "Code" button
3. Click "Download ZIP"
4. Extract on your local machine

### Option 2: Create tarball on VM
```bash
cd /home/patel
tar -czf tsfoundational_eval.tar.gz tsfoundational_eval/ \
  --exclude='tsfoundational_eval/.venv' \
  --exclude='tsfoundational_eval/__pycache__' \
  --exclude='tsfoundational_eval/.git'
```

Then download `tsfoundational_eval.tar.gz` using SCP or your cloud console.

### Option 3: SCP Direct Transfer
If you have SSH access to the VM:

```bash
# On your local machine
scp -r username@vm-ip:/home/patel/tsfoundational_eval ./
```

## What's Included

Your repository includes:
- All Python code with GPU support
- Model implementations
- Data folder with CSV files
- Requirements.txt
- Documentation
- GPU utilities

**Not included (per .gitignore):**
- .venv/ (virtual environment)
- __pycache__/ (Python cache)
- *.zip files
- Model checkpoints (*.ckpt)
- Log files
