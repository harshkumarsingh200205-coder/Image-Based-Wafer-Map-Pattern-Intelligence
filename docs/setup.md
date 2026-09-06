# Environment Setup & Installation Guide

## 1. Prerequisites
- Python 3.10 or higher.
- Git.
- Optional: CUDA-compatible GPU for faster CNN training and Grad-CAM generation (CPU fallback fully supported).

## 2. Setting Up Virtual Environment

### On Windows (PowerShell / Command Prompt):
```powershell
# Navigate to project directory
cd "C:\Projects\Image-Based Wafer Map Pattern Intelligence"

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
.venv\Scripts\Activate.ps1
```

### On Linux / macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Installing Dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Verifying Installation
Run the test suite to ensure all libraries and core modules are operational:
```bash
pytest tests/ -v
```

## 5. Configuration
Configuration settings are managed in [`configs/config.yaml`](file:///c:/Projects/Image-Based%20Wafer%20Map%20Pattern%20Intelligence/configs/config.yaml). Adjust batch sizes, image resolutions, or confidence thresholds as required for your hardware.
