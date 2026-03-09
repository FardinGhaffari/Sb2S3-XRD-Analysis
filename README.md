# Sb2S3 XRD Texture Analysis

To install the required packages, run the following command: "pip install -r requirements.txt" in your terminal

This repository contains Python scripts for processing XRD data, specifically focused on Sb2S3 thin films. It includes automated baseline subtraction, FTO-based peak alignment, and Texture Coefficient (TC) calculations.

## 📁 Project Structure

```text
SB2S3/
├── data/
│   ├── processed_data/         # Cleaned/Baseline-subtracted files
│   └── raw_data/               # Original XRD files (Batch 1, 2, 3,...)
├── results/
│   ├── csv_files/              # Calculated TC results
│   └── plots/                  # Exported PNGs/PDFs
├── scripts/
│   ├── functions.py            # Core Pseudo-Voigt & TC logic
│   └── *.ipynb                 # Analysis notebooks (run these)
├── .gitignore                  # Ignores __pycache__ and local junk
└── requirements.txt            # Python dependencies
