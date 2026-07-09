#!/bin/bash

# Setup script for Adaptive CEFR English Language & Prosody Modeling AI
# Author: Antigravity AI

set -e # Stop script on error

echo "=========================================================="
echo " Starting Automated ML Pipeline Setup"
echo "=========================================================="

# 1. Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment already exists."
fi

# 2. Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# 3. Upgrade pip and install requirements
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 4. Run data prep and modeling pipeline
echo "Step 1/4: Parsing Oxford PDF vocabulary..."
python AI/parse_oxford_pdf.py

echo "Step 2/4: Generating simulated speech/prosody dataset..."
python AI/generate_dataset.py

echo "Step 3/4: Cleaning generated dataset (IQR Outlier Filtering)..."
python AI/clean_data.py

echo "Step 4/4: Training CEFR Classifier & Prosody Regressor..."
python AI/train.py

echo "=========================================================="
echo " Setup and Model Training Complete!"
echo " All models saved in the 'models/' directory."
echo " You can now run the 'Presentation_Model.ipynb' notebook."
echo "=========================================================="
