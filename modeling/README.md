# Adaptive CEFR English Language & Prosody Modeling AI

This repository contains the Machine Learning modeling pipeline for the Adaptive English Language Learning System (CEFR). The system is built using a Camera-as-UI concept and Generative AI scenarios to adaptively evaluate student English proficiency based on speech prosody and linguistic features.

## Project Structure

```
modeling-AI/
├── AI/
│   ├── parse_oxford_pdf.py   # Extracts vocabulary from Oxford PDFs
│   ├── generate_dataset.py   # Generates simulated prosody and linguistic features
│   ├── clean_data.py         # Performs IQR outlier filtering and data cleaning
│   └── train.py              # Trains CEFR classification and prosody regression models
├── data/
│   └── (oxford_vocabulary.csv, dataset, clean dataset - ignored by git)
├── models/
│   ├── Oxford 3000 CEFR Level.pdf
│   ├── Oxford 5000 by CEFR Level.pdf
│   └── (best_cefr_classifier.pkl, prosody_regressor.pkl - ignored by git)
├── Presentation_Model.ipynb  # Interactive demonstration & EDA notebook
├── requirements.txt          # Python dependencies
├── setup.sh                  # Setup script for automated initialization
└── .gitignore                # Excludes venv, large binary models, and temp files
```

## Features Used

The model uses a combination of prosodic and linguistic/pronunciation features to predict the student's CEFR level (`A1`-`C2`) and prosody similarity (`0`-`100`):

- **Prosody**: `pitch_mean`, `pitch_std`, `pitch_contour_slope`, `energy_rms`, `duration_seconds`, `speech_rate`, `response_time_ms`
- **Linguistic / Fluency**: `lexical_diversity`, `grammar_error_rate`, `pronunciation_accuracy`, `pause_ratio`, `filler_words_rate`, `wpm_consistency`, `asr_confidence`, `semantic_relevance`
- **Representations**: Pretrained speech representation features (`whisper_feat_1`, `whisper_feat_2`, `whisper_feat_3`)

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Bash environment (macOS/Linux)

### Automated Setup

You can set up the environment, install dependencies, parse the vocabulary, generate synthetic speech data, clean the datasets, and train all models with a single command:

```bash
chmod +x setup.sh
./setup.sh
```

### Manual Setup Steps

If you prefer to run steps manually:

1. **Create and Activate Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Step 1: Parse Oxford PDFs**:
   ```bash
   python AI/parse_oxford_pdf.py
   ```

4. **Step 2: Generate Simulated Dataset**:
   ```bash
   python AI/generate_dataset.py
   ```

5. **Step 3: Clean Dataset**:
   ```bash
   python AI/clean_data.py
   ```

6. **Step 4: Train and Save Models**:
   ```bash
   python AI/train.py
   ```

## Model Training & Inference Calibration
- **Anti-Leakage Validation**: The model uses `GroupShuffleSplit` on `speaker_id` to ensure that speaker vocal signatures do not leak from the training set to the testing set.
- **Inference Calibration**: Standard Platt-scaled probabilities for SVM can be overly polarized. In `Presentation_Model.ipynb`, we calibrate the prediction probabilities using Temperature Scaling ($T=2.5$) for a softer, more realistic distribution suitable for visual demonstrations.
