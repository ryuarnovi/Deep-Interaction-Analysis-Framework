# Adaptive CEFR English Language & Prosody Modeling AI

This repository contains the Machine Learning modeling pipeline for the **Adaptive English Language Learning System (CEFR)**. The system is built using a Camera-as-UI concept and Generative AI scenarios to adaptively evaluate student English proficiency based on speech prosody and linguistic features.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Data Pipeline (End-to-End Flow)](#data-pipeline-end-to-end-flow)
- [Features Used](#features-used)
- [Model Architecture & Explanation](#model-architecture--explanation)
  - [Production Pipeline — `train.py`](#1-production-pipeline--trainpy)
  - [Stacking Ensemble — `Stacking_Ensemble_Model.ipynb`](#2-stacking-ensemble--stacking_ensemble_modelipynb)
  - [Presentation & EDA — `Presentation_Model.ipynb`](#3-presentation--eda--presentation_modelipynb)
- [Anti-Leakage Validation & Inference Calibration](#anti-leakage-validation--inference-calibration)
- [Getting Started](#getting-started)

---

## Project Structure

```
Deep-Interaction-Analysis-Framework/
├── AI/
│   ├── parse_oxford_pdf.py          # Step 1: Extracts vocabulary from Oxford PDFs
│   ├── generate_dataset.py          # Step 2: Generates simulated prosody & linguistic features
│   ├── clean_data.py                # Step 3: IQR outlier filtering & data cleaning
│   ├── train.py                     # Step 4: Trains CEFR classification & prosody regression
│   └── inference.py                 # UI Bridge: Predicts CEFR & Prosody and outputs UI JSON
├── data/
│   ├── oxford_vocabulary.csv        # Parsed vocabulary (generated)
│   ├── oxford_prosody_dataset.csv   # Full simulated dataset (generated)
│   └── oxford_prosody_dataset_clean.csv  # Cleaned dataset (generated)
├── models/
│   ├── Oxford 3000 CEFR Level.pdf   # Source PDF for vocabulary
│   ├── Oxford 5000 by CEFR Level.pdf
│   ├── best_cefr_classifier.pkl     # Best classifier model (generated)
│   ├── prosody_regressor.pkl        # Prosody regression model (generated)
│   └── label_encoder.pkl            # Label encoder (generated)
├── Presentation_Model.ipynb         # Interactive demonstration & EDA notebook
├── Stacking_Ensemble_Model.ipynb    # Stacking Ensemble classification notebook
├── requirements.txt                 # Python dependencies
├── setup.sh                         # Automated setup script
└── .gitignore
```

---

## Data Pipeline (End-to-End Flow)

The project follows a **4-step sequential pipeline** to go from raw PDF vocabulary lists to production-ready ML models:

```
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│   Step 1: Parse     │     │   Step 2: Generate   │     │   Step 3: Clean     │     │   Step 4: Train     │
│   Oxford PDFs       │────▶│   Simulated Data     │────▶│   Dataset           │────▶│   & Save Models     │
│                     │     │                      │     │                      │     │                     │
│  parse_oxford_pdf.py│     │ generate_dataset.py  │     │  clean_data.py      │     │  train.py           │
└─────────────────────┘     └──────────────────────┘     └──────────────────────┘     └─────────────────────┘
        │                            │                            │                            │
        ▼                            ▼                            ▼                            ▼
  oxford_vocabulary.csv    oxford_prosody_dataset.csv   oxford_prosody_dataset_clean.csv   *.pkl models
```

### Step 1 — Parse Oxford PDFs (`parse_oxford_pdf.py`)

Reads the **Oxford 3000** and **Oxford 5000** PDF wordlists using `pdfplumber`. It extracts words and their CEFR level labels (A1–C2) by:
- Dividing each PDF page into 3 columns and sorting words spatially.
- Detecting CEFR level headings and associating subsequent vocabulary entries with their POS tags.
- Deduplicating words and writing the result to `data/oxford_vocabulary.csv`.

### Step 2 — Generate Simulated Dataset (`generate_dataset.py`)

Creates a rich, multi-speaker synthetic speech dataset from the vocabulary list. For each word:
- **150 virtual speakers** are simulated, each with a unique speaker accent bias (`speaker_bias`).
- **5 scenarios** (`interview`, `emergency`, `casual_chat`, `presentation`, `academic_discussion`) add contextual diversity.
- **20 numeric features** are generated per sample with CEFR-level-dependent statistical distributions (e.g., higher proficiency → higher `pronunciation_accuracy`, lower `grammar_error_rate`).
- **C2 vocabulary is synthesized** to ensure all 6 CEFR levels are represented.
- Output: `data/oxford_prosody_dataset.csv` (~10,800 samples).

### Step 3 — Clean Dataset (`clean_data.py`)

Applies a multi-stage data quality pipeline:
1. **Deduplication** — Removes duplicate (word, speaker_id) pairs.
2. **Missing values** — Drops rows with missing targets; imputes numeric features with the median.
3. **Logical range validation** — Ensures physically valid values (e.g., `duration_seconds ≥ 0.1`, `0 ≤ prosody_similarity ≤ 100`).
4. **IQR outlier removal** — Removes extreme outliers on `pitch_std`, `duration_seconds`, and `response_time_ms` using a 3× IQR threshold.
5. **Text normalization** — Lowercases words and keeps only alphabetic/hyphenated tokens.

### Step 4 — Train & Save Models (`train.py`)

Trains both **classification** and **regression** models. Details in the [Model Architecture](#model-architecture--explanation) section below.

### Step 5 — Inference & UI Bridge (`inference.py`)

Serves as the runtime bridge between the raw model predictions and the frontend UI:
- Loads the trained classification, regression, and label encoding models.
- Accepts raw student speaking features (20 parameters).
- Performs **Temperature Scaling calibration** on predictions to make probabilities suited for UI visualization.
- Evaluates individual features to compile user-friendly lists of strengths and targeted improvements.
- Outputs a clean, structured JSON format that the UI can directly render for student feedback.

---

## Features Used

The model uses **20 numeric features** across three categories to predict the student's CEFR level (A1–C2) and prosody similarity (0–100):

| Category | Features | Description |
|---|---|---|
| **Prosody** | `pitch_mean`, `pitch_std`, `pitch_contour_slope`, `energy_rms`, `duration_seconds`, `speech_rate`, `response_time_ms` | Acoustic and temporal characteristics of speech |
| **Linguistic / Fluency** | `lexical_diversity`, `grammar_error_rate`, `pronunciation_accuracy`, `pause_ratio`, `filler_words_rate`, `wpm_consistency`, `asr_confidence`, `semantic_relevance` | Measures of language competence and fluency |
| **Deep Representations** | `whisper_feat_1`, `whisper_feat_2`, `whisper_feat_3` | Simulated pretrained Whisper/Wav2Vec2 embeddings |
| **User Context** | `user_prior_score`, `prosody_similarity` | Historical performance and target regression variable |

> **Note**: The identifier columns `word`, `speaker_id`, and `scenario_id` are **excluded** from training features to prevent shortcut learning.

---

## Model Architecture & Explanation

This project contains **three modeling approaches** at different levels of complexity:

### 1. Production Pipeline — `train.py`

This is the main training script that outputs deployable `.pkl` models.

#### Classification (CEFR Level Prediction)

Four classifiers are trained and compared; the best one (by macro F1-score) is automatically saved:

| Model | Key Hyperparameters |
|---|---|
| **XGBoost** | `n_estimators=100`, `learning_rate=0.1`, `max_depth=5` |
| **Random Forest** | `n_estimators=150`, `max_depth=15`, `class_weight='balanced'` |
| **SVM (RBF)** | `C=3.0`, `gamma='scale'`, `probability=True` |
| **Deep MLP** | `hidden_layers=(128, 64)`, `early_stopping=True` |

Each model is wrapped in an `ImbPipeline` with:
1. **`ColumnTransformer`** → `StandardScaler` on all numeric features (fitted only on training data to prevent data leakage).
2. **Classifier** → One of the four models above.

The best model is saved as `models/best_cefr_classifier.pkl`.

#### Regression (Prosody Similarity Prediction)

A **`RandomForestRegressor`** (`n_estimators=100`) predicts the continuous `prosody_similarity` score (0–100). It is saved as `models/prosody_regressor.pkl`.

#### Data Splitting Strategy

Uses **`GroupShuffleSplit`** on `speaker_id` (85/15 split) to ensure that all samples from the same speaker appear entirely in the train **or** test set — never both. This prevents the model from memorizing speaker-specific vocal signatures (anti-leakage).

---

### 2. Stacking Ensemble — `Stacking_Ensemble_Model.ipynb`

This notebook implements an **advanced Stacking Ensemble** approach with hyperparameter tuning. The workflow is:

```
┌─────────────────────────────────────────────────────────┐
│                   DATA PREPARATION                      │
│  Load → Drop identifiers → LabelEncode → Stratified    │
│  train_test_split (80/20) → StratifiedKFold (5-fold)    │
└─────────────────────────┬───────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
   ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
   │  SVM (RBF)  │ │ Random      │ │  XGBoost    │
   │  Pipeline   │ │ Forest      │ │  Pipeline   │
   │             │ │ Pipeline    │ │             │
   │ + SelectK   │ │             │ │             │
   │ + SMOTE     │ │ + SMOTE     │ │ + SMOTE     │
   │ + Scaler    │ │ + Scaler    │ │ + Scaler    │
   └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
          │               │               │
          │  RandomizedSearchCV (10 iter)  │
          │  scoring='f1_weighted'         │
          │               │               │
          ▼               ▼               ▼
   ┌─────────────────────────────────────────┐
   │         STACKING CLASSIFIER             │
   │                                         │
   │  Base estimators: best SVM, RF, XGB     │
   │  Meta-learner: LogisticRegression       │
   └─────────────────────┬───────────────────┘
                         ▼
              Final CEFR Prediction
```

#### Key differences from `train.py`:

| Aspect | `train.py` (Production) | `Stacking_Ensemble_Model.ipynb` |
|---|---|---|
| **Split method** | `GroupShuffleSplit` by speaker | `train_test_split` with stratification |
| **Tuning** | Fixed hyperparameters | `RandomizedSearchCV` (10 iterations × 5 folds) |
| **Class imbalance** | None | **SMOTE** oversampling in each pipeline |
| **Feature selection** | All features | SVM uses **`SelectKBest`** (ANOVA F-test) |
| **Ensemble** | Single best model | **Stacking** (SVM + RF + XGB → LogisticRegression) |
| **Scoring metric** | Macro F1 | Weighted F1 |

#### Individual Model Pipelines:

- **SVM**: `Preprocessor → SelectKBest → SMOTE → SVC(rbf)` — tuned over `C ∈ {0.1, 1, 3, 10}`, `gamma ∈ {scale, 0.01, 0.001}`, `k ∈ {5, 10, 15, all}`.
- **Random Forest**: `Preprocessor → SMOTE → RandomForestClassifier` — tuned over `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`.
- **XGBoost**: `Preprocessor → SMOTE → XGBClassifier` — tuned over `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`.

After individual tuning, the three best estimators are combined into a **`StackingClassifier`** with `LogisticRegression` as the meta-learner. The stacking model learns to optimally weight each base model's predictions.

---

### 3. Presentation & EDA — `Presentation_Model.ipynb`

This notebook provides:
- **Exploratory Data Analysis** (EDA): CEFR class distribution, feature correlations, prosody-vs-CEFR visualizations.
- **Model comparison**: Same 4 classifiers as `train.py` + prosody regression, with detailed evaluation metrics and confusion matrices.
- **Inference simulation**: A visual demonstration of camera-based feedback using combined CEFR classification (Top-2 predictions) and prosody regression scores, with **Temperature Scaling** calibration.

---

## Anti-Leakage Validation & Inference Calibration

### Anti-Leakage Validation

The production model (`train.py`) uses **`GroupShuffleSplit`** on `speaker_id` to guarantee that speaker vocal signatures do not leak from the training set to the testing set. This means:
- Every sample from a given speaker is in **either** the train set or the test set, never both.
- The model must generalize to **unseen speakers**, not memorize individual vocal patterns.

The Stacking Ensemble notebook uses `StratifiedKFold` for cross-validation to ensure proportional CEFR class representation in every fold, combined with SMOTE to address class imbalance within each fold.

### Inference Calibration

Standard Platt-scaled probabilities for SVM can be overly polarized (very high confidence even on ambiguous samples). In `Presentation_Model.ipynb`, we calibrate the prediction probabilities using **Temperature Scaling** ($T = 2.5$):

$$p_{\text{calibrated}} = \text{softmax}\left(\frac{\text{logits}}{T}\right)$$

This produces a softer, more realistic probability distribution suitable for visual demonstrations and UI feedback.

---

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

### Running the Notebooks

After the data pipeline is complete, open the notebooks:

```bash
jupyter notebook Presentation_Model.ipynb
jupyter notebook Stacking_Ensemble_Model.ipynb
```

Or upload them to [Kaggle](https://www.kaggle.com/) — adjust the dataset path in cell #2 to point to your Kaggle dataset location.
