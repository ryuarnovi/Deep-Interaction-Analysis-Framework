#!/usr/bin/env python3
"""
Kaggle Training Script - Oxford English CEFR & Prosody Assessment
================================================================
Copy and paste this code into a Kaggle Notebook cell.
This script trains:
1. CEFR Level Classifier (A1-C2) - XGBoost, Random Forest, SVM, MLP
2. Prosody Similarity Regressor (1-100) - Random Forest
It also generates evaluation metrics and charts for presentation.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    f1_score, precision_score, recall_score, 
    mean_squared_error, classification_report, confusion_matrix
)

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

# 1. Deteksi Path File Dataset (Mendukung lokal & Kaggle)
def find_dataset():
    kaggle_input_dir = '/kaggle/input'
    local_path = 'data/oxford_prosody_dataset_clean.csv'
    
    # Cek jika di Kaggle
    if os.path.exists(kaggle_input_dir):
        for root, dirs, files in os.walk(kaggle_input_dir):
            for file in files:
                if file == 'oxford_prosody_dataset_clean.csv':
                    path = os.path.join(root, file)
                    print(f"Dataset ditemukan di Kaggle: {path}")
                    return path
                
    # Fallback ke path lokal
    if os.path.exists(local_path):
        print(f"Dataset ditemukan di Lokal: {local_path}")
        return local_path
        
    raise FileNotFoundError(
        "Dataset 'oxford_prosody_dataset_clean.csv' tidak ditemukan. "
        "Jika di Kaggle, pastikan Anda sudah menambahkan dataset ke notebook Anda."
    )

# 2. Load & Preprocess Data
def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)
    print(f"Berhasil memuat dataset. Total baris: {len(df)}, Kolom: {len(df.columns)}")
    
    # Fitur latih: Keluarkan identifiers dan target
    drop_cols = ['word', 'cefr_level', 'speaker_id', 'scenario_id', 'prosody_similarity']
    # Filter kolom yang ada di dataset untuk menghindari error
    drop_cols = [col for col in drop_cols if col in df.columns]
    
    X = df.drop(columns=drop_cols)
    y_class = df['cefr_level']
    y_reg = df['prosody_similarity'] if 'prosody_similarity' in df.columns else None
    
    # Encode level CEFR
    le = LabelEncoder()
    le.fit(['A1', 'A2', 'B1', 'B2', 'C1', 'C2'])
    y_class_encoded = le.transform(y_class)
    
    # Pembicara untuk validasi independen pembicara (mencegah data leakage)
    speakers = df['speaker_id'] if 'speaker_id' in df.columns else pd.Series(range(len(df)))
    
    return X, y_class_encoded, y_reg, speakers, le

# 3. Training & Evaluasi Model Klasifikasi (CEFR)
def train_classifiers(X_train, y_train, X_test, y_test, preprocessor, le):
    print("\n" + "="*50)
    # Gunakan print biasa tanpa karakter UTF-8 aneh agar aman di log Kaggle
    print("=== TRAINING KLASIFIKASI TINGKAT CEFR ===")
    print("="*50)
    
    models = {
        'XGBoost': XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, eval_metric='mlogloss', n_jobs=-1),
        'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=15, min_samples_split=5, random_state=42, class_weight='balanced', n_jobs=-1),
        'SVM': SVC(kernel='rbf', C=3.0, gamma='scale', random_state=42, probability=True),
        'Deep MLP': MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200, random_state=42, early_stopping=True)
    }
    
    best_model = None
    best_f1 = 0
    best_model_name = ""
    results = {}
    
    for name, model in models.items():
        print(f"\nMelatih {name}...")
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        f1 = f1_score(y_test, y_pred, average='macro')
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        
        print(f"Hasil {name}:")
        print(f"  Macro F1 : {f1:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall   : {recall:.4f}")
        
        results[name] = f1
        
        if f1 > best_f1:
            best_f1 = f1
            best_model = pipeline
            best_model_name = name
            
    print(f"\nModel Klasifikasi Terbaik: {best_model_name} dengan F1: {best_f1:.4f}")
    
    # Simpan model terbaik ke output directory
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/best_cefr_classifier.pkl')
    joblib.dump(le, 'models/label_encoder.pkl')
    print("Model disimpan ke 'models/best_cefr_classifier.pkl'")
    
    # Cetak laporan klasifikasi lengkap
    y_pred_best = best_model.predict(X_test)
    print("\nLaporan Klasifikasi Lengkap (Model Terbaik):")
    print(classification_report(y_test, y_pred_best, target_names=le.classes_))
    
    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred_best)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.ylabel('Tingkat Asli (True)')
    plt.xlabel('Prediksi AI (Predicted)')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Confusion Matrix disimpan sebagai 'confusion_matrix.png'")
    plt.show()
    
    return best_model, best_model_name, results

# 4. Training & Evaluasi Model Regresi (Skor Pronunciation)
def train_regressor(X_train, y_train_reg, X_test, y_test_reg, preprocessor):
    print("\n" + "="*50)
    print("=== TRAINING REGRESI SKOR PENGUCAPAN ===")
    print("="*50)
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    
    print("Melatih Random Forest Regressor...")
    pipeline.fit(X_train, y_train_reg)
    y_pred = pipeline.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred))
    print(f"Random Forest Regressor RMSE: {rmse:.4f}")
    
    # Simpan model regresi
    os.makedirs('models', exist_ok=True)
    joblib.dump(pipeline, 'models/prosody_regressor.pkl')
    print("Model Regresi disimpan ke 'models/prosody_regressor.pkl'")
    
    # Plot Feature Importance (Menunjukkan fitur mana yang paling memengaruhi skor bicara)
    reg_model = pipeline.named_steps['regressor']
    importances = reg_model.feature_importances_
    features = X_train.columns
    
    df_imp = pd.DataFrame({'Fitur': features, 'Kepentingan': importances})
    df_imp = df_imp.sort_values(by='Kepentingan', ascending=False).head(10)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Kepentingan', y='Fitur', data=df_imp, palette='viridis')
    plt.title('Top 10 Fitur Paling Berpengaruh dalam Penilaian Bicara')
    plt.xlabel('Skor Kepentingan Fitur')
    plt.ylabel('Fitur Akustik / Linguistik')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("Chart Feature Importance disimpan sebagai 'feature_importance.png'")
    plt.show()

def main():
    # 1. Cari file dataset
    try:
        filepath = find_dataset()
    except FileNotFoundError as e:
        print(e)
        return
        
    # 2. Muat dan preprocess
    X, y_class, y_reg, speakers, le = load_and_preprocess(filepath)
    
    # 3. Speaker-Independent Split (Validasi Mandiri Pembicara untuk mencegah data leakage)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_idx, test_idx = next(gss.split(X, y_class, groups=speakers))
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train_class, y_test_class = y_class[train_idx], y_class[test_idx]
    y_train_reg, y_test_reg = y_reg.iloc[train_idx], y_reg.iloc[test_idx]
    
    print(f"\nJumlah sampel Latih (Train): {len(X_train)} (Speaker unik: {speakers.iloc[train_idx].nunique()})")
    print(f"Jumlah sampel Uji (Test)  : {len(X_test)} (Speaker unik: {speakers.iloc[test_idx].nunique()})")
    
    # 4. Standardisasi
    numeric_features = X.columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features)
        ])
    
    # 5. Jalankan Klasifikasi
    best_clf, best_name, class_results = train_classifiers(X_train, y_train_class, X_test, y_test_class, preprocessor, le)
    
    # 6. Jalankan Regresi jika data prosody similarity tersedia
    if y_train_reg is not None and not y_train_reg.isna().all():
        train_regressor(X_train, y_train_reg, X_test, y_test_reg, preprocessor)
        
    print("\n" + "="*50)
    print("PROSES TRAINING SELESAI!")
    print("="*50)
    print("Semua model tersimpan di direktori 'models/'.")
    print("Visualisasi ('confusion_matrix.png' & 'feature_importance.png') siap diunduh untuk presentasi.")

if __name__ == "__main__":
    main()
