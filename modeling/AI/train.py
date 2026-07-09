import pandas as pd
import numpy as np
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, precision_score, recall_score, mean_squared_error, classification_report
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

def load_and_preprocess_data(filepath='data/oxford_prosody_dataset.csv'):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}. Please run generate_dataset.py first.")
        
    df = pd.read_csv(filepath)
    
    # Fitur latih: Keluarkan identifiers dan targets
    X = df.drop(columns=['word', 'cefr_level', 'speaker_id', 'scenario_id', 'prosody_similarity'])
    y_class = df['cefr_level']
    y_reg = df['prosody_similarity']
    
    # Encode label CEFR
    le = LabelEncoder()
    le.fit(['A1', 'A2', 'B1', 'B2', 'C1', 'C2'])
    y_class_encoded = le.transform(y_class)
    
    # Pembicara untuk validasi independen pembicara
    speakers = df['speaker_id']
    
    return X, y_class_encoded, y_reg, speakers, le

def train_classification_models(X_train, y_train, X_test, y_test, preprocessor, le):
    print("--- Training Classification Models (CEFR Level) ---")
    
    models = {
        'XGBoost': XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, eval_metric='mlogloss', n_jobs=-1),
        'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=15, min_samples_split=5, random_state=42, class_weight='balanced', n_jobs=-1),
        'SVM': SVC(kernel='rbf', C=3.0, gamma='scale', random_state=42, probability=True),
        'Deep MLP': MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200, random_state=42, early_stopping=True)
    }
    
    best_model = None
    best_f1 = 0
    best_model_name = ""
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        pipeline = ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        f1 = f1_score(y_test, y_pred, average='macro')
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        
        print(f"{name} Results:")
        print(f"  Macro F1: {f1:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_model = pipeline
            best_model_name = name
            
    print(f"\nBest Classification Model: {best_model_name} with F1: {best_f1:.4f}")
    
    # Menyimpan model klasifikasi terbaik dan label encoder
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/best_cefr_classifier.pkl')
    joblib.dump(le, 'models/label_encoder.pkl')
    print("Saved best classifier to 'models/best_cefr_classifier.pkl'")
    
    # Cetak laporan klasifikasi untuk model terbaik
    y_pred_best = best_model.predict(X_test)
    print("\nLaporan Klasifikasi Model Terbaik:")
    print(classification_report(y_test, y_pred_best, target_names=le.classes_))
    
    return best_model

def train_regression_model(X_train, y_train_reg, X_test, y_test_reg, preprocessor):
    print("\n--- Training Regression Model (Prosody Similarity) ---")
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    
    print("Training Random Forest Regressor...")
    pipeline.fit(X_train, y_train_reg)
    y_pred = pipeline.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred))
    print(f"Random Forest Regressor RMSE: {rmse:.4f}")
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(pipeline, 'models/prosody_regressor.pkl')
    print("Saved prosody regressor to 'models/prosody_regressor.pkl'")

def main():
    try:
        X, y_class, y_reg, speakers, le = load_and_preprocess_data()
    except FileNotFoundError as e:
        print(e)
        return
        
    # 2. Speaker-Independent Split (Anti-Leakage)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_idx, test_idx = next(gss.split(X, y_class, groups=speakers))
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train_class, y_test_class = y_class[train_idx], y_class[test_idx]
    y_train_reg, y_test_reg = y_reg.iloc[train_idx], y_reg.iloc[test_idx]
    
    print(f"Jumlah sampel Train : {len(X_train)} (Speaker unik: {speakers.iloc[train_idx].nunique()})")
    print(f"Jumlah sampel Test  : {len(X_test)} (Speaker unik: {speakers.iloc[test_idx].nunique()})")
    
    # 3. Pra-pemrosesan (Standardisasi dimasukkan ke dalam Pipeline untuk menghindari data leakage)
    numeric_features = X.columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features)
        ])
    
    # Train Classification
    train_classification_models(X_train, y_train_class, X_test, y_test_class, preprocessor, le)
    
    # Train Regression
    train_regression_model(X_train, y_train_reg, X_test, y_test_reg, preprocessor)
    
    print("\nModel training complete. All models saved in 'models/' directory.")

if __name__ == "__main__":
    main()
