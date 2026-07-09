import os
import joblib
import numpy as np
import pandas as pd

class CEFRProsodyPredictor:
    def __init__(self, models_dir='models'):
        """
        Inisialisasi predictor dengan memuat model klasifikasi CEFR,
        model regresi prosodi, dan label encoder.
        """
        self.classifier_path = os.path.join(models_dir, 'best_cefr_classifier.pkl')
        self.regressor_path = os.path.join(models_dir, 'prosody_regressor.pkl')
        self.label_encoder_path = os.path.join(models_dir, 'label_encoder.pkl')
        
        # Cek ketersediaan model
        if not all(os.path.exists(p) for p in [self.classifier_path, self.regressor_path, self.label_encoder_path]):
            raise FileNotFoundError(
                "Model pkl tidak ditemukan di folder 'models/'. "
                "Pastikan Anda telah menjalankan pipeline training (setup.sh atau python AI/train.py)."
            )
            
        print("Memuat model AI...")
        self.classifier = joblib.load(self.classifier_path)
        self.regressor = joblib.load(self.regressor_path)
        self.label_encoder = joblib.load(self.label_encoder_path)
        
        # Daftar fitur wajib sesuai urutan saat model dilatih
        self.feature_columns = [
            'pitch_mean', 'pitch_std', 'pitch_contour_slope', 'energy_rms', 
            'duration_seconds', 'speech_rate', 'response_time_ms', 
            'lexical_diversity', 'grammar_error_rate', 'pronunciation_accuracy', 
            'pause_ratio', 'filler_words_rate', 'wpm_consistency', 
            'asr_confidence', 'semantic_relevance', 
            'whisper_feat_1', 'whisper_feat_2', 'whisper_feat_3', 
            'user_prior_score'
        ]

    def _softmax(self, x):
        """Fungsi pembantu Softmax numerik yang stabil."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def _apply_temperature_scaling(self, probabilities, temperature=2.5):
        """
        Melakukan kalibrasi probabilitas menggunakan Temperature Scaling.
        Menghindari probabilitas yang terlalu percaya diri/polarisasi ekstrem (Platt-scaled SVM bias)
        sehingga menghasilkan visualisasi probabilitas UI yang lebih halus.
        """
        # Konversi probabilitas kembali ke log-odds (logits) secara aproksimasi
        # dengan eps untuk menghindari log(0)
        eps = 1e-15
        probabilities = np.clip(probabilities, eps, 1 - eps)
        logits = np.log(probabilities / (1 - probabilities))
        
        # Skalakan logits dengan temperature
        scaled_logits = logits / temperature
        
        # Softmax kembali untuk mendapatkan probabilitas terkalibrasi
        return self._softmax(scaled_logits)

    def predict(self, feature_dict, temperature=2.5):
        """
        Menerima data input fitur pengguna, melakukan prediksi CEFR dan skor Prosodi,
        serta memformulasikan simpulan feedback untuk UI.
        """
        # 1. Konversi input dict ke DataFrame dan pastikan kesesuaian kolom
        missing_cols = [col for col in self.feature_columns if col not in feature_dict]
        if missing_cols:
            raise ValueError(f"Fitur input tidak lengkap. Kurang kolom: {missing_cols}")
            
        # Susun fitur sesuai urutan kolom asli saat training
        input_data = pd.DataFrame([{col: feature_dict[col] for col in self.feature_columns}])
        
        # 2. Prediksi Kelas CEFR (Probabilitas)
        # Mendapatkan probabilitas dasar dari model classifier terbaik (misal SVM / XGBoost)
        raw_probs = self.classifier.predict_proba(input_data)[0]
        
        # Kalibrasi probabilitas dengan Temperature Scaling untuk UI
        calibrated_probs = self._apply_temperature_scaling(raw_probs, temperature)
        
        # Hubungkan probabilitas terkalibrasi dengan kelas CEFR masing-masing (A1-C2)
        classes = self.label_encoder.classes_
        cefr_prob_map = {classes[i]: float(calibrated_probs[i]) for i in range(len(classes))}
        
        # Urutkan berdasarkan probabilitas tertinggi (untuk deteksi Top-1 dan Top-2)
        sorted_probs = sorted(cefr_prob_map.items(), key=lambda x: x[1], reverse=True)
        top_1_cefr, top_1_conf = sorted_probs[0]
        top_2_cefr, top_2_conf = sorted_probs[1]
        
        # 3. Prediksi Skor Prosodi (0-100) menggunakan regressor
        predicted_prosody_score = float(self.regressor.predict(input_data)[0])
        
        # 4. Formulasi Analisis Fitur & Simpulan Feedback Bahasa Indonesia untuk UI
        feedback_notes = []
        strengths = []
        improvements = []
        
        # Analisis detail parameter akustik & linguistik
        # A. Pengucapan & Kelancaran
        if feature_dict['pronunciation_accuracy'] >= 80:
            strengths.append("Akurasi pengucapan (pronunciation) sangat baik dan terdengar natural.")
        elif feature_dict['pronunciation_accuracy'] < 60:
            improvements.append("Perlu meningkatkan kejelasan pengucapan kata-kata tertentu (artikulasi).")
            
        if feature_dict['pause_ratio'] > 0.35:
            improvements.append("Terlalu banyak jeda diam saat berbicara. Cobalah melatih kelancaran aliran kalimat.")
        else:
            strengths.append("Aliran berbicara lancar dengan rasio jeda diam yang minimal.")
            
        if feature_dict['filler_words_rate'] > 0.15:
            improvements.append("Sering menggunakan kata pengisi seperti 'uh' atau 'um'. Cobalah kurangi jeda canggung ini.")
            
        # B. Intonasi & Ritme (Prosodi)
        if predicted_prosody_score >= 75:
            strengths.append("Ritme dan intonasi suara Anda sudah sangat mirip dengan penutur asli (native speaker).")
        elif predicted_prosody_score < 50:
            improvements.append("Intonasi terdengar agak datar. Cobalah gunakan variasi nada (pitch std) saat mengekspresikan kalimat.")
            
        # C. Relevansi Pemahaman & Kecepatan Respons
        if feature_dict['semantic_relevance'] >= 75:
            strengths.append("Jawaban Anda sangat relevan dengan konteks skenario yang diberikan.")
        elif feature_dict['semantic_relevance'] < 50:
            improvements.append("Penyampaian pesan kurang fokus pada konteks skenario. Sesuaikan jawaban dengan instruksi.")

        if feature_dict['response_time_ms'] > 2000:
            improvements.append("Waktu berpikir sebelum menjawab agak lama. Latih spontanitas agar lebih percaya diri.")

        # Buat teks simpulan komprehensif untuk ditampilkan di UI
        if len(improvements) == 0:
            kesimpulan_saran = "Luar biasa! Kemampuan berbicara Anda sudah sangat matang secara prosodi dan konten bahasa Inggris."
        else:
            kesimpulan_saran = f"Fokus utama latihan Anda berikutnya: {', '.join(improvements[:2])}"

        # 5. Output Terstruktur Terkalibrasi untuk UI
        ui_output = {
            "prediction": {
                "cefr_level": top_1_cefr,
                "confidence": round(top_1_conf * 100, 2), # Persentase (e.g. 78.50%)
                "alternative_cefr_level": top_2_cefr,
                "alternative_confidence": round(top_2_conf * 100, 2),
                "all_probabilities": {k: round(v * 100, 2) for k, v in cefr_prob_map.items()}
            },
            "prosody": {
                "score": round(predicted_prosody_score, 1), # Skor 0-100 (e.g. 84.5)
                "category": "Excellent" if predicted_prosody_score >= 80 else ("Good" if predicted_prosody_score >= 60 else "Need Improvement")
            },
            "analysis": {
                "strengths": strengths,
                "improvements": improvements,
                "summary_feedback": kesimpulan_saran
            }
        }
        
        return ui_output

# Blok Demo untuk pengujian CLI jika script dijalankan langsung
if __name__ == '__main__':
    print("=== Demo Uji Coba AI Inference Model ===")
    
    # 1. Inisialisasi predictor
    try:
        predictor = CEFRProsodyPredictor()
    except FileNotFoundError as e:
        print(e)
        exit(1)
        
    # 2. Contoh data input tiruan (seperti data yang didapat dari Web Speech API + Librosa DSP)
    # Ini merepresentasikan profil suara siswa level "B2" yang cukup baik
    sample_input = {
        'pitch_mean': 175.2,
        'pitch_std': 25.1,
        'pitch_contour_slope': 0.48,
        'energy_rms': 0.12,
        'duration_seconds': 2.4,
        'speech_rate': 2.8,
        'response_time_ms': 1100.0,
        
        # Kelancaran & Tata Bahasa
        'lexical_diversity': 0.72,
        'grammar_error_rate': 0.08,
        'pronunciation_accuracy': 82.5,
        'pause_ratio': 0.21,
        'filler_words_rate': 0.05,
        'wpm_consistency': 78.0,
        'asr_confidence': 0.88,
        'semantic_relevance': 81.0,
        
        # Deep representation (Whisper simulated output)
        'whisper_feat_1': 4.2,   # Nilai tinggi mencerminkan kelas CEFR atas (B2-C2)
        'whisper_feat_2': 1.8,
        'whisper_feat_3': 2.1,
        
        # Konteks User
        'user_prior_score': 74.0
    }
    
    # 3. Jalankan prediksi
    result = predictor.predict(sample_input)
    
    # 4. Tampilkan output yang diformat rapi untuk UI
    import json
    print("\n--- OUTPUT FORMAT JSON UNTUK UI ---")
    print(json.dumps(result, indent=2, ensure_ascii=False))
