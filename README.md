# CEFR Speech Coach: Implementasi AI untuk Penilaian Otomatis Kemampuan Berbicara Bahasa Inggris

Sistem kecerdasan buatan berbasis Machine Learning dan Deep Learning untuk menilai kemampuan berbicara bahasa Inggris secara adaptif dan otomatis menggunakan kerangka standar internasional CEFR (Common European Framework of Reference for Languages). 

Proyek ini menggabungkan ekstraksi fitur akustik (prosodi), fitur linguistik (fluency/pelafalan), serta representasi suara tingkat tinggi menggunakan model state-of-the-art speech-to-text.

**Demo Aplikasi Web (Streamlit):** [https://cefr-speech-coach.streamlit.app/](https://cefr-speech-coach.streamlit.app/)

---

## Ringkasan Proyek

CEFR Speech Coach mengimplementasikan dua pendekatan utama untuk menilai kemampuan berbicara:
1. Machine Learning Tradisional: Menggunakan fitur terstruktur yang diekstraksi dari ucapan (seperti pitch, energi, durasi, lexical diversity, filler words, dan pause ratio) untuk mengklasifikasikan tingkat CEFR (A1-C2) dan memprediksi skor kemiripan prosodi (0-100). Algoritma yang digunakan meliputi Deep MLP, XGBoost, Random Forest, dan SVM (RBF Kernel).
2. Deep Learning (End-to-End): Melakukan fine-tuning pada model representasi audio Wav2Vec2 (facebook/wav2vec2-base) menggunakan teknik LoRA (Low-Rank Adaptation) untuk mengklasifikasikan tingkat CEFR langsung dari sinyal audio mentah.

---

## Fitur yang Digunakan dalam Pemodelan

Model menggunakan kombinasi fitur prosodi akustik dan fitur linguistik untuk memprediksi tingkat CEFR (A1 hingga C2) serta tingkat kemiripan prosodi (skala 0 hingga 100):

1. Fitur Prosodi Akustik:
   * pitch_mean: Nilai rata-rata frekuensi dasar (F0) dari ucapan.
   * pitch_std: Standar deviasi dari frekuensi dasar (F0) untuk mengukur variasi intonasi.
   * pitch_contour_slope: Kemiringan kontur pitch untuk menggambarkan dinamika nada bicara.
   * energy_rms: Nilai Root Mean Square dari amplitudo sinyal untuk mengukur intensitas energi suara.
   * duration_seconds: Durasi total sinyal ucapan dalam satuan detik.
   * speech_rate: Kecepatan berbicara yang dihitung berdasarkan jumlah kata atau suku kata per unit waktu.
   * response_time_ms: Waktu respons jeda sebelum mulai berbicara dalam milidetik.

2. Fitur Linguistik dan Kelancaran:
   * lexical_diversity: Keberagaman kosakata yang digunakan (misalnya Type-Token Ratio / TTR).
   * grammar_error_rate: Persentase kesalahan tata bahasa dalam kalimat yang diucapkan.
   * pronunciation_accuracy: Skor akurasi fonem dari ucapan dibandingkan dengan pelafalan standar.
   * pause_ratio: Rasio waktu jeda atau diam terhadap total durasi bicara.
   * filler_words_rate: Frekuensi penggunaan kata pengisi (seperti "uhm", "uh", "like") selama berbicara.
   * wpm_consistency: Konsistensi kecepatan berbicara dalam kata per menit.
   * asr_confidence: Tingkat keyakinan model Automatic Speech Recognition (ASR) terhadap hasil transkripsi.
   * semantic_relevance: Relevansi semantik jawaban pembicara terhadap topik atau skenario yang diberikan.

3. Fitur Representasi Audio Tingkat Tinggi:
   * whisper_feat_1, whisper_feat_2, whisper_feat_3: Vektor representasi fitur tersembunyi (hidden representation) yang diekstraksi dari model pra-latih Whisper untuk menangkap karakteristik semantik dan akustik suara yang lebih kompleks.

---

## Struktur Repositori

Jika Anda menginisialisasi Git pada direktori utama (cefr-speech-coach), struktur proyek Anda akan terlihat seperti berikut:

```text
cefr-speech-coach/
├── .gitignore                         # Mengabaikan venv, bobot model besar, dan dataset generatif
├── README.md                          # Dokumentasi utama proyek (file ini)
├── modeling/                          # Pipeline kode Machine Learning, API, dan Notebook eksperimen
│   ├── .context/                      # Dokumen arsitektur dan spesifikasi API
│   ├── .dockerignore                  # Daftar file diabaikan oleh Docker
│   ├── Dockerfile                     # Konfigurasi containerization untuk API
│   ├── Stacking_Ensemble_Model.ipynb  # Notebook pelatihan Stacking Ensemble
│   ├── CEFR Speech Coach.ipynb        # Jupyter Notebook interaktif untuk EDA & demo model
│   ├── requirements.txt               # Daftar pustaka Python (dependencies)
│   ├── setup.sh                       # Script automasi setup environment & training pipeline lokal
│   ├── setup_docker.sh                # Script automasi pipeline & API via Docker
│   ├── NOTE.md                        # Catatan arsitektur dan pengembangan model
│   ├── kaggle_training.py             # Script latih model mandiri untuk di-copy ke Kaggle Notebook
│   ├── AI/                            # Modul utama pipeline ML dan API
│   │   ├── parse_oxford_pdf.py        # Ekstrasi kosakata dari PDF Oxford 3000 & 5000
│   │   ├── generate_dataset.py        # Simulasi dataset fitur prosodi multi-speaker
│   │   ├── clean_data.py              # Pembersihan data & penanganan outlier (IQR method)
│   │   ├── train.py                   # Script latih model klasifikasi & regresi prosodi
│   │   ├── app.py                     # API microservice menggunakan FastAPI
│   │   └── inference.py               # Modul inferensi model klasifikasi dan regresi
│   └── models/                        # Dokumen kosa kata asli (bobot model .pkl di-ignore)
│       ├── Oxford 3000 CEFR Level.pdf
│       └── Oxford 5000 by CEFR Level.pdf
└── results/                           # Visualisasi grafik hasil evaluasi model
    ├── eda_distributions.png          # Visualisasi distribusi data fitur
    ├── final_comparison.png           # Grafik perbandingan akurasi model ML
    ├── partA_regression.png           # Hasil fitting model regresi prosodi
    ├── partA_results.png              # Grafik performa model klasifikasi CEFR
    └── partB_results.png              # Grafik performa fine-tuning Wav2Vec2 + LoRA
```



## Langkah Memulai (Getting Started)

### Metode 1: Menggunakan Docker (Direkomendasikan - Tanpa Memerlukan Python Lokal)

Pastikan Docker telah terpasang dan berjalan di sistem Anda.

#### 1. Menjalankan Docker Setup Otomatis
Anda dapat melakukan build image Docker, menjalankan seluruh pipeline data persiapan/training di dalam container volume, dan memulai service FastAPI dengan satu perintah:
```bash
cd modeling
chmod +x setup_docker.sh
./setup_docker.sh
```

Script ini akan:
- Membangun Docker image bernama `cefr-prosody-api`.
- Menjalankan pipeline langkah 1 hingga 4 di dalam container temporer, lalu menyalin hasil model `.pkl` dan dataset `.csv` ke direktori lokal `./models/` dan `./data/` melalui mounting volume.
- Memulai microservice API pada port container `8000`.

#### 2. Menguji Endpoint API
Setelah kontainer berjalan, Anda dapat berinteraksi dengan layanan API:
- Cek Status (GET):
  ```bash
  curl http://localhost:8000/
  ```
- Analisis Fitur Bicara (POST):
  ```bash
  curl -X POST http://localhost:8000/analyze \
    -H "Content-Type: application/json" \
    -d '{
      "pitch_mean": 175.2,
      "pitch_std": 25.1,
      "pitch_contour_slope": 0.48,
      "energy_rms": 0.12,
      "duration_seconds": 2.4,
      "speech_rate": 2.8,
      "response_time_ms": 1100.0,
      "lexical_diversity": 0.72,
      "grammar_error_rate": 0.08,
      "pronunciation_accuracy": 82.5,
      "pause_ratio": 0.21,
      "filler_words_rate": 0.05,
      "wpm_consistency": 78.0,
      "asr_confidence": 0.88,
      "semantic_relevance": 81.0,
      "whisper_feat_1": 4.2,
      "whisper_feat_2": 1.8,
      "whisper_feat_3": 2.1,
      "user_prior_score": 74.0
    }'
  ```

---

### Metode 2: Python Lokal & Virtual Environment (venv)

#### Prasyarat
* Python 3.10 atau versi di atasnya.
* Sistem operasi macOS, Linux, atau Windows (dengan Git Bash / WSL).

#### 1. Setup Otomatis Lokal
Masuk ke folder `modeling/` dan jalankan script setup untuk mempersiapkan venv, menginstal paket, dan melatih model secara lokal:
```bash
cd modeling
chmod +x setup.sh
./setup.sh
```

#### 2. Cara Manual Step-by-Step
Jika Anda ingin menjalankan setiap tahapan pipeline secara terpisah:

1. Membuat & Mengaktifkan Virtual Environment:
   ```bash
   python3 -m venv venv
   # Di macOS/Linux:
   source venv/bin/activate
   # Di Windows (Command Prompt):
   venv\Scripts\activate
   ```

2. Menginstal Dependensi:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. Langkah 1: Parsing PDF Oxford (Mengekstrak kosakata berdasarkan tingkat CEFR):
   ```bash
   python AI/parse_oxford_pdf.py
   ```

4. Langkah 2: Simulasi/Generasi Dataset (Membuat fitur prosodi akustik simulatif):
   ```bash
   python AI/generate_dataset.py
   ```

5. Langkah 3: Data Cleaning (Menyaring outlier menggunakan metode Interquartile Range / IQR):
   ```bash
   python AI/clean_data.py
   ```

6. Langkah 4: Training & Evaluasi Model:
   ```bash
   python AI/train.py
   ```
   *Model klasifikasi terbaik (best_cefr_classifier.pkl) dan model regressor (prosody_regressor.pkl) akan tersimpan di dalam folder models/.*

7. Memulai Server API secara Manual:
   ```bash
   uvicorn AI.app:app --host 0.0.0.0 --port 8000 --reload
   ```

---

### Menjalankan Notebook Eksperimen

Untuk membuka Jupyter notebook lokal (`CEFR Speech Coach.ipynb` atau `Stacking_Ensemble_Model.ipynb`) guna analisis data eksploratif (EDA) atau pelatihan model stacking ensemble:

```bash
source venv/bin/activate
jupyter notebook Stacking_Ensemble_Model.ipynb
```

---

## Hasil Evaluasi Utama

Berdasarkan hasil eksperimen pemodelan, berikut adalah performa model yang berhasil dicapai:

### 1. Klasifikasi Tingkat CEFR (Machine Learning Tradisional)
Evaluasi diuji menggunakan teknik Speaker-Independent Split (GroupShuffleSplit berdasarkan speaker_id) untuk mencegah kebocoran informasi (data leakage) karakteristik suara pembicara.

| Model | Akurasi | Macro Precision | Macro Recall | Macro F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Deep MLP** (Terbaik) | **88.31%** | **0.8769** | **0.8751** | **0.8752** |
| Random Forest | 85.06% | 0.8443 | 0.8415 | 0.8419 |
| XGBoost | 84.74% | 0.8396 | 0.8385 | 0.8387 |
| SVM (RBF Kernel) | 83.44% | 0.8258 | 0.8259 | 0.8247 |

### 2. Klasifikasi CEFR Tingkat Lanjut (Wav2Vec2 + LoRA)
Latih-ulang model audio Wav2Vec2 menggunakan teknik adaptasi parameter-efisien LoRA pada data audio nyata mencapai akurasi 41.34% dan Macro F1-Score 0.4471. Perbedaan hasil dipengaruhi oleh variasi noise, aksen, dan keterbatasan dataset audio nyata dibanding data simulasi terstruktur.

### 3. Prediksi Skor Prosodi (Regressor)
Model regresi Random Forest Regressor digunakan untuk memprediksi kecocokan prosodi (0-100) dan menghasilkan performa evaluasi:
* Root Mean Squared Error (RMSE): 8.2388
* Koefisien Determinasi (R2): 0.6301

---

## Menjalankan Eksperimen di Cloud (Kaggle)

Notebook resmi untuk melatih dan mengevaluasi model di platform Kaggle dapat diakses melalui tautan berikut:

[Notebook Kaggle CEFR Speech Coach](https://www.kaggle.com/code/ryuarnovi/cefr-speech-coach)

Jika Anda ingin melakukan pelatihan ulang model menggunakan GPU di Kaggle secara mandiri:
1. Kunjungi tautan Notebook Kaggle di atas, klik "Copy & Edit" untuk membuat salinan notebook di akun Anda.
2. Pastikan file dataset bersih `oxford_prosody_dataset_clean.csv` sudah ditambahkan sebagai input dataset pada panel kanan.
3. Anda juga dapat menggunakan script [kaggle_training.py](file://ML/cefr-speech-coach/modeling/kaggle_training.py) dan menyalin seluruh isi kodenya ke dalam satu cell kosong pada Kaggle Notebook baru.
4. Jalankan seluruh cell untuk melakukan training model, mengekspor model terlatih, serta menghasilkan visualisasi chart `confusion_matrix.png` dan `feature_importance.png` ke direktori keluaran (/kaggle/working/).
