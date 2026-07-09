# CEFR Speech Coach: Implementasi AI untuk Penilaian Otomatis Kemampuan Berbicara Bahasa Inggris

Sistem kecerdasan buatan berbasis Machine Learning dan Deep Learning untuk menilai kemampuan berbicara bahasa Inggris secara adaptif dan otomatis menggunakan kerangka standar internasional CEFR (Common European Framework of Reference for Languages). 

Proyek ini menggabungkan ekstraksi fitur akustik (prosodi), fitur linguistik (fluency/pelafalan), serta representasi suara tingkat tinggi menggunakan model state-of-the-art speech-to-text.

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
├── makalah/                           # Dokumen akademis dan naskah publikasi ilmiah
│   ├── Makalah_CEFR_Speech_Coach.md   # Naskah makalah dalam format Markdown
│   ├── Makalah_CEFR_Speech_Coach.pdf   # Hasil ekspor dokumen ilmiah ke PDF
│   ├── Makalah_CEFR_Speech_Coach.docx  # Dokumen makalah dalam format Word
│   ├── convert_makalah.py             # Script utilitas untuk konversi format makalah
│   └── images/                        # Aset gambar pendukung makalah
├── modeling/                          # Pipeline kode Machine Learning dan Notebook eksperimen
│   ├── requirements.txt               # Daftar pustaka Python (dependencies)
│   ├── setup.sh                       # Script automasi setup environment & training pipeline
│   ├── CEFR Speech Coach.ipynb        # Jupyter Notebook interaktif untuk EDA & demo model
│   ├── kaggle_training.py             # Script latih model mandiri untuk di-copy ke Kaggle Notebook
│   ├── AI/                            # Modul utama pipeline ML
│   │   ├── parse_oxford_pdf.py        # Ekstrasi kosakata dari PDF Oxford 3000 & 5000
│   │   ├── generate_dataset.py        # Simulasi dataset fitur prosodi multi-speaker
│   │   ├── clean_data.py              # Pembersihan data & penanganan outlier (IQR method)
│   │   └── train.py                   # Script latih model klasifikasi & regresi prosodi
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

---

## Panduan Push ke Repository Git

Untuk menjaga repositori tetap bersih, ringan, dan cepat saat proses git push atau git clone, Anda hanya perlu mengunggah kode sumber, visualisasi, dan dokumen makalah. File berukuran besar atau bersifat dinamis harus diabaikan melalui file .gitignore.

### File yang Wajib di-Push (Diunggah ke Repo):
* Makalah & Gambar: Seluruh folder makalah/ (berisi PDF, Docx, MD, dan gambar-gambarnya).
* Kode Sumber: Seluruh script Python di dalam modeling/AI/, modeling/kaggle_training.py, dan modeling/CEFR Speech Coach.ipynb.
* Dokumentasi & Konfigurasi: README.md, modeling/requirements.txt, modeling/setup.sh, dan .gitignore.
* Dataset Referensi Statis: File PDF kosakata Oxford di modeling/models/.
* Visualisasi Hasil: File gambar grafik .png di dalam folder results/.

### File yang Harus Diabaikan dalam .gitignore (Dilarang di-Push):
* Virtual Environment (venv/, .venv/): Berisi instalasi dependensi lokal (~500 MB+).
* File Serialisasi Model (models/*.pkl, models/*.joblib): Bobot biner model terlatih. File ini besar dan bisa digenerasi ulang lewat training.
* Hasil Latih Deep Learning (results/wav2vec2-cefr/, results/cefr-wav2vec2-final/): Folder checkpoint Wav2Vec2 hasil training berukuran ratusan MB hingga GB.
* Dataset Dinamis (modeling/data/*.csv): Dataset CSV hasil simulasi yang dibuat otomatis oleh script.
* File ZIP Backup (CEFR Speech Coach Results.zip): Berukuran 312 MB (melebihi batas file GitHub sebesar 100 MB).

---

## Langkah Memulai (Getting Started)

### Prasyarat
* Python 3.10 atau versi di atasnya.
* Sistem operasi macOS, Linux, atau Windows (dengan Git Bash / WSL).

### Cara Cepat (Automated Setup)
Direktori modeling/ sudah dilengkapi dengan script shell setup.sh untuk melakukan inisialisasi lingkungan virtual python, instalasi dependensi, persiapan dataset, pembersihan data outlier, hingga proses training model secara otomatis.

1. Buka terminal dan masuk ke folder modeling:
   ```bash
   cd modeling
   ```
2. Berikan izin eksekusi dan jalankan script:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

---

### Cara Manual Step-by-Step

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

---

## Hasil Evaluasi Utama

Berdasarkan eksperimen dan naskah ilmiah yang dilampirkan pada folder makalah/, berikut adalah performa model yang berhasil dicapai:

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
