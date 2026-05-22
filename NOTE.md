# 📋 NOTE — Penjelasan Alur Sistem & Pondasi Akurasi AI

> Dokumen ini menjelaskan alur kerja sistem **Aplikasi ML Adaptive Vocab (SBL & Intonasi)** berdasarkan mind map arsitektur proyek, serta bagaimana folder `AI/` berfungsi sebagai **pondasi akurasi** untuk seluruh sistem.

---

## Daftar Isi

- [Gambaran Umum Sistem](#gambaran-umum-sistem)
- [1. Pendahuluan, Gap, & Permasalahan](#1-pendahuluan-gap--permasalahan)
- [2. Arsitektur Sistem](#2-arsitektur-sistem)
- [3. Alur Analisis Suara](#3-alur-analisis-suara)
- [4. Model Machine Learning (Pondasi Akurasi)](#4-model-machine-learning-pondasi-akurasi)
- [5. Sistem Adaptif & CEFR](#5-sistem-adaptif--cefr)
- [6. Hasil & Evaluasi (Rencana)](#6-hasil--evaluasi-rencana)
- [Pemetaan Kode AI/ ke Mind Map](#pemetaan-kode-ai-ke-mind-map)

---

## Gambaran Umum Sistem

Sistem ini adalah **aplikasi pembelajaran Bahasa Inggris adaptif** yang menggabungkan Scenario-Based Learning (SBL) dengan analisis intonasi/prosodi suara menggunakan Machine Learning. Tujuan utamanya:

- **Menilai kualitas pengucapan** siswa secara real-time (bukan sekadar Speech-to-Text biasa).
- **Mengklasifikasikan level CEFR** (A1–C2) berdasarkan fitur suara multidimensi.
- **Menyesuaikan tingkat kesulitan** skenario secara otomatis berdasarkan prediksi ML.

```
                    ┌──────────────────────────────────────────┐
                    │                                          │
                    │   Aplikasi ML Adaptive Vocab             │
                    │   (SBL & Intonasi)                       │
                    │                                          │
                    └────────────────┬─────────────────────────┘
                                     │
          ┌──────────────┬───────────┼───────────┬──────────────┐
          │              │           │           │              │
          ▼              ▼           ▼           ▼              ▼
    ┌───────────┐  ┌──────────┐ ┌────────┐ ┌─────────┐  ┌──────────┐
    │ 1. Gap &  │  │ 2. Arsi- │ │ 3. Alur│ │ 4. Model│  │ 5. Sistem│
    │ Masalah   │  │ tektur   │ │ Suara  │ │ ML      │  │ Adaptif  │
    └───────────┘  └──────────┘ └────────┘ └─────────┘  └──────────┘
                                                │
                                                ▼
                                          ┌──────────┐
                                          │ 6. Hasil │
                                          │ Evaluasi │
                                          └──────────┘
```

---

## 1. Pendahuluan, Gap, & Permasalahan

### Masalah Utama
Metode pembelajaran bahasa Inggris konvensional memiliki kelemahan mendasar:
- **Hafalan tanpa konteks** — Siswa menghafal kosakata secara terisolasi, tanpa praktik di situasi nyata.
- **Evaluasi pengucapan sebatas teks biasa** — Sistem yang ada hanya membandingkan teks yang diucapkan (Speech-to-Text), bukan kualitas *bagaimana* kata tersebut diucapkan.

### Kesenjangan (Gap)

| Pendekatan Lama | Pendekatan Baru (Sistem Ini) |
|---|---|
| Frasa acak tanpa konteks | **Scenario-Based Learning (SBL)** — latihan di konteks nyata (wawancara, presentasi, dll.) |
| STT biasa (hanya verifikasi kata) | **Analisis Akustik/Intonasi** — mengukur pitch, stress, durasi, kelancaran |
| Level statis (manual) | **Dynamic Adaptivity via ML** — level CEFR otomatis naik/turun berdasarkan performa |

### Rumusan Masalah
1. Bagaimana membangun SBL interaktif di web?
2. Bagaimana mengukur kualitas intonasi (Pitch, Stress, Duration)?
3. Bagaimana ML menyesuaikan tingkat kesulitan secara otomatis (real-time)?

> **Jawaban**: Folder `AI/` berisi seluruh pipeline yang menjawab rumusan masalah #2 dan #3 melalui data generation, cleaning, dan model training.

---

## 2. Arsitektur Sistem

Sistem terdiri dari 3 lapisan teknologi:

```
┌─────────────────────────────────────────────────────────────┐
│                        FRONTEND                              │
│   Web Speech API (Input Suara via Browser)                   │
│   Camera-as-UI Concept                                       │
└──────────────────────────┬──────────────────────────────────┘
                           │  Audio stream / text
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                        BACKEND                               │
│   Node.js / Golang API                                       │
│   • Database (user progress, skor historis)                  │
│   • Tracking (sesi latihan, waktu respons)                   │
│   • REST API endpoint untuk komunikasi frontend ↔ ML         │
└──────────────────────────┬──────────────────────────────────┘
                           │  Feature vector / audio
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                      MICROSERVICE                            │
│   Python (Pemrosesan Sinyal & ML)                            │
│   • Ekstraksi fitur prosodi (DSP)                            │
│   • Inferensi model klasifikasi CEFR (.pkl)                  │
│   • Inferensi model regresi prosodi (.pkl)                   │
│   • Sumber kode: folder AI/                                  │
└─────────────────────────────────────────────────────────────┘
```

### Peran Folder `AI/` dalam Arsitektur

Folder `AI/` adalah **microservice layer** — seluruh kode di dalamnya bertanggung jawab untuk:
1. Membangun vocabulary base dari Oxford PDF → `parse_oxford_pdf.py`
2. Menghasilkan dataset simulasi multi-speaker → `generate_dataset.py`
3. Membersihkan data dari noise & outlier → `clean_data.py`
4. Melatih model klasifikasi & regresi → `train.py`

Model yang dihasilkan (`best_cefr_classifier.pkl`, `prosody_regressor.pkl`) kemudian di-*load* oleh microservice Python untuk melakukan inferensi real-time.

---

## 3. Alur Analisis Suara

Ini adalah inti dari bagaimana sistem menganalisis suara pengguna secara end-to-end:

```
┌─────────────────┐
│     INPUT        │
│ Suara user       │   User berbicara/menebak kosakata rumpang
│ (Microphone)     │   di dalam skenario SBL
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  VERIFIKASI KATA │
│  Speech-to-Text  │   Menggunakan Web Speech API / Whisper
│  (STT)           │   → Menghasilkan teks transkripsi
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│             EKSTRAKSI FITUR (DSP)                │
│                                                  │
│  Dari sinyal audio mentah, diekstrak:            │
│                                                  │
│  ┌──────────────┐  ┌──────────────┐              │
│  │ Pitch (F₀)   │  │ Stress       │              │
│  │ pitch_mean    │  │ energy_rms   │              │
│  │ pitch_std     │  │              │              │
│  │ pitch_contour │  │              │              │
│  │ _slope        │  │              │              │
│  └──────────────┘  └──────────────┘              │
│                                                  │
│  ┌──────────────┐  ┌──────────────┐              │
│  │ Duration     │  │ Fluency      │              │
│  │ duration_    │  │ speech_rate   │              │
│  │ seconds      │  │ pause_ratio   │              │
│  │ response_    │  │ filler_words  │              │
│  │ time_ms      │  │ _rate         │              │
│  └──────────────┘  └──────────────┘              │
│                                                  │
│  ┌──────────────────────────────────┐            │
│  │ Deep Representation (Whisper)    │            │
│  │ whisper_feat_1, _2, _3           │            │
│  └──────────────────────────────────┘            │
└────────────────────────┬────────────────────────┘
                         │
                         ▼
              ┌────────────────────┐
              │  MODEL ML (AI/)    │
              │  Klasifikasi CEFR  │
              │  Regresi Prosodi   │
              └────────┬───────────┘
                       │
                       ▼
              Prediksi Level + Skor
```

### Koneksi ke Kode `AI/`

| Tahap Analisis Suara | File di `AI/` | Penjelasan |
|---|---|---|
| Definisi fitur prosodi & linguistik | `generate_dataset.py` | Mendefinisikan 20 fitur yang diekstrak dari sinyal audio (baris 50–84) |
| Pembersihan data fitur | `clean_data.py` | Memastikan fitur valid secara fisik (durasi > 0.1s, speech_rate > 0, dll.) |
| Pelatihan model dari fitur | `train.py` | Fitur menjadi input X, model belajar memetakan fitur → CEFR level |

---

## 4. Model Machine Learning (Pondasi Akurasi)

> **Ini adalah bagian terpenting** — folder `AI/` berisi seluruh fondasi akurasi yang menentukan seberapa baik sistem dapat menilai kemampuan bahasa Inggris pengguna.

### 4.1 Pipeline Data sebagai Pondasi

Akurasi model sangat bergantung pada kualitas data. Pipeline di `AI/` memastikan data yang digunakan untuk training berkualitas tinggi:

```
Oxford PDF  ──▶  parse_oxford_pdf.py  ──▶  oxford_vocabulary.csv
(5,401 kata)       (Ekstraksi kosakata       (Kata + label CEFR)
                    dengan POS tag)
                         │
                         ▼
                 generate_dataset.py  ──▶  oxford_prosody_dataset.csv
                 (150 speaker simulasi       (10,802 sampel × 24 kolom)
                  5 skenario konteks
                  20 fitur numerik)
                         │
                         ▼
                    clean_data.py     ──▶  oxford_prosody_dataset_clean.csv
                 (Deduplikasi, IQR,          (Data bersih, siap training)
                  validasi logis,
                  normalisasi teks)
                         │
                         ▼
                      train.py        ──▶  *.pkl (model tersimpan)
```

### 4.2 Intonation Scorer — Klasifikasi CEFR

**Tujuan**: Memprediksi level kemampuan bahasa Inggris (A1, A2, B1, B2, C1, C2) dari fitur suara.

**Algoritma yang dilatih dan dibandingkan** (di `train.py`, baris 44–49):

| Model | Tipe | Peran | Kelebihan |
|---|---|---|---|
| **Random Forest** | Ensemble (Bagging) | Intonation Scorer utama | Tahan terhadap overfitting, menangani fitur non-linear |
| **SVM (RBF kernel)** | Kernel method | Intonation Scorer alternatif | Akurat di ruang berdimensi tinggi, margin maksimal |
| **XGBoost** | Ensemble (Boosting) | Intonation Scorer kompetitif | Cepat, menangani class imbalance, regularisasi |
| **Deep MLP** | Neural Network | Deep learner | Menangkap interaksi fitur kompleks |

**Proses seleksi otomatis** (baris 75–78):
```python
if f1 > best_f1:
    best_f1 = f1
    best_model = pipeline
    best_model_name = name
```
→ Model dengan **Macro F1-Score tertinggi** otomatis dipilih dan disimpan sebagai `best_cefr_classifier.pkl`.

**Mengapa Macro F1?** Karena dataset memiliki 6 kelas CEFR yang tidak seimbang. Macro F1 memastikan model tidak bias ke kelas mayoritas — setiap level CEFR dievaluasi secara adil.

### 4.3 Adaptive Engine — Regresi Prosodi

**Tujuan**: Memprediksi skor kemiripan prosodi (0–100) yang menunjukkan seberapa mirip pola intonasi siswa dengan pola ideal.

**Algoritma** (di `train.py`, baris 98–101):
```python
Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
])
```

**Metrik evaluasi**: RMSE (Root Mean Squared Error) — semakin rendah, semakin akurat.

**Kombinasi kedua model** inilah yang memungkinkan:
- **Intonation Scorer** menentukan "level apa yang sekarang?"
- **Adaptive Engine** menentukan "seberapa jauh dari ideal?" → dasar untuk keputusan upgrade/downgrade level.

### 4.4 Anti-Leakage: Menjaga Integritas Akurasi

Salah satu pondasi akurasi terpenting ada di **strategi pemisahan data** (baris 122–123):

```python
gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
train_idx, test_idx = next(gss.split(X, y_class, groups=speakers))
```

**Mengapa ini krusial?**

```
❌ TANPA GroupShuffleSplit (Data Leakage):
   Train: SPK_001 kata "apple", SPK_001 kata "banana"
   Test:  SPK_001 kata "cherry"
   → Model menghafal SUARA SPK_001, bukan kemampuan bahasanya
   → Akurasi di test set PALSU TINGGI (inflated)

✅ DENGAN GroupShuffleSplit (Anti-Leakage):
   Train: SPK_001 semua kata, SPK_002 semua kata, ...
   Test:  SPK_050 semua kata, SPK_051 semua kata, ...
   → Model HARUS generalisasi ke pembicara yang BELUM PERNAH didengar
   → Akurasi di test set REALISTIS
```

### 4.5 Pipeline Preprocessing dalam Model

Preprocessing dilakukan **di dalam Pipeline** (baris 58–61), bukan di luar:

```python
pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),    # StandardScaler fit di TRAIN saja
    ('classifier', model)
])
```

**Mengapa ini penting untuk akurasi?**
- `StandardScaler` di dalam pipeline berarti scaler hanya di-fit pada data training.
- Jika scaler di-fit di luar (pada seluruh data), informasi statistik data test "bocor" ke proses training → **data leakage** → akurasi palsu.

---

## 5. Sistem Adaptif & CEFR

Setelah model dilatih, sistem adaptif bekerja dengan 3 mekanisme:

### 5.1 Skoring

```
Skor Akhir = w₁ × Skor_Vocab + w₂ × Skor_Intonasi + w₃ × Skor_Waktu
```

- **Skor_Vocab**: Apakah kata yang diucapkan benar? (dari STT)
- **Skor_Intonasi**: Seberapa mirip prosodi dengan pola ideal? (dari `prosody_regressor.pkl`)
- **Skor_Waktu**: Berapa lama waktu respons? (`response_time_ms`)

### 5.2 Klasifikasi

Output dari `best_cefr_classifier.pkl` memberikan prediksi level CEFR:

```
Input fitur suara → Model → Prediksi: "B2" (probabilitas: 78%)
                                        "B1" (probabilitas: 15%)
                                        "C1" (probabilitas: 5%)
                                        ...
```

### 5.3 Adaptasi

Berdasarkan kombinasi skor dan klasifikasi, sistem otomatis:

```
┌──────────────────────────────────────────────────────┐
│                LOGIKA ADAPTASI                        │
│                                                       │
│  IF skor tinggi DAN prediksi CEFR ≥ level sekarang   │
│  THEN ──▶ UPGRADE (naikkan kesulitan skenario)        │
│                                                       │
│  IF skor rendah DAN prediksi CEFR < level sekarang   │
│  THEN ──▶ DOWNGRADE (turunkan kesulitan skenario)     │
│                                                       │
│  IF skor stabil DAN prediksi CEFR = level sekarang   │
│  THEN ──▶ MAINTAIN (variasi skenario di level sama)   │
└──────────────────────────────────────────────────────┘
```

Contoh adaptasi skenario:

| Level Sekarang | Prediksi ML | Skor Prosodi | Aksi |
|---|---|---|---|
| A2 | B1 | 82/100 | ⬆️ Upgrade ke skenario B1 |
| B2 | B1 | 45/100 | ⬇️ Downgrade ke skenario B1 |
| C1 | C1 | 70/100 | ↔️ Tetap di C1, ganti skenario |

---

## 6. Hasil & Evaluasi (Rencana)

### 6.1 Akurasi ML

Model dievaluasi dengan metrik standar yang diproduksi oleh `train.py`:

| Metrik | Penjelasan | Relevansi |
|---|---|---|
| **Precision** | Dari semua prediksi "B2", berapa % yang benar B2? | Menghindari false positive (label salah ke user) |
| **Recall** | Dari semua siswa B2 asli, berapa % yang terdeteksi? | Menghindari false negative (siswa tidak terdeteksi levelnya) |
| **F1-Score** | Harmonic mean precision & recall | Metrik utama seleksi model terbaik |
| **RMSE** | Error rata-rata prediksi skor prosodi | Akurasi model regresi prosodi |

Output evaluasi dari `train.py`:
```
Laporan Klasifikasi Model Terbaik:
              precision    recall  f1-score   support
          A1       0.xx      0.xx      0.xx       xxx
          A2       0.xx      0.xx      0.xx       xxx
          B1       0.xx      0.xx      0.xx       xxx
          B2       0.xx      0.xx      0.xx       xxx
          C1       0.xx      0.xx      0.xx       xxx
          C2       0.xx      0.xx      0.xx       xxx

Random Forest Regressor RMSE: x.xxxx
```

### 6.2 Dampak Belajar

Rencana evaluasi dampak terhadap pengguna:

```
Pre-test (Sebelum)              Post-test (Sesudah)
┌──────────────┐                ┌──────────────┐
│ Level CEFR   │                │ Level CEFR   │
│ siswa awal   │  ──▶ Latihan   │ siswa akhir  │
│              │   menggunakan  │              │
│ Skor prosodi │   sistem       │ Skor prosodi │
│ awal         │   adaptif      │ akhir        │
└──────────────┘                └──────────────┘
                      │
                      ▼
              Peningkatan = Post - Pre
```

### 6.3 Gamifikasi & UX

| Komponen | Metrik | Tujuan |
|---|---|---|
| **System Usability Scale (SUS)** | Skor 0–100 | Mengukur kemudahan penggunaan sistem |
| **Sistem Poin** | Poin per sesi latihan | Memotivasi konsistensi latihan |
| **Sistem Hint** | Jumlah hint yang dipakai | Mengukur tingkat kebutuhan bantuan |

---

## Pemetaan Kode AI/ ke Mind Map

Tabel berikut menunjukkan bagaimana setiap file di folder `AI/` menjadi **pondasi akurasi** yang memetakan langsung ke komponen mind map:

| File di `AI/` | Komponen Mind Map | Fungsi sebagai Pondasi Akurasi |
|---|---|---|
| `parse_oxford_pdf.py` | ① Pendahuluan | Membangun basis kosakata yang valid dari sumber resmi Oxford (5,401 kata × 6 level CEFR) |
| `generate_dataset.py` | ③ Alur Analisis Suara | Mendefinisikan fitur DSP yang diekstrak: Pitch (F₀), Stress (Energi RMS), Duration, Fluency, dan Deep Representations |
| `clean_data.py` | ③ Alur Analisis Suara | Menjamin kualitas data input model — data kotor = akurasi rendah |
| `train.py` (klasifikasi) | ④ Intonation Scorer | Melatih RF/SVM/XGB/MLP → memilih model terbaik untuk klasifikasi skor intonasi |
| `train.py` (regresi) | ④ Adaptive Engine | Melatih RandomForestRegressor → memprediksi probabilitas gagal/sukses prosodi |
| `train.py` (GroupShuffleSplit) | ④ Model ML | Anti-leakage → memastikan akurasi yang dilaporkan adalah REAL, bukan inflated |
| `inference.py` (UI Bridge) | ⑤ Sistem Adaptif & CEFR | Menjembatani model ML dan UI: Memuat model `.pkl`, melakukan *temperature scaling*, menganalisis kelemahan/kelebihan user, dan menghasilkan JSON terstruktur untuk UI |
| Output `.pkl` models | ⑤ Sistem Adaptif & CEFR | Model yang disimpan digunakan untuk skoring, klasifikasi level, dan adaptasi real-time |
| Metrik F1/RMSE | ⑥ Hasil & Evaluasi | Precision, Recall, F1-Score sebagai bukti akurasi sistem |

---

## Ringkasan Alur End-to-End

```
  ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
  │  Oxford   │     │ Dataset  │     │  Data    │     │  Model   │
  │  PDF      │────▶│ Generasi │────▶│ Cleaning │────▶│ Training │
  │ (sumber)  │     │ (150 spk)│     │ (IQR)    │     │ (4 model)│
  └──────────┘     └──────────┘     └──────────┘     └─────┬────┘
                                                           │
       ┌───────────────────────────────────────────────────┘
       │
       ▼
  ┌──────────┐     ┌────────────────┐     ┌──────────┐     ┌──────────┐
  │  Model   │     │  inference.py  │     │ Prediksi │     │ Adaptasi │
  │  .pkl    │────▶│ (UI Conclusion │────▶│ CEFR +   │────▶│ Skenario │
  │ (deploy) │     │   JSON Map)    │     │ Prosodi  │     │ Otomatis │
  └──────────┘     └────────────────┘     └──────────┘     └──────────┘
                                               │
                                               ▼
                                         ┌──────────┐
                                         │ Evaluasi │
                                         │ F1, RMSE │
                                         │ Pre/Post │
                                         └──────────┘
```

> **Kesimpulan**: Folder `AI/` bukan sekadar kumpulan skrip — ia adalah **keseluruhan pondasi akurasi** yang menentukan apakah sistem dapat menilai kemampuan bahasa Inggris dengan benar. Tanpa pipeline data yang bersih, model yang tervalidasi anti-leakage, model inference (`inference.py`) yang menerjemahkan angka ke simpulan UI manusiawi, dan metrik evaluasi yang rigorous, sistem adaptif di level 5 dan 6 tidak akan berfungsi dengan baik.
