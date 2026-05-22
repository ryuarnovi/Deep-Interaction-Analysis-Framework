# Prompt Pembuatan Aplikasi Frontend (Next.js)
## Sistem Pembelajaran Bahasa Inggris Adaptif (CEFR & Prosody)

Salin seluruh isi dokumen ini dan berikan kepada AI coding assistant Anda (seperti Cursor, Claude, atau v0.dev) untuk membangun aplikasi web Frontend yang terintegrasi dengan microservice AI yang telah dibuat.

---

```text
Buatkan aplikasi web berbasis Next.js (menggunakan React, TypeScript, dan Vanilla CSS / Tailwind CSS) untuk "Adaptive CEFR English Language & Prosody Learning System" yang memiliki antarmuka premium, interaktif, dan modern (Dark Mode, Glassmorphism, Neon Glow).

Aplikasi ini harus terintegrasi dengan microservice AI Python yang berjalan di http://localhost:8000/analyze.

Berikut adalah spesifikasi detail halaman, fitur, alur kerja, dan desain yang harus diimplementasikan:

---

### 1. Desain UI & Tema Visual (Aesthetics)
- Tema: Dark Mode premium dengan skema warna harmonis (Slate dark background, Electric Indigo/Violet accents, Emerald Green untuk indikator sukses, dan Amber untuk peringatan).
- Font: Gunakan Google Fonts 'Outfit' atau 'Inter' untuk kesan modern.
- Efek: Backdrop blur (glassmorphism), neon border glow untuk panel aktif, dan mikro-animasi transisi saat berpindah status.

---

### 2. Layout & Komponen Utama Halaman
Halaman utama dibagi menjadi 3 panel responsif (Dashboard 3 Kolom):

#### Kolom Kiri: Scenario-Based Learning (SBL) Selector
- Tampilkan daftar 5 skenario latihan dalam bentuk kartu interaktif:
  1. Job Interview (Wawancara Kerja)
  2. Presentation (Presentasi Proyek)
  3. Emergency Situation (Situasi Darurat)
  4. Academic Discussion (Diskusi Akademis)
  5. Casual Chat (Obrolan Santai)
- Setiap kartu memiliki icon representatif, deskripsi singkat, dan tingkat kesulitan awal (misal: "Target: B2").

#### Kolom Tengah: Interactive Camera-as-UI & Recording Panel
- **Camera-as-UI Box:** Panel webcam aktif (menggunakan getUserMedia API) dengan overlay bingkai neon bersinar. Webcam ini memberikan kesan interaktif langsung (Camera-as-UI).
- **Interactive Prompt Card:** Menampilkan instruksi kosakata rumpang atau tantangan kalimat berdasarkan skenario terpilih (misal: "Sebutkan alasan mengapa Anda tertarik bergabung... [kosakata kunci: 'Contribution', 'Opportunity']").
- **Voice Waveform Visualizer:** Saat mikrofon merekam, tampilkan animasi gelombang suara (canvas/CSS wave) untuk umpan balik visual bahwa suara sedang dideteksi.
- **Recording Controls:** Tombol "Mulai Latihan" (Record) dan "Selesai" (Stop). Gunakan Web Speech API bawaan browser untuk mentranskripsi suara siswa secara real-time di layar.

#### Kolom Kanan: Real-time Evaluation & CEFR Dashboard
Panel ini menampilkan respon JSON dari AI server setelah tombol "Selesai" ditekan:
- **CEFR Score Dial:** Lingkaran progress bar radial interaktif yang menampilkan Level CEFR Utama (A1-C2) dan skor keyakinan (Confidence %).
- **Prosody Gauge:** Slider horizontal/gauge untuk "Prosody Score" (0-100) lengkap dengan kategori visual (Need Improvement / Good / Excellent).
- **Strengths (Kelebihan):** Daftar poin dengan ikon centang hijau (Emerald) berdasarkan respon JSON `analysis.strengths`.
- **Improvements (Saran Perbaikan):** Daftar poin dengan ikon lampu/panah merah berdasarkan respon JSON `analysis.improvements`.
- **Summary Feedback Bubble:** Balon chat berisi ringkasan feedback lisan bahasa Indonesia dari AI.

---

### 3. Alur Pengumpulan Data & Integrasi API

#### A. Backend for Frontend (BFF) di Next.js
Buat file Route Handler Next.js (`app/api/evaluate/route.ts` atau `pages/api/evaluate.ts`) untuk mem-forward request dari Frontend ke Docker AI Microservice secara aman tanpa memicu masalah CORS browser.

Endpoint internal Next.js:
POST /api/evaluate -> Mengirim parameter ke http://localhost:8000/analyze
Next.js harus mengirim balik data mentah JSON hasil analisis ke frontend.

#### B. Pengumpulan Fitur di Frontend (Client-side Speech Simulation)
Saat user menekan tombol "Mulai", aktifkan Web Speech API. Setelah tombol "Selesai" ditekan, kumpulkan metrik suara secara dinamis:
- `duration_seconds`: Hitung selang waktu antara Start dan Stop.
- `response_time_ms`: Hitung jeda milidetik sebelum pengguna mulai berbicara pertama kali.
- `speech_rate`: Jumlah kata yang diucapkan dibagi durasi detik.
- `pause_ratio`: Estimasi rasio diam (dapat disimulasikan dari jeda antar kata di Web Speech API).
- `pronunciation_accuracy`, `grammar_error_rate`, `semantic_relevance`: Nilai keakuratan yang disimulasikan dari teks transkripsi dibandingkan dengan kunci jawaban skenario.
- `whisper_feat_1`, `_2`, `_3`: Kirim nilai representasi acak/simulasi sesuai dengan level target skenario saat ini.
- `user_prior_score`: Skor historis pengguna saat ini (disimpan di state/localStorage).

Kirim seluruh 20 parameter tersebut ke POST `/api/evaluate` dalam format JSON berikut:
{
  "pitch_mean": 175.2, "pitch_std": 25.1, "pitch_contour_slope": 0.48, "energy_rms": 0.12,
  "duration_seconds": 2.4, "speech_rate": 2.8, "response_time_ms": 1100.0,
  "lexical_diversity": 0.72, "grammar_error_rate": 0.08, "pronunciation_accuracy": 82.5,
  "pause_ratio": 0.21, "filler_words_rate": 0.05, "wpm_consistency": 78.0,
  "asr_confidence": 0.88, "semantic_relevance": 81.0,
  "whisper_feat_1": 4.2, "whisper_feat_2": 1.8, "whisper_feat_3": 2.1,
  "user_prior_score": 74.0
}

---

### 4. Fitur Adaptasi Otomatis (Dynamic Level Adjuster)
- Jika level CEFR yang dikembalikan dari API lebih tinggi atau sama dengan level skenario saat ini, tampilkan notifikasi toast animasi: "Selamat! Level Anda naik. Membuka skenario berikutnya!"
- Ubah skenario selanjutnya secara otomatis ke tingkat kosakata yang lebih menantang (A2 -> B1 -> B2 dst).
- Simpan progress level CEFR tertinggi pengguna di LocalStorage untuk memantau kemajuan belajar jangka panjang.

Bangun kode aplikasi ini dengan struktur folder Next.js App Router yang bersih. Tulis styling CSS dalam modul-modul modular agar performa pemuatan super cepat.
```
