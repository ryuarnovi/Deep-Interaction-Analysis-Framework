#!/bin/bash

# Setup script untuk Adaptive CEFR English Language & Prosody Modeling AI menggunakan DOCKER
# Author: Antigravity AI

set -e # Hentikan script jika ada error

echo "=========================================================="
# shellcheck disable=SC2154
echo " Memulai Automated ML Pipeline Setup via DOCKER"
echo "=========================================================="

# 1. Build Docker image
echo "Langkah 1/5: Membangun (building) Docker Image..."
docker build -t cefr-prosody-api .

# Buat folder lokal jika belum ada agar volume mounting tidak error
mkdir -p data models

# 2. Jalankan parse PDF
echo "Langkah 2/5: Menjalankan parsing PDF kosakata Oxford di dalam Docker..."
docker run --rm \
  -v "$(pwd)"/data:/app/data \
  -v "$(pwd)"/models:/app/models \
  cefr-prosody-api python AI/parse_oxford_pdf.py

# 3. Jalankan generate dataset
echo "Langkah 3/5: Membuat dataset simulasi suara di dalam Docker..."
docker run --rm \
  -v "$(pwd)"/data:/app/data \
  -v "$(pwd)"/models:/app/models \
  cefr-prosody-api python AI/generate_dataset.py

# 4. Jalankan clean data
echo "Langkah 4/5: Membersihkan dataset (IQR Outlier Filtering) di dalam Docker..."
docker run --rm \
  -v "$(pwd)"/data:/app/data \
  -v "$(pwd)"/models:/app/models \
  cefr-prosody-api python AI/clean_data.py

# 5. Jalankan model training
echo "Langkah 5/5: Melatih model CEFR & Prosody di dalam Docker..."
docker run --rm \
  -v "$(pwd)"/data:/app/data \
  -v "$(pwd)"/models:/app/models \
  cefr-prosody-api python AI/train.py

echo "=========================================================="
echo " Pipeline training sukses dijalankan via Docker!"
echo " Model (.pkl) dan dataset (.csv) tersimpan di folder lokal Anda."
echo "=========================================================="

# 6. Jalankan API Server
echo "Menjalankan API microservice di port 8000..."
# Hentikan container lama jika ada
docker rm -f cefr-prosody-service 2>/dev/null || true

docker run -d \
  -p 8000:8000 \
  -v "$(pwd)"/models:/app/models \
  --name cefr-prosody-service \
  cefr-prosody-api

echo "=========================================================="
echo " API Server FastAPI berhasil berjalan!"
echo " Silakan uji coba endpoint dengan menembak:"
echo " -> GET  http://localhost:8000/ (Cek status server)"
echo " -> POST http://localhost:8000/analyze (Kirim data fitur suara)"
echo "=========================================================="
