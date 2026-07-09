# 1. Gunakan base image Python yang ringan
FROM python:3.10-slim

# 2. Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Tentukan working directory di dalam container
WORKDIR /app

# 4. Install dependency sistem yang dibutuhkan oleh XGBoost (libgomp1)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 5. Salin requirements.txt terlebih dahulu untuk memanfaatkan cache layer Docker
COPY requirements.txt /app/

# 6. Install dependensi Python
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 7. Salin seluruh file proyek ke dalam working directory container
COPY . /app/

# 8. Expose port 8000 untuk API FastAPI
EXPOSE 8000

# 9. Jalankan FastAPI server menggunakan Uvicorn
CMD ["uvicorn", "AI.app:app", "--host", "0.0.0.0", "--port", "8000"]
