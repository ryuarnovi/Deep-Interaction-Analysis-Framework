import pandas as pd
import numpy as np
import os

def clean_dataset(input_path='data/oxford_prosody_dataset.csv', output_path='data/oxford_prosody_dataset_clean.csv'):
    if not os.path.exists(input_path):
        print(f"Error: File input {input_path} tidak ditemukan.")
        return None

    print(f"=== Memulai Proses Pembersihan Data: {input_path} ===")
    df = pd.read_csv(input_path)
    initial_rows = len(df)
    print(f"Jumlah baris awal: {initial_rows}")

    # 1. Menghapus Nilai Duplikat (Word & Speaker)
    # Jika pembicara yang sama mengucapkan kata yang sama lebih dari sekali, simpan entri terakhir
    df = df.drop_duplicates(subset=['word', 'speaker_id'], keep='last')
    print(f"Setelah menghapus duplikat pembicara-kata: {len(df)} baris")

    # 2. Penanganan Nilai Kosong (Missing Values / NaN)
    # Hapus baris jika label target (cefr_level atau prosody_similarity) kosong
    df = df.dropna(subset=['cefr_level', 'prosody_similarity'])
    
    # Imputasi nilai kosong untuk fitur numerik menggunakan nilai Median
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"  -> Imputasi median untuk nilai kosong pada kolom: {col}")

    # 3. Validasi Batas Logis Fitur (Logical Range Validation)
    # Memastikan metrik audio dan respons masuk dalam logika fisik yang benar
    df = df[
        (df['duration_seconds'] >= 0.1) &        # Durasi suara minimal 0.1 detik
        (df['speech_rate'] > 0) &                # Kecepatan bicara harus positif
        (df['response_time_ms'] >= 0) &          # Waktu respons tidak boleh negatif
        (df['prosody_similarity'] >= 0) & (df['prosody_similarity'] <= 100) # Skor kemiripan dalam batas 0-100
    ]
    print(f"Setelah validasi batas logis: {len(df)} baris")

    # 4. Deteksi & Penanganan Outlier Ekstrem (Metode IQR)
    # Outlier ekstrem biasanya terjadi karena noise mic, gangguan koneksi, atau error ekstraksi
    # Kita menggunakan batas longgar 3x IQR agar hanya menghapus anomali ekstrem
    for col in ['pitch_std', 'duration_seconds', 'response_time_ms']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3.0 * IQR
        upper_bound = Q3 + 3.0 * IQR
        
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    print(f"Setelah pembersihan outlier ekstrem (3.0 * IQR): {len(df)} baris")

    # 5. Normalisasi String / Teks
    df['word'] = df['word'].astype(str).str.strip().str.lower()
    # Hanya simpan kata yang valid secara alfabetik (termasuk karakter underscore dan angka untuk kata sintetis C2)
    df = df[df['word'].str.match(r'^[a-z\-0-9_]+$')]
    print(f"Setelah normalisasi teks kosakata: {len(df)} baris")

    # 6. Menyimpan Dataset Hasil Pembersihan
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    final_rows = len(df)
    
    print(f"=== Pembersihan Selesai! ===")
    print(f"Jumlah baris akhir: {final_rows} (Terfilter: {initial_rows - final_rows} baris)")
    
    return df

if __name__ == '__main__':
    clean_dataset()
