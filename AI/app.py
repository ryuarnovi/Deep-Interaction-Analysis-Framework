import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from AI.inference import CEFRProsodyPredictor

app = FastAPI(
    title="CEFR & Prosody Analyzer API",
    description="Microservice untuk memprediksi level CEFR dan skor kemiripan prosodi suara berdasarkan parameter akustik dan linguistik.",
    version="1.0.0"
)

# Inisialisasi predictor (memuat model pkl saat server start)
try:
    predictor = CEFRProsodyPredictor(models_dir='models')
except Exception as e:
    print(f"Error memuat model: {e}")
    print("Harap pastikan file model (.pkl) berada di folder 'models/'.")
    predictor = None

# Definisi skema input Pydantic
class AnalysisInput(BaseModel):
    pitch_mean: float = Field(..., description="Rata-rata pitch (F0) suara", example=175.2)
    pitch_std: float = Field(..., description="Variasi pitch suara", example=25.1)
    pitch_contour_slope: float = Field(..., description="Kemiringan kontur intonasi", example=0.48)
    energy_rms: float = Field(..., description="Rata-rata energi RMS suara", example=0.12)
    duration_seconds: float = Field(..., description="Durasi rekaman suara dalam detik", example=2.4)
    speech_rate: float = Field(..., description="Kecepatan bicara (kata per detik)", example=2.8)
    response_time_ms: float = Field(..., description="Jeda sebelum menjawab dalam milidetik", example=1100.0)
    
    lexical_diversity: float = Field(..., description="Rasio variasi kosakata (0-1)", example=0.72)
    grammar_error_rate: float = Field(..., description="Rasio kesalahan tata bahasa (0-1)", example=0.08)
    pronunciation_accuracy: float = Field(..., description="Ketepatan pengucapan kata (0-100)", example=82.5)
    pause_ratio: float = Field(..., description="Rasio jeda diam vs total durasi (0-1)", example=0.21)
    filler_words_rate: float = Field(..., description="Rasio kata pengisi (uh, um) per kata (0-1)", example=0.05)
    wpm_consistency: float = Field(..., description="Stabilitas tempo bicara (0-100)", example=78.0)
    asr_confidence: float = Field(..., description="Tingkat akurasi transkripsi STT (0-1)", example=0.88)
    semantic_relevance: float = Field(..., description="Relevansi konten dengan topik (0-100)", example=81.0)
    
    whisper_feat_1: float = Field(..., description="Embedding fitur Whisper ke-1", example=4.2)
    whisper_feat_2: float = Field(..., description="Embedding fitur Whisper ke-2", example=1.8)
    whisper_feat_3: float = Field(..., description="Embedding fitur Whisper ke-3", example=2.1)
    
    user_prior_score: float = Field(..., description="Skor riwayat kemampuan user terdahulu (0-100)", example=74.0)

@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "CEFR & Prosody Analyzer API",
        "model_loaded": predictor is not None
    }

@app.post("/analyze", summary="Menganalisis audio berdasarkan 20 parameter input")
def analyze_voice(input_data: AnalysisInput):
    if predictor is None:
        raise HTTPException(
            status_code=500, 
            detail="Model AI tidak berhasil dimuat di server. Hubungi administrator."
        )
    try:
        # Jalankan prediksi dan kembalikan JSON terstruktur untuk UI
        result = predictor.predict(input_data.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
