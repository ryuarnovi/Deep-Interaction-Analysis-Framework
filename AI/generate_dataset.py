import pandas as pd
import numpy as np
import random
import os

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_prosody_dataset(vocab_csv='data/oxford_vocabulary.csv', output_csv='data/oxford_prosody_dataset.csv', samples_per_word=2):
    if not os.path.exists(vocab_csv):
        print(f"File {vocab_csv} not found. Please run parse_oxford_pdf.py first.")
        return
        
    vocab_df = pd.read_csv(vocab_csv)
    
    # Synthesize C2 words if needed
    print("Synthesizing C2 vocabulary to ensure C2 class exists...")
    c2_words = [f"advanced_c2_word_{i}" for i in range(500)]
    c2_df = pd.DataFrame({'word': c2_words, 'cefr_level': 'C2'})
    vocab_df = pd.concat([vocab_df, c2_df], ignore_index=True)

    cefr_levels = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
    scenarios = ['interview', 'emergency', 'casual_chat', 'presentation', 'academic_discussion']
    
    # Generate 150 different speakers to simulate a multi-speaker environment
    speakers = [f"SPK_{i:03d}" for i in range(1, 151)]
    
    # We assign speaker accents or speaking styles (mean shifts)
    speaker_bias = {spk: np.random.normal(0, 5) for spk in speakers}
    
    data = []
    
    print(f"Generating rich features for {len(vocab_df)} words...")
    
    for _, row in vocab_df.iterrows():
        word = row['word']
        level = row['cefr_level']
        
        if level not in cefr_levels:
            level = 'B1'
            
        level_idx = cefr_levels.index(level)
        
        for _ in range(samples_per_word):
            scenario = np.random.choice(scenarios)
            speaker = np.random.choice(speakers)
            bias = speaker_bias[speaker]
            
            # --- Prosody Features ---
            pitch_mean = max(50, np.random.normal(loc=150 + level_idx*5 + bias, scale=15))
            pitch_std = max(1, np.random.normal(loc=30 - level_idx*2, scale=4)) 
            pitch_contour_slope = np.random.normal(loc=0.5, scale=0.15)
            energy_rms = max(0.01, np.random.normal(loc=0.1 + level_idx*0.01, scale=0.03))
            duration_seconds = max(0.5, np.random.normal(loc=3.0 - level_idx*0.2, scale=0.4))
            speech_rate = max(0.5, np.random.normal(loc=2.0 + level_idx*0.4, scale=0.4))
            response_time_ms = max(200, np.random.normal(loc=1500 - level_idx*180, scale=250))
            
            # --- Linguistic & Pronunciation Features (Sesuai masukan kritik) ---
            # 1. Lexical diversity (unique words ratio) - higher for advanced
            lexical_diversity = min(1.0, max(0.1, np.random.normal(loc=0.4 + level_idx*0.08, scale=0.06)))
            # 2. Grammar error rate - lower for advanced
            grammar_error_rate = min(0.6, max(0.0, np.random.normal(loc=0.25 - level_idx*0.04, scale=0.03)))
            # 3. Pronunciation accuracy (0-100) - higher for advanced
            pronunciation_accuracy = min(100.0, max(0.0, np.random.normal(loc=60.0 + level_idx*6.0, scale=5.0)))
            # 4. Pause ratio (silent duration / total duration) - lower for advanced
            pause_ratio = min(0.8, max(0.05, np.random.normal(loc=0.4 - level_idx*0.05, scale=0.04)))
            # 5. Filler words rate (e.g. uh, um per word) - lower for advanced
            filler_words_rate = min(0.5, max(0.0, np.random.normal(loc=0.18 - level_idx*0.025, scale=0.02)))
            # 6. WPM Consistency (variance in speaking speed) - higher (more stable) for advanced
            wpm_consistency = min(100.0, max(10.0, np.random.normal(loc=55.0 + level_idx*6.0, scale=6.0)))
            # 7. ASR confidence - higher for clear, advanced speakers
            asr_confidence = min(1.0, max(0.2, np.random.normal(loc=0.70 + level_idx*0.04, scale=0.05)))
            # 8. Semantic similarity to ideal answer (0-100) - higher for advanced
            semantic_relevance = min(100.0, max(10.0, np.random.normal(loc=50.0 + level_idx*7.0, scale=8.0)))
            
            # --- Simulated Whisper/Wav2Vec2 Pretrained Speech Representation Features ---
            # 3-dimension embeddings representing pre-trained model features that have high separation
            whisper_feat_1 = np.random.normal(loc=level_idx * 1.5, scale=1.0)
            whisper_feat_2 = np.random.normal(loc=(5 - level_idx) * 1.2, scale=1.0)
            whisper_feat_3 = np.random.normal(loc=(level_idx % 3) * 2.0, scale=0.8)

            user_prior_score = min(100, max(0, np.random.normal(loc=40 + level_idx*9, scale=8)))
            prosody_similarity = min(100, max(0, np.random.normal(loc=50 + level_idx*7, scale=8)))
            
            data.append({
                'word': word,
                'cefr_level': level,
                'speaker_id': speaker,
                'scenario_id': scenario,
                'pitch_mean': pitch_mean,
                'pitch_std': pitch_std,
                'pitch_contour_slope': pitch_contour_slope,
                'energy_rms': energy_rms,
                'duration_seconds': duration_seconds,
                'speech_rate': speech_rate,
                'response_time_ms': response_time_ms,
                'lexical_diversity': lexical_diversity,
                'grammar_error_rate': grammar_error_rate,
                'pronunciation_accuracy': pronunciation_accuracy,
                'pause_ratio': pause_ratio,
                'filler_words_rate': filler_words_rate,
                'wpm_consistency': wpm_consistency,
                'asr_confidence': asr_confidence,
                'semantic_relevance': semantic_relevance,
                'whisper_feat_1': whisper_feat_1,
                'whisper_feat_2': whisper_feat_2,
                'whisper_feat_3': whisper_feat_3,
                'user_prior_score': user_prior_score,
                'prosody_similarity': prosody_similarity
            })
        
    df = pd.DataFrame(data)
    
    os.makedirs('data', exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Generated {len(df)} prosody + linguistic samples and saved to {output_csv}")
    
if __name__ == "__main__":
    generate_prosody_dataset()
