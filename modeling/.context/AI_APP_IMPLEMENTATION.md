#  AI App Generation Prompt: Adaptive English Speaking SBL & Intonasi

Dokumen ini berisi **System Prompt** siap pakai yang bisa Anda salin dan tempel (copy-paste) ke AI pembangun frontend (seperti **v0.dev**, **Bolt.new**, **Cursor (Composer)**, atau **Lovable.dev**) untuk membuat aplikasi web **Next.js (App Router)** yang memiliki UI premium, dinamis, dan terintegrasi dengan microservice AI yang telah dibuat di backend.

---

## Cara Penggunaan
1. **Salin teks prompt di bawah** secara utuh.
2. **Tempelkan ke chat box** AI pembangun UI pilihan Anda (misalnya v0.dev atau Bolt.new).
3. AI tersebut akan men-generate struktur folder, halaman UI, komponen animasi, dan integrasi API Next.js Route Handler (BFF).
4. Setelah UI selesai dibuat, jalankan backend FastAPI Anda (`docker-compose` atau `setup_docker.sh`) di port `8000` dan hubungkan Next.js BFF ke URL backend tersebut.

---

#  SALIN PROMPT DI BAWAH INI

```text
You are a world-class frontend developer and UI/UX designer. Your task is to build a premium, state-of-the-art Web Application for "Adaptive English Speaking & Intonation Analyzer" using Next.js (App Router), Tailwind CSS, Lucide Icons, and Framer Motion. 

This application connects to a FastAPI AI microservice that evaluates speaking performance based on 20 acoustic & linguistic features, returning CEFR levels (A1-C2), prosody scores, and localized Indonesian diagnostic feedback.

---

###  DESIGN SYSTEM & AESTHETICS (WOW FACTOR)
1. Theme: Sleek Dark Mode with cybernetic accents (deep blues, neon emeralds, royal purples, and soft slate).
2. Layout style: Glassmorphism panels (backdrop-filter: blur) with subtle gradient borders.
3. Typography: Premium clean sans-serif (e.g., Inter, Plus Jakarta Sans).
4. Animations: Smooth micro-interactions, pulsing record button, slide-in result cards, and dynamic radar/radial charts using Recharts or native SVG.
5. NO Placeholders: Use real, functional states, beautiful charts, and interactive controls.

---

### 📂 SYSTEM ARCHITECTURE & INTEGRATION
The frontend must communicate with the FastAPI backend via a Next.js BFF (Backend For Frontend) Route Handler (e.g., `app/api/analyze/route.ts`) to avoid CORS issues and secure requests.

#### FastAPI Endpoint Target:
- URL: `POST http://localhost:8000/analyze`
- Request Payload (JSON containing 20 features):
  ```json
  {
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
  }
  ```

- Expected API Response (Structured JSON in Indonesian):
  ```json
  {
    "prediction": {
      "cefr_level": "B2",
      "confidence": 78.5,
      "alternative_cefr_level": "B1",
      "alternative_confidence": 15.2,
      "all_probabilities": {
        "A1": 1.2, "A2": 2.5, "B1": 15.2, "B2": 78.5, "C1": 2.1, "C2": 0.5
      }
    },
    "prosody": {
      "score": 84.5,
      "category": "Excellent"
    },
    "analysis": {
      "strengths": [
        "Akurasi pengucapan (pronunciation) sangat baik dan terdengar natural.",
        "Aliran berbicara lancar dengan rasio jeda diam yang minimal."
      ],
      "improvements": [
        "Sering menggunakan kata pengisi seperti 'uh' atau 'um'. Cobalah kurangi jeda canggung ini."
      ],
      "summary_feedback": "Fokus utama latihan Anda berikutnya: Sering menggunakan kata pengisi seperti 'uh' atau 'um'."
    }
  }
  ```

---

###  KEY PAGES & COMPONENTS TO CREATE

#### 1. Scenario-Based Learning (SBL) Hub
- A dashboard showcasing learning scenarios (e.g., "Job Interview at Tech Company", "Checking in at London Hotel", "Scientific Pitching Session").
- Each scenario displays:
  - Theme/Context
  - Target difficulty (A2, B2, C1, etc.)
  - Progress status (Completed, In Progress, Locked)
- Adaptive Engine Indicator: A visual panel showing the user's current CEFR level (A1-C2) and how the system dynamically adjusts upcoming scenarios based on recent performance.

#### 2. Interactive Practice Screen (The Core Interface)
- **Camera-as-UI Component**: A modern floating video container simulating a real-time interview/presentation setting. User can toggle camera on/off.
- **Dialogue Prompt**: Shows the system's character prompt (e.g., Interviewer: "Tell me about your experience with React and Tailwind CSS").
- **Vocabulary Gap Filling**: Highlight target vocabulary (e.g., words derived from Oxford Vocabulary base) that the user should try to include.
- **Speech Recording Panel**:
  - Pulse-animated record button.
  - Live audio waveform representation.
  - Status display: "Mendengarkan...", "Menganalisis Audio...", "Mengukur Intonasi...".
  - *Simulation fallbacks*: Since capturing exact raw audio features in the browser requires complex web-audio parsing, implement a toggle/slider control panel for developer testing to override/simulate the 20 audio parameters, OR integrate Web Speech API (`webkitSpeechRecognition`) to dynamically calculate real parameters like `duration_seconds`, `speech_rate`, and `asr_confidence`.

#### 3. Real-Time AI Analysis Dashboard (Post-Inference)
- **CEFR level indicator**: Large circular progress/gauge displaying the predicted level (e.g., "B2") with its confidence percentage.
- **CEFR Level Probability Chart**: A clean bar/radar chart showing the distribution of probabilities for all CEFR levels (A1 to C2) after Temperature Scaling.
- **Prosody Similarity Score**: A beautiful gauge meter showing the prosody score (0-100) and its category (Excellent, Good, Need Improvement).
- **Diagnostic Feedback Panel**:
  - **Kelebihan (Strengths)**: Green tags/list items with check icons.
  - **Rencana Perbaikan (Improvements)**: Amber/Red warning cards with suggestions.
  - **Kesimpulan & Saran**: A highlighted card summarizing what to focus on next.
- **Interactive Feature Breakdown Accordion**: Group the 20 parameters into 3 categories with sliders/progress bars to show user scores compared to native speaker threshold values:
  1. *Kelancaran & Pengucapan (Fluency & Pronunciation)*: pronunciation_accuracy, pause_ratio, filler_words_rate, wpm_consistency.
  2. *Intonasi & Prosodi (Intonation & Rhythm)*: pitch_mean, pitch_std, pitch_contour_slope, energy_rms.
  3. *Relevansi Konten (Content & Semantics)*: lexical_diversity, grammar_error_rate, asr_confidence, semantic_relevance.

#### 4. Adaptive Progression Alert
- Trigger an overlay screen/modal when a user completes a scenario and triggers an automatic upgrade or downgrade:
  - "Selamat! Level CEFR Anda meningkat menjadi B2!"
  - "Menyesuaikan Skenario Latihan Berikutnya..."

---

###  CODE SPECIFICATIONS
- Create the Next.js Route Handler BFF at `app/api/analyze/route.ts` that safely forwards data to `http://localhost:8000/analyze`.
- Write high-quality, typed TypeScript interfaces for all inputs and API responses.
- Ensure responsive design (works beautifully on mobile and desktop viewports).
- Use local states for recording stages, API loading states, error states, and mock toggle indicators for testing.
```
