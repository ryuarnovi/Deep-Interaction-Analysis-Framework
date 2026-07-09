import pandas as pd

def get_clean_display_df(df_input, num_rows=5):
    """
    Mengembalikan DataFrame yang telah diformat dengan sangat rapi (human-readable)
    untuk keperluan presentasi visual, dasbor, atau Jupyter Notebook.
    """
    clean_df = df_input.head(num_rows).copy()
    
    # 1. Normalisasi nama skenario agar ramah pengguna (UI friendly) — Taksonomi 8 Kategori
    scenario_map = {
        # 1. Daily Life
        'greeting_intro': 'Sapaan & Perkenalan',
        'shopping': 'Belanja',
        'ordering_food': 'Makan di Restoran',
        'asking_directions': 'Bertanya Arah',
        'small_talk': 'Obrolan Ringan',
        # 2. Workplace & Business
        'meeting_presentation': 'Rapat & Presentasi',
        'job_interview': 'Wawancara Kerja',
        'negotiation': 'Negosiasi',
        'business_call': 'Telepon Bisnis',
        'giving_feedback': 'Umpan Balik',
        # 3. Academic
        'class_discussion': 'Diskusi Kelas',
        'assignment_presentation': 'Presentasi Tugas',
        'academic_consultation': 'Bimbingan Akademik',
        'group_project': 'Kerja Kelompok',
        'oral_exam': 'Ujian Lisan',
        # 4. Social & Leisure
        'inviting': 'Undangan',
        'travel_story': 'Cerita Liburan',
        'team_sports': 'Olahraga Tim',
        'party_chat': 'Obrolan Pesta',
        'giving_advice': 'Memberi Saran',
        # 5. Public Services
        'airport': 'Bandara',
        'hotel': 'Hotel',
        'bank': 'Bank',
        'post_office': 'Kantor Pos',
        'pharmacy_clinic': 'Apotek / Klinik',
        # 6. Emergency & Health
        'emergency_call': 'Panggilan Darurat',
        'reporting_theft': 'Laporan Kehilangan',
        'doctor_consultation': 'Konsultasi Dokter',
        'first_aid': 'Pertolongan Pertama',
        # 7. Phone & Virtual
        'voicemail': 'Pesan Suara',
        'appointment_scheduling': 'Janji via Telepon',
        'video_conference': 'Konferensi Video',
        'customer_service': 'Layanan Pelanggan',
        # 8. Cross-cultural
        'local_customs': 'Kebiasaan Lokal',
        'clarifying_misunderstanding': 'Klarifikasi Miskomunikasi',
        'cultural_etiquette': 'Etiket Budaya'
    }
    if 'scenario_id' in clean_df.columns:
        clean_df['scenario_id'] = clean_df['scenario_id'].map(scenario_map).fillna(clean_df['scenario_id'])
    
    # 2. Hapus fitur Whisper representasi dalam (Deep Learning) karena tidak intuitif untuk user
    features_to_drop = ['whisper_feat_1', 'whisper_feat_2', 'whisper_feat_3']
    clean_df = clean_df.drop(columns=[col for col in features_to_drop if col in clean_df.columns], errors='ignore')
    
    # 3. Bulatkan metrik persentase & berikan simbol '%'
    percentage_cols = ['pronunciation_accuracy', 'wpm_consistency', 'semantic_relevance', 'user_prior_score', 'prosody_similarity']
    for col in percentage_cols:
        if col in clean_df.columns:
            clean_df[col] = clean_df[col].round(1).astype(str) + '%'
            
    # 4. Berikan pembulatan & satuan pada metrik audio/durasi
    if 'pitch_mean' in clean_df.columns:
        clean_df['pitch_mean'] = clean_df['pitch_mean'].round(1).astype(str) + ' Hz'
    if 'pitch_std' in clean_df.columns:
        clean_df['pitch_std'] = clean_df['pitch_std'].round(1)
    if 'pitch_contour_slope' in clean_df.columns:
        clean_df['pitch_contour_slope'] = clean_df['pitch_contour_slope'].round(2)
    if 'energy_rms' in clean_df.columns:
        clean_df['energy_rms'] = clean_df['energy_rms'].round(3)
    if 'duration_seconds' in clean_df.columns:
        clean_df['duration_seconds'] = clean_df['duration_seconds'].round(2).astype(str) + ' s'
    if 'speech_rate' in clean_df.columns:
        clean_df['speech_rate'] = clean_df['speech_rate'].round(1).astype(str) + ' syl/s'
    if 'pause_ratio' in clean_df.columns:
        clean_df['pause_ratio'] = clean_df['pause_ratio'].round(2)
    if 'filler_words_rate' in clean_df.columns:
        clean_df['filler_words_rate'] = clean_df['filler_words_rate'].round(2)
    if 'asr_confidence' in clean_df.columns:
        clean_df['asr_confidence'] = clean_df['asr_confidence'].round(2)
    if 'grammar_error_rate' in clean_df.columns:
        clean_df['grammar_error_rate'] = clean_df['grammar_error_rate'].round(2)
    if 'lexical_diversity' in clean_df.columns:
        clean_df['lexical_diversity'] = clean_df['lexical_diversity'].round(2)

    return clean_df


def get_scenario_category(scenario_id):
    """Mengembalikan kategori induk dari skenario berdasarkan Taksonomi 8 Kategori."""
    category_map = {
        # 1. Daily Life
        'greeting_intro': 1, 'shopping': 1, 'ordering_food': 1,
        'asking_directions': 1, 'small_talk': 1,
        # 2. Workplace & Business
        'meeting_presentation': 2, 'job_interview': 2, 'negotiation': 2,
        'business_call': 2, 'giving_feedback': 2,
        # 3. Academic
        'class_discussion': 3, 'assignment_presentation': 3,
        'academic_consultation': 3, 'group_project': 3, 'oral_exam': 3,
        # 4. Social & Leisure
        'inviting': 4, 'travel_story': 4, 'team_sports': 4,
        'party_chat': 4, 'giving_advice': 4,
        # 5. Public Services
        'airport': 5, 'hotel': 5, 'bank': 5,
        'post_office': 5, 'pharmacy_clinic': 5,
        # 6. Emergency & Health
        'emergency_call': 6, 'reporting_theft': 6,
        'doctor_consultation': 6, 'first_aid': 6,
        # 7. Phone & Virtual
        'voicemail': 7, 'appointment_scheduling': 7,
        'video_conference': 7, 'customer_service': 7,
        # 8. Cross-cultural
        'local_customs': 8, 'clarifying_misunderstanding': 8, 'cultural_etiquette': 8
    }
    category_names = {
        1: 'Daily Life',
        2: 'Workplace & Business',
        3: 'Academic',
        4: 'Social & Leisure',
        5: 'Public Services',
        6: 'Emergency & Health',
        7: 'Phone & Virtual',
        8: 'Cross-cultural'
    }
    cat_id = category_map.get(scenario_id, 0)
    return category_names.get(cat_id, 'Unknown')
