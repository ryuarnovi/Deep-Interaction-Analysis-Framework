"""
CEFR Speech Coach — Streamlit Dashboard

Aplikasi web interaktif untuk analisis, demonstrasi model, dan evaluasi
sistem penilaian otomatis kemampuan berbicara Bahasa Inggris berbasis CEFR.

Soal 4 UAS: Deployment & Streamlit Application.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.metrics import (
    confusion_matrix, classification_report,
    mean_squared_error, r2_score
)
from sklearn.model_selection import GroupShuffleSplit

# ──────────────────────────────────────────────────────────────────────
# PAGE CONFIG & GLOBAL STYLES
# ──────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CEFR Speech Coach",
    page_icon="C",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6C63FF 0%, #48C6EF 50%, #6F86D6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        line-height: 1.2;
    }
    .hero-subtitle {
        font-size: 1.0rem;
        color: #8892B0;
        margin-bottom: 1.5rem;
        font-weight: 300;
    }

    .metric-card {
        background: linear-gradient(135deg, #1A1F2E 0%, #252B3B 100%);
        border: 1px solid rgba(108, 99, 255, 0.2);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 32px rgba(108, 99, 255, 0.15);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #6C63FF;
        margin: 0.3rem 0;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #8892B0;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .cefr-badge {
        display: inline-block;
        padding: 0.4rem 1.2rem;
        border-radius: 24px;
        font-weight: 600;
        font-size: 1.6rem;
        margin: 0.5rem 0;
    }
    .cefr-A1 { background: linear-gradient(135deg, #00C9FF, #92FE9D); color: #0E1117; }
    .cefr-A2 { background: linear-gradient(135deg, #43E97B, #38F9D7); color: #0E1117; }
    .cefr-B1 { background: linear-gradient(135deg, #F6D365, #FDA085); color: #0E1117; }
    .cefr-B2 { background: linear-gradient(135deg, #FFA726, #FF7043); color: #0E1117; }
    .cefr-C1 { background: linear-gradient(135deg, #A18CD1, #FBC2EB); color: #0E1117; }
    .cefr-C2 { background: linear-gradient(135deg, #6C63FF, #48C6EF); color: #FAFAFA; }

    .section-divider {
        height: 3px;
        background: linear-gradient(90deg, #6C63FF 0%, transparent 100%);
        border: none;
        margin: 2rem 0;
        border-radius: 2px;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0E1117 0%, #1A1F2E 100%);
    }

    .info-box {
        background: rgba(108, 99, 255, 0.08);
        border-left: 4px solid #6C63FF;
        border-radius: 0 12px 12px 0;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
    }

    .strength-item {
        background: rgba(67, 233, 123, 0.08);
        border-left: 4px solid #43E97B;
        border-radius: 0 8px 8px 0;
        padding: 0.7rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.95rem;
    }
    .improvement-item {
        background: rgba(255, 112, 67, 0.08);
        border-left: 4px solid #FF7043;
        border-radius: 0 8px 8px 0;
        padding: 0.7rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.95rem;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; padding: 8px 20px; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────
# DATA & MODEL LOADING
# ──────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_dataset():
    path = os.path.join(BASE_DIR, "data", "oxford_prosody_dataset_clean.csv")
    if not os.path.exists(path):
        st.error(f"Dataset tidak ditemukan: {path}")
        return None
    return pd.read_csv(path)

@st.cache_resource
def load_models():
    models_dir = os.path.join(BASE_DIR, "models")
    clf_path = os.path.join(models_dir, "best_cefr_classifier.pkl")
    reg_path = os.path.join(models_dir, "prosody_regressor.pkl")
    le_path = os.path.join(models_dir, "label_encoder.pkl")
    models = {}
    if os.path.exists(clf_path):
        models['classifier'] = joblib.load(clf_path)
    if os.path.exists(reg_path):
        models['regressor'] = joblib.load(reg_path)
    if os.path.exists(le_path):
        models['label_encoder'] = joblib.load(le_path)
    return models

FEATURE_COLUMNS = [
    'pitch_mean', 'pitch_std', 'pitch_contour_slope', 'energy_rms',
    'duration_seconds', 'speech_rate', 'response_time_ms',
    'lexical_diversity', 'grammar_error_rate', 'pronunciation_accuracy',
    'pause_ratio', 'filler_words_rate', 'wpm_consistency',
    'asr_confidence', 'semantic_relevance',
    'whisper_feat_1', 'whisper_feat_2', 'whisper_feat_3',
    'user_prior_score'
]

CEFR_LEVELS = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
CEFR_COLORS = {
    'A1': '#00C9FF', 'A2': '#43E97B', 'B1': '#F6D365',
    'B2': '#FF7043', 'C1': '#A18CD1', 'C2': '#6C63FF'
}
CEFR_COLOR_SEQ = ['#00C9FF', '#43E97B', '#F6D365', '#FF7043', '#A18CD1', '#6C63FF']


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def temperature_scale(probs, T=2.5):
    eps = 1e-15
    probs = np.clip(probs, eps, 1 - eps)
    logits = np.log(probs / (1 - probs))
    return softmax(logits / T)


# ──────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0;">
        <h2 style="background: linear-gradient(135deg, #6C63FF, #48C6EF);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                    margin: 0.5rem 0 0 0; font-size: 1.4rem;">
            CEFR Speech Coach
        </h2>
        <p style="color: #8892B0; font-size: 0.8rem; margin-top: 0.2rem;">
            Penilaian Otomatis Bahasa Inggris
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "Navigasi",
        [
            "Dashboard EDA",
            "Model Demo",
            "Evaluasi Model",
            "Interpretasi Hasil",
            "Dokumentasi"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:#555; font-size:0.75rem; padding:1rem 0;">
        <p>Ryu Arnovi</p>
        <p>UAS Machine Learning 2025</p>
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────
# PAGE 1: DASHBOARD EDA
# ──────────────────────────────────────────────────────────────────────

if page == "Dashboard EDA":
    st.markdown('<h1 class="hero-title">Exploratory Data Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Distribusi data fitur prosodi dan linguistik dari dataset Oxford Vocabulary</p>', unsafe_allow_html=True)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    df = load_dataset()
    if df is None:
        st.stop()

    # Overview Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Sampel</div>
            <div class="metric-value">{len(df):,}</div>
            <div class="metric-label">data points</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Kata Unik</div>
            <div class="metric-value">{df['word'].nunique():,}</div>
            <div class="metric-label">kosakata Oxford</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Jumlah Speaker</div>
            <div class="metric-value">{df['speaker_id'].nunique()}</div>
            <div class="metric-label">pembicara simulasi</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Level CEFR</div>
            <div class="metric-value">6</div>
            <div class="metric-label">A1 -- C2</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Distribusi CEFR", "Distribusi Fitur", "Korelasi",
        "Box Plot", "Scatter Plot"
    ])

    with tab1:
        st.subheader("Distribusi Sampel per Level CEFR")
        cefr_counts = df['cefr_level'].value_counts().reindex(CEFR_LEVELS)
        fig = go.Figure(data=[
            go.Bar(
                x=CEFR_LEVELS,
                y=cefr_counts.values,
                marker=dict(
                    color=CEFR_COLOR_SEQ,
                    line=dict(color='rgba(255,255,255,0.1)', width=1)
                ),
                text=cefr_counts.values,
                textposition='outside',
                textfont=dict(size=14, color='#FAFAFA')
            )
        ])
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(title="Level CEFR", tickfont=dict(size=14)),
            yaxis=dict(title="Jumlah Sampel", gridcolor='rgba(255,255,255,0.05)'),
            height=450, margin=dict(t=40)
        )
        st.plotly_chart(fig, width="stretch")

        st.subheader("Distribusi Skenario Percakapan")
        scenario_counts = df['scenario_id'].value_counts().head(15)
        fig2 = go.Figure(data=[
            go.Bar(
                y=scenario_counts.index, x=scenario_counts.values,
                orientation='h',
                marker=dict(color=scenario_counts.values, colorscale='Viridis',
                            line=dict(color='rgba(255,255,255,0.1)', width=1)),
                text=scenario_counts.values, textposition='outside'
            )
        ])
        fig2.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(title="Jumlah Sampel", gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(title=""), height=500, margin=dict(l=10, t=30)
        )
        st.plotly_chart(fig2, width="stretch")

    with tab2:
        st.subheader("Distribusi Fitur Prosodi dan Linguistik")
        prosody_feats = ['pitch_mean', 'pitch_std', 'energy_rms', 'speech_rate',
                         'duration_seconds', 'response_time_ms', 'pronunciation_accuracy',
                         'lexical_diversity']
        for i in range(0, len(prosody_feats), 2):
            row_feats = prosody_feats[i:i+2]
            cols = st.columns(len(row_feats))
            for j, feat in enumerate(row_feats):
                with cols[j]:
                    fig = px.histogram(
                        df, x=feat, color='cefr_level',
                        category_orders={'cefr_level': CEFR_LEVELS},
                        color_discrete_sequence=CEFR_COLOR_SEQ,
                        nbins=50, barmode='overlay', opacity=0.7,
                        title=f"Distribusi: {feat}"
                    )
                    fig.update_layout(
                        template='plotly_dark',
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        height=350, showlegend=(i == 0 and j == 0),
                        margin=dict(t=40, b=30),
                        yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
                    )
                    st.plotly_chart(fig, width="stretch")

    with tab3:
        st.subheader("Heatmap Korelasi Antar Fitur")
        numeric_cols = df[FEATURE_COLUMNS + ['prosody_similarity']].corr()
        fig, ax = plt.subplots(figsize=(14, 11))
        mask = np.triu(np.ones_like(numeric_cols, dtype=bool))
        sns.heatmap(
            numeric_cols, mask=mask, annot=True, fmt='.2f',
            cmap='RdYlBu_r', center=0, square=True, linewidths=0.5,
            ax=ax, annot_kws={"size": 7}, cbar_kws={"shrink": 0.8}
        )
        ax.set_facecolor('#0E1117')
        fig.patch.set_facecolor('#0E1117')
        ax.tick_params(colors='#FAFAFA', labelsize=8)
        ax.set_title("Korelasi Pearson Antar Fitur", color='#FAFAFA', fontsize=14, pad=15)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab4:
        st.subheader("Box Plot Fitur per Level CEFR")
        selected_feature = st.selectbox(
            "Pilih fitur:", FEATURE_COLUMNS + ['prosody_similarity'], index=0
        )
        fig = px.box(
            df, x='cefr_level', y=selected_feature, color='cefr_level',
            category_orders={'cefr_level': CEFR_LEVELS},
            color_discrete_sequence=CEFR_COLOR_SEQ,
            title=f"Distribusi {selected_feature} per Level CEFR"
        )
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            height=500, showlegend=False,
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
        )
        st.plotly_chart(fig, width="stretch")

        st.subheader("Statistik Deskriptif per Level CEFR")
        stats = df.groupby('cefr_level')[selected_feature].describe().round(2)
        stats = stats.reindex(CEFR_LEVELS)
        st.dataframe(stats, width="stretch")

    with tab5:
        st.subheader("Scatter Plot Interaktif")
        col_x, col_y = st.columns(2)
        with col_x:
            x_axis = st.selectbox("Sumbu X:", FEATURE_COLUMNS, index=0)
        with col_y:
            y_axis = st.selectbox("Sumbu Y:", FEATURE_COLUMNS, index=9)

        fig = px.scatter(
            df.sample(min(3000, len(df)), random_state=42),
            x=x_axis, y=y_axis, color='cefr_level',
            category_orders={'cefr_level': CEFR_LEVELS},
            color_discrete_sequence=CEFR_COLOR_SEQ,
            opacity=0.6, title=f"{x_axis} vs {y_axis} (sampel 3000)"
        )
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            height=550,
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
        )
        st.plotly_chart(fig, width="stretch")


# ──────────────────────────────────────────────────────────────────────
# PAGE 2: MODEL DEMO
# ──────────────────────────────────────────────────────────────────────

elif page == "Model Demo":
    st.markdown('<h1 class="hero-title">Model Demo — Prediksi CEFR</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Input parameter fitur suara untuk mendapatkan prediksi level CEFR dan skor prosodi</p>', unsafe_allow_html=True)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    models = load_models()
    if 'classifier' not in models or 'label_encoder' not in models:
        st.error("File model .pkl tidak ditemukan. Pastikan file ada di folder models/.")
        st.stop()

    # Preset Profiles
    st.markdown("### Profil Cepat")
    presets = {
        "A1": {
            'pitch_mean': 130.0, 'pitch_std': 35.0, 'pitch_contour_slope': 0.3,
            'energy_rms': 0.06, 'duration_seconds': 3.5, 'speech_rate': 1.5,
            'response_time_ms': 2200.0, 'lexical_diversity': 0.25, 'grammar_error_rate': 0.22,
            'pronunciation_accuracy': 50.0, 'pause_ratio': 0.45, 'filler_words_rate': 0.2,
            'wpm_consistency': 45.0, 'asr_confidence': 0.6, 'semantic_relevance': 40.0,
            'whisper_feat_1': 0.0, 'whisper_feat_2': 6.0, 'whisper_feat_3': 0.0,
            'user_prior_score': 30.0
        },
        "A2": {
            'pitch_mean': 145.0, 'pitch_std': 30.0, 'pitch_contour_slope': 0.38,
            'energy_rms': 0.08, 'duration_seconds': 3.2, 'speech_rate': 1.9,
            'response_time_ms': 1850.0, 'lexical_diversity': 0.35, 'grammar_error_rate': 0.18,
            'pronunciation_accuracy': 58.0, 'pause_ratio': 0.38, 'filler_words_rate': 0.16,
            'wpm_consistency': 52.0, 'asr_confidence': 0.67, 'semantic_relevance': 50.0,
            'whisper_feat_1': 1.5, 'whisper_feat_2': 4.8, 'whisper_feat_3': 2.0,
            'user_prior_score': 42.0
        },
        "B1": {
            'pitch_mean': 155.0, 'pitch_std': 26.0, 'pitch_contour_slope': 0.45,
            'energy_rms': 0.10, 'duration_seconds': 2.8, 'speech_rate': 2.4,
            'response_time_ms': 1500.0, 'lexical_diversity': 0.50, 'grammar_error_rate': 0.14,
            'pronunciation_accuracy': 68.0, 'pause_ratio': 0.30, 'filler_words_rate': 0.11,
            'wpm_consistency': 62.0, 'asr_confidence': 0.76, 'semantic_relevance': 62.0,
            'whisper_feat_1': 3.0, 'whisper_feat_2': 3.6, 'whisper_feat_3': 4.0,
            'user_prior_score': 55.0
        },
        "B2": {
            'pitch_mean': 165.0, 'pitch_std': 22.0, 'pitch_contour_slope': 0.5,
            'energy_rms': 0.12, 'duration_seconds': 2.5, 'speech_rate': 2.8,
            'response_time_ms': 1100.0, 'lexical_diversity': 0.65, 'grammar_error_rate': 0.09,
            'pronunciation_accuracy': 80.0, 'pause_ratio': 0.22, 'filler_words_rate': 0.06,
            'wpm_consistency': 75.0, 'asr_confidence': 0.87, 'semantic_relevance': 78.0,
            'whisper_feat_1': 4.5, 'whisper_feat_2': 2.4, 'whisper_feat_3': 2.0,
            'user_prior_score': 70.0
        },
        "C1": {
            'pitch_mean': 175.0, 'pitch_std': 18.0, 'pitch_contour_slope': 0.58,
            'energy_rms': 0.14, 'duration_seconds': 2.1, 'speech_rate': 3.4,
            'response_time_ms': 700.0, 'lexical_diversity': 0.78, 'grammar_error_rate': 0.05,
            'pronunciation_accuracy': 90.0, 'pause_ratio': 0.15, 'filler_words_rate': 0.03,
            'wpm_consistency': 85.0, 'asr_confidence': 0.92, 'semantic_relevance': 88.0,
            'whisper_feat_1': 6.0, 'whisper_feat_2': 1.2, 'whisper_feat_3': 0.0,
            'user_prior_score': 82.0
        },
        "C2": {
            'pitch_mean': 185.0, 'pitch_std': 15.0, 'pitch_contour_slope': 0.65,
            'energy_rms': 0.16, 'duration_seconds': 1.8, 'speech_rate': 3.8,
            'response_time_ms': 500.0, 'lexical_diversity': 0.88, 'grammar_error_rate': 0.02,
            'pronunciation_accuracy': 96.0, 'pause_ratio': 0.10, 'filler_words_rate': 0.01,
            'wpm_consistency': 92.0, 'asr_confidence': 0.96, 'semantic_relevance': 95.0,
            'whisper_feat_1': 7.5, 'whisper_feat_2': 0.5, 'whisper_feat_3': 2.0,
            'user_prior_score': 92.0
        },
        "Manual Input": None
    }

    selected_preset = st.radio("Pilih profil:", list(presets.keys()), horizontal=True)
    defaults = presets[selected_preset] if presets[selected_preset] is not None else presets["B2"]

    # Input Form
    st.markdown("### Parameter Input")
    with st.expander("Fitur Prosodi Akustik", expanded=True):
        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            pitch_mean = st.slider("Pitch Mean (Hz)", 50.0, 300.0, defaults['pitch_mean'], 0.5)
            energy_rms = st.slider("Energy RMS", 0.01, 0.3, defaults['energy_rms'], 0.005)
            response_time_ms = st.slider("Response Time (ms)", 100.0, 5000.0, defaults['response_time_ms'], 50.0)
        with pc2:
            pitch_std = st.slider("Pitch Std Dev", 1.0, 60.0, defaults['pitch_std'], 0.5)
            duration_seconds = st.slider("Duration (s)", 0.5, 8.0, defaults['duration_seconds'], 0.1)
        with pc3:
            pitch_contour_slope = st.slider("Pitch Contour Slope", -0.5, 1.5, defaults['pitch_contour_slope'], 0.05)
            speech_rate = st.slider("Speech Rate (words/s)", 0.5, 6.0, defaults['speech_rate'], 0.1)

    with st.expander("Fitur Linguistik dan Kelancaran", expanded=True):
        lc1, lc2, lc3 = st.columns(3)
        with lc1:
            lexical_diversity = st.slider("Lexical Diversity", 0.1, 1.0, defaults['lexical_diversity'], 0.01)
            pause_ratio = st.slider("Pause Ratio", 0.05, 0.8, defaults['pause_ratio'], 0.01)
            asr_confidence = st.slider("ASR Confidence", 0.2, 1.0, defaults['asr_confidence'], 0.01)
        with lc2:
            grammar_error_rate = st.slider("Grammar Error Rate", 0.0, 0.5, defaults['grammar_error_rate'], 0.01)
            filler_words_rate = st.slider("Filler Words Rate", 0.0, 0.4, defaults['filler_words_rate'], 0.01)
            semantic_relevance = st.slider("Semantic Relevance", 10.0, 100.0, defaults['semantic_relevance'], 1.0)
        with lc3:
            pronunciation_accuracy = st.slider("Pronunciation Accuracy", 0.0, 100.0, defaults['pronunciation_accuracy'], 0.5)
            wpm_consistency = st.slider("WPM Consistency", 10.0, 100.0, defaults['wpm_consistency'], 1.0)
            user_prior_score = st.slider("User Prior Score", 0.0, 100.0, defaults['user_prior_score'], 1.0)

    with st.expander("Fitur Representasi Deep Learning"):
        wc1, wc2, wc3 = st.columns(3)
        with wc1:
            whisper_feat_1 = st.slider("Whisper Feat 1", -5.0, 12.0, defaults['whisper_feat_1'], 0.1)
        with wc2:
            whisper_feat_2 = st.slider("Whisper Feat 2", -5.0, 12.0, defaults['whisper_feat_2'], 0.1)
        with wc3:
            whisper_feat_3 = st.slider("Whisper Feat 3", -5.0, 8.0, defaults['whisper_feat_3'], 0.1)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    if st.button("Jalankan Prediksi", type="primary", width="stretch"):
        input_data = pd.DataFrame([{
            'pitch_mean': pitch_mean, 'pitch_std': pitch_std,
            'pitch_contour_slope': pitch_contour_slope, 'energy_rms': energy_rms,
            'duration_seconds': duration_seconds, 'speech_rate': speech_rate,
            'response_time_ms': response_time_ms, 'lexical_diversity': lexical_diversity,
            'grammar_error_rate': grammar_error_rate, 'pronunciation_accuracy': pronunciation_accuracy,
            'pause_ratio': pause_ratio, 'filler_words_rate': filler_words_rate,
            'wpm_consistency': wpm_consistency, 'asr_confidence': asr_confidence,
            'semantic_relevance': semantic_relevance, 'whisper_feat_1': whisper_feat_1,
            'whisper_feat_2': whisper_feat_2, 'whisper_feat_3': whisper_feat_3,
            'user_prior_score': user_prior_score
        }])

        with st.spinner("Memproses prediksi..."):
            clf = models['classifier']
            le = models['label_encoder']
            raw_probs = clf.predict_proba(input_data)[0]
            cal_probs = temperature_scale(raw_probs, T=2.5)
            classes = le.classes_

            prob_map = {classes[i]: float(cal_probs[i]) for i in range(len(classes))}
            sorted_probs = sorted(prob_map.items(), key=lambda x: x[1], reverse=True)
            top_level, top_conf = sorted_probs[0]
            alt_level, alt_conf = sorted_probs[1]

            prosody_score = None
            if 'regressor' in models:
                prosody_score = float(models['regressor'].predict(input_data)[0])
                prosody_score = max(0, min(100, prosody_score))

        # Display Results
        st.markdown("## Hasil Prediksi")

        res_col1, res_col2, res_col3 = st.columns([1.2, 1, 1])
        with res_col1:
            st.markdown(f"""
            <div class="metric-card" style="padding: 2rem;">
                <div class="metric-label">Prediksi Level CEFR</div>
                <div class="cefr-badge cefr-{top_level}">{top_level}</div>
                <div style="font-size: 1.3rem; color: #FAFAFA; margin-top: 0.5rem;">
                    Confidence: <strong>{top_conf*100:.1f}%</strong>
                </div>
                <div style="color: #8892B0; font-size: 0.9rem; margin-top: 0.3rem;">
                    Alternatif: {alt_level} ({alt_conf*100:.1f}%)
                </div>
            </div>
            """, unsafe_allow_html=True)

        with res_col2:
            if prosody_score is not None:
                cat = "Excellent" if prosody_score >= 80 else ("Good" if prosody_score >= 60 else "Perlu Latihan")
                cat_color = "#43E97B" if prosody_score >= 80 else ("#F6D365" if prosody_score >= 60 else "#FF7043")
                st.markdown(f"""
                <div class="metric-card" style="padding: 2rem;">
                    <div class="metric-label">Skor Prosodi</div>
                    <div class="metric-value" style="color: {cat_color};">{prosody_score:.1f}</div>
                    <div style="color: {cat_color}; font-size: 1.1rem;">{cat}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Model regresi tidak tersedia.")

        with res_col3:
            radar_vals = [pronunciation_accuracy/100, lexical_diversity,
                          asr_confidence, semantic_relevance/100, wpm_consistency/100]
            radar_labels = ['Pengucapan', 'Kosakata', 'Kejelasan', 'Relevansi', 'Konsistensi']
            fig_radar = go.Figure(data=go.Scatterpolar(
                r=radar_vals + [radar_vals[0]],
                theta=radar_labels + [radar_labels[0]],
                fill='toself',
                fillcolor='rgba(108, 99, 255, 0.15)',
                line=dict(color='#6C63FF', width=2),
                marker=dict(size=6, color='#6C63FF')
            ))
            fig_radar.update_layout(
                polar=dict(
                    bgcolor='rgba(0,0,0,0)',
                    radialaxis=dict(visible=True, range=[0, 1], gridcolor='rgba(255,255,255,0.1)'),
                    angularaxis=dict(gridcolor='rgba(255,255,255,0.1)')
                ),
                template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                showlegend=False, height=300,
                margin=dict(t=30, b=30, l=30, r=30),
                title=dict(text="Profil Kemampuan", font=dict(size=13, color='#8892B0'))
            )
            st.plotly_chart(fig_radar, width="stretch")

        # Probability distribution
        st.markdown("### Distribusi Probabilitas Semua Level CEFR")
        fig_probs = go.Figure(data=[
            go.Bar(
                x=CEFR_LEVELS,
                y=[prob_map.get(lv, 0)*100 for lv in CEFR_LEVELS],
                marker=dict(
                    color=[CEFR_COLORS[lv] for lv in CEFR_LEVELS],
                    line=dict(color='rgba(255,255,255,0.2)', width=1)
                ),
                text=[f"{prob_map.get(lv,0)*100:.1f}%" for lv in CEFR_LEVELS],
                textposition='outside', textfont=dict(size=13, color='#FAFAFA')
            )
        ])
        fig_probs.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(title="Level CEFR"),
            yaxis=dict(title="Probabilitas (%)", range=[0, 100], gridcolor='rgba(255,255,255,0.05)'),
            height=380, margin=dict(t=30)
        )
        st.plotly_chart(fig_probs, width="stretch")

        # Feedback Analysis
        st.markdown("### Analisis dan Feedback")
        strengths = []
        improvements = []

        if pronunciation_accuracy >= 80:
            strengths.append("Akurasi pengucapan baik dan terdengar natural.")
        elif pronunciation_accuracy < 60:
            improvements.append("Perlu meningkatkan kejelasan pengucapan kata (artikulasi).")

        if pause_ratio > 0.35:
            improvements.append("Terlalu banyak jeda diam. Latih kelancaran aliran kalimat.")
        else:
            strengths.append("Aliran berbicara lancar dengan jeda diam minimal.")

        if filler_words_rate > 0.15:
            improvements.append("Frekuensi kata pengisi (uh, um) tinggi. Kurangi jeda canggung.")

        if prosody_score is not None:
            if prosody_score >= 75:
                strengths.append("Ritme dan intonasi sudah mirip penutur asli.")
            elif prosody_score < 50:
                improvements.append("Intonasi terdengar datar. Gunakan variasi nada saat mengekspresikan kalimat.")

        if semantic_relevance >= 75:
            strengths.append("Jawaban relevan dengan konteks skenario.")
        elif semantic_relevance < 50:
            improvements.append("Penyampaian pesan kurang fokus pada konteks skenario.")

        if response_time_ms > 2000:
            improvements.append("Waktu berpikir sebelum menjawab agak lama. Latih spontanitas.")

        fc1, fc2 = st.columns(2)
        with fc1:
            st.markdown("#### Kekuatan")
            if strengths:
                for s in strengths:
                    st.markdown(f'<div class="strength-item">{s}</div>', unsafe_allow_html=True)
            else:
                st.info("Belum ada kekuatan yang terdeteksi.")
        with fc2:
            st.markdown("#### Area Perbaikan")
            if improvements:
                for imp in improvements:
                    st.markdown(f'<div class="improvement-item">{imp}</div>', unsafe_allow_html=True)
            else:
                st.success("Semua aspek sudah baik.")


# ──────────────────────────────────────────────────────────────────────
# PAGE 3: MODEL EVALUATION
# ──────────────────────────────────────────────────────────────────────

elif page == "Evaluasi Model":
    st.markdown('<h1 class="hero-title">Evaluasi Model</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Perbandingan performa model klasifikasi CEFR dan regresi prosodi</p>', unsafe_allow_html=True)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Model Comparison Table
    st.markdown("### Perbandingan Performa Model Klasifikasi")
    st.markdown("""
    <div class="info-box">
        Evaluasi menggunakan <strong>Speaker-Independent Split</strong> (GroupShuffleSplit berdasarkan speaker_id)
        untuk mencegah kebocoran data (data leakage) karakteristik suara pembicara ke test set.
    </div>
    """, unsafe_allow_html=True)

    comparison_data = {
        'Model': ['Deep MLP (Terbaik)', 'Random Forest', 'XGBoost', 'SVM (RBF Kernel)'],
        'Akurasi': ['88.31%', '85.06%', '84.74%', '83.44%'],
        'Macro Precision': [0.8769, 0.8443, 0.8396, 0.8258],
        'Macro Recall': [0.8751, 0.8415, 0.8385, 0.8259],
        'Macro F1-Score': [0.8752, 0.8419, 0.8387, 0.8247]
    }
    st.dataframe(pd.DataFrame(comparison_data), width="stretch", hide_index=True)

    # Bar chart comparison
    models_list = ['Deep MLP', 'Random Forest', 'XGBoost', 'SVM']
    metrics = {
        'Precision': [0.8769, 0.8443, 0.8396, 0.8258],
        'Recall': [0.8751, 0.8415, 0.8385, 0.8259],
        'F1-Score': [0.8752, 0.8419, 0.8387, 0.8247]
    }
    fig_comp = go.Figure()
    bar_colors = ['#6C63FF', '#48C6EF', '#43E97B']
    for i, (metric_name, values) in enumerate(metrics.items()):
        fig_comp.add_trace(go.Bar(
            name=metric_name, x=models_list, y=values,
            marker_color=bar_colors[i],
            text=[f"{v:.4f}" for v in values], textposition='outside',
            textfont=dict(size=11)
        ))
    fig_comp.update_layout(
        barmode='group', template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=450,
        yaxis=dict(range=[0.78, 0.92], title="Skor", gridcolor='rgba(255,255,255,0.05)'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(t=60)
    )
    st.plotly_chart(fig_comp, width="stretch")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    tab_eval1, tab_eval2, tab_eval3 = st.tabs([
        "Confusion Matrix", "Feature Importance", "Regresi Prosodi"
    ])

    df = load_dataset()
    models_loaded = load_models()

    with tab_eval1:
        st.subheader("Confusion Matrix — Model Terbaik (Deep MLP)")
        if df is not None and 'classifier' in models_loaded and 'label_encoder' in models_loaded:
            clf = models_loaded['classifier']
            le = models_loaded['label_encoder']
            X = df.drop(columns=['word', 'cefr_level', 'speaker_id', 'scenario_id', 'prosody_similarity'])
            y_true_encoded = le.transform(df['cefr_level'])
            speakers = df['speaker_id']

            gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
            train_idx, test_idx = next(gss.split(X, y_true_encoded, groups=speakers))
            X_test = X.iloc[test_idx]
            y_test = y_true_encoded[test_idx]

            y_pred = clf.predict(X_test)
            all_labels = list(range(len(le.classes_)))
            cm = confusion_matrix(y_test, y_pred, labels=all_labels)

            fig_cm = go.Figure(data=go.Heatmap(
                z=cm, x=le.classes_, y=le.classes_,
                colorscale='Purples',
                text=cm, texttemplate="%{text}",
                textfont=dict(size=14, color='white'),
                hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>"
            ))
            fig_cm.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(title="Predicted Level", side='bottom'),
                yaxis=dict(title="Actual Level", autorange='reversed'),
                height=500, margin=dict(t=30)
            )
            st.plotly_chart(fig_cm, width="stretch")

            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred, labels=all_labels, target_names=le.classes_, output_dict=True, zero_division=0)
            st.dataframe(pd.DataFrame(report).transpose().round(4), width="stretch")
        else:
            st.warning("Model atau dataset tidak tersedia.")

    with tab_eval2:
        st.subheader("Feature Importance — Model Classifier")
        if 'classifier' in models_loaded:
            clf = models_loaded['classifier']
            try:
                classifier_step = clf.named_steps.get('classifier', None)
                if classifier_step is None:
                    for step_name, step_obj in clf.steps:
                        if hasattr(step_obj, 'feature_importances_'):
                            classifier_step = step_obj
                            break

                if hasattr(classifier_step, 'feature_importances_'):
                    importances = classifier_step.feature_importances_
                    feat_imp_df = pd.DataFrame({
                        'Feature': FEATURE_COLUMNS,
                        'Importance': importances
                    }).sort_values('Importance', ascending=True)

                    fig_imp = go.Figure(data=[
                        go.Bar(
                            y=feat_imp_df['Feature'], x=feat_imp_df['Importance'],
                            orientation='h',
                            marker=dict(color=feat_imp_df['Importance'], colorscale='Viridis',
                                        line=dict(width=1, color='rgba(255,255,255,0.1)')),
                            text=[f"{v:.4f}" for v in feat_imp_df['Importance']],
                            textposition='outside'
                        )
                    ])
                    fig_imp.update_layout(
                        template='plotly_dark',
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        height=550,
                        xaxis=dict(title="Importance Score", gridcolor='rgba(255,255,255,0.05)'),
                        margin=dict(l=10, t=30)
                    )
                    st.plotly_chart(fig_imp, width="stretch")
                else:
                    st.info("Model terbaik (MLP) tidak memiliki atribut `feature_importances_`.")
                    st.markdown("""
                    Deep MLP menggunakan weight matrices internal yang tidak langsung
                    dapat diinterpretasikan sebagai feature importance. Untuk model
                    berbasis tree (Random Forest, XGBoost), feature importance tersedia
                    secara langsung dari algoritma.
                    """)
            except Exception as e:
                st.warning(f"Gagal mengekstrak feature importance: {e}")
        else:
            st.warning("Model classifier tidak tersedia.")

    with tab_eval3:
        st.subheader("Evaluasi Regresi — Prosody Similarity Score")
        if df is not None and 'regressor' in models_loaded:
            reg = models_loaded['regressor']
            X = df.drop(columns=['word', 'cefr_level', 'speaker_id', 'scenario_id', 'prosody_similarity'])
            y_reg = df['prosody_similarity']
            speakers = df['speaker_id']

            gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
            train_idx, test_idx = next(gss.split(X, y_reg, groups=speakers))
            X_test = X.iloc[test_idx]
            y_test = y_reg.iloc[test_idx]
            y_pred_reg = reg.predict(X_test)

            rmse = np.sqrt(mean_squared_error(y_test, y_pred_reg))
            r2 = r2_score(y_test, y_pred_reg)

            mc1, mc2 = st.columns(2)
            with mc1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">RMSE</div>
                    <div class="metric-value">{rmse:.4f}</div>
                    <div class="metric-label">Root Mean Squared Error</div>
                </div>""", unsafe_allow_html=True)
            with mc2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">R-squared</div>
                    <div class="metric-value">{r2:.4f}</div>
                    <div class="metric-label">Koefisien Determinasi</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            fig_reg = go.Figure()
            fig_reg.add_trace(go.Scatter(
                x=y_test.values, y=y_pred_reg, mode='markers',
                marker=dict(color='#6C63FF', size=4, opacity=0.4), name='Prediksi'
            ))
            line_range = [min(y_test.min(), y_pred_reg.min()), max(y_test.max(), y_pred_reg.max())]
            fig_reg.add_trace(go.Scatter(
                x=line_range, y=line_range, mode='lines',
                line=dict(color='#FF7043', dash='dash', width=2), name='Prediksi Sempurna'
            ))
            fig_reg.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(title="Actual Prosody Score", gridcolor='rgba(255,255,255,0.05)'),
                yaxis=dict(title="Predicted Prosody Score", gridcolor='rgba(255,255,255,0.05)'),
                height=500, margin=dict(t=30)
            )
            st.plotly_chart(fig_reg, width="stretch")

            st.subheader("Distribusi Residual")
            residuals = y_test.values - y_pred_reg
            fig_resid = px.histogram(
                x=residuals, nbins=60,
                labels={'x': 'Residual (Actual - Predicted)', 'count': 'Frequency'},
                color_discrete_sequence=['#6C63FF']
            )
            fig_resid.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                height=350, yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                margin=dict(t=30)
            )
            st.plotly_chart(fig_resid, width="stretch")
        else:
            st.warning("Model regresi atau dataset tidak tersedia.")


# ──────────────────────────────────────────────────────────────────────
# PAGE 4: INTERPRETASI HASIL
# ──────────────────────────────────────────────────────────────────────

elif page == "Interpretasi Hasil":
    st.markdown('<h1 class="hero-title">Interpretasi Hasil</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Penjelasan model, insights bisnis, dan rekomendasi pengembangan</p>', unsafe_allow_html=True)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown("### Cara Kerja Model")
    st.markdown("Sistem menggunakan dua model ML yang saling melengkapi:")

    imc1, imc2 = st.columns(2)
    with imc1:
        st.markdown("""
        <div class="metric-card" style="text-align:left; padding: 1.5rem;">
            <h4 style="color: #6C63FF;">Intonation Scorer (Classifier)</h4>
            <p style="color: #C0C0C0;">Memprediksi level CEFR (A1-C2) dari 19 fitur akustik dan linguistik.</p>
            <ul style="color: #8892B0;">
                <li>Model terbaik: <strong>Deep MLP</strong></li>
                <li>Macro F1-Score: <strong>0.8752</strong></li>
                <li>Metrik seleksi: Macro F1 (adil untuk kelas imbalanced)</li>
                <li>Pembanding: RF, XGBoost, SVM</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with imc2:
        st.markdown("""
        <div class="metric-card" style="text-align:left; padding: 1.5rem;">
            <h4 style="color: #48C6EF;">Adaptive Engine (Regressor)</h4>
            <p style="color: #C0C0C0;">Memprediksi skor kemiripan prosodi (0-100) dengan pola ideal native speaker.</p>
            <ul style="color: #8892B0;">
                <li>Algoritma: <strong>Random Forest Regressor</strong></li>
                <li>RMSE: <strong>~8.24</strong></li>
                <li>R-squared: <strong>~0.63</strong></li>
                <li>Output: skor kontinu untuk keputusan adaptasi</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### Keputusan Teknis")

    with st.expander("Anti-Leakage: GroupShuffleSplit", expanded=True):
        st.markdown("""
        **Masalah:** Jika data dari pembicara yang sama muncul di training dan test set, model menghafal
        karakter suara pembicara, bukan kemampuan bahasanya. Akurasi menjadi inflated.

        **Solusi:** `GroupShuffleSplit(groups=speaker_id)` memastikan seluruh data satu pembicara
        hanya ada di satu split (train atau test, tidak keduanya).

        | Aspek | Tanpa GroupShuffleSplit | Dengan GroupShuffleSplit |
        |---|---|---|
        | Risiko | Model menghafal suara | Model generalisasi ke pembicara baru |
        | Akurasi | Palsu tinggi | Realistis |
        | Deployment | Gagal di real-world | Siap produksi |
        """)

    with st.expander("Temperature Scaling untuk UI"):
        st.markdown("""
        **Masalah:** Model SVM/MLP sering menghasilkan probabilitas yang terlalu percaya diri
        (misalnya 99.9% untuk satu kelas). Distribusi terlalu polarisasi untuk tampilan UI.

        **Solusi:** Temperature Scaling (T=2.5) meng-smoothing distribusi probabilitas
        tanpa mengubah urutan prediksi.

        ```
        Sebelum: A1=0.1%, A2=0.2%, B1=0.5%, B2=99.1%, C1=0.05%, C2=0.05%
        Sesudah: A1=4.2%, A2=6.8%, B1=12.5%, B2=48.3%, C1=15.1%, C2=13.1%
        ```
        """)

    with st.expander("Pipeline Preprocessing"):
        st.markdown("""
        `StandardScaler` dimasukkan di dalam Pipeline (`ImbPipeline`), bukan di luar:

        ```python
        pipeline = ImbPipeline([
            ('preprocessor', ColumnTransformer([('num', StandardScaler(), features)])),
            ('classifier', model)
        ])
        ```

        Scaler hanya di-fit pada data training. Jika di-fit di luar pipeline
        (pada seluruh data), statistik test set bocor ke proses training (data leakage).
        """)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown("### Insights untuk Pembelajaran Bahasa Inggris")
    st.markdown("""
    <div class="info-box">
        <h4 style="color: #6C63FF; margin-top: 0;">1. Penilaian Otomatis yang Holistik</h4>
        <p>Model tidak hanya mengecek <em>apa</em> yang diucapkan (STT), tetapi <em>bagaimana</em>
        cara pengucapannya (prosodi, intonasi, kelancaran). Evaluasi ini lebih komprehensif
        dibanding aplikasi belajar bahasa konvensional.</p>
    </div>

    <div class="info-box">
        <h4 style="color: #48C6EF; margin-top: 0;">2. Adaptasi Level Secara Real-Time</h4>
        <p>Kombinasi Classifier + Regressor memungkinkan logika adaptasi otomatis:</p>
        <ul>
            <li><strong>Upgrade</strong>: Skor tinggi dan prediksi CEFR &ge; level sekarang</li>
            <li><strong>Downgrade</strong>: Skor rendah dan prediksi CEFR &lt; level sekarang</li>
            <li><strong>Maintain</strong>: Skor stabil dan prediksi CEFR = level sekarang</li>
        </ul>
    </div>

    <div class="info-box">
        <h4 style="color: #43E97B; margin-top: 0;">3. Feedback Terstruktur</h4>
        <p>Output model diterjemahkan ke feedback yang actionable: kekuatan (aspek yang sudah baik)
        dan area perbaikan (aspek yang perlu dilatih), sehingga pengalaman belajar lebih personal.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown("### Flowchart Sistem Adaptif")
    st.markdown("""
    ```
    Input Suara       ML Processing          Output
    (19 fitur)  --->  Classifier (MLP)  ---> Level CEFR + Confidence
                      Regressor (RF)    ---> Skor Prosodi (0-100)
                                             Feedback dan Rekomendasi
                                                    |
                                                    v
                                             Logika Adaptasi
                                             - Upgrade
                                             - Maintain
                                             - Downgrade
    ```
    """)

    st.markdown("### Rekomendasi Pengembangan")
    rec1, rec2 = st.columns(2)
    with rec1:
        st.markdown("""
        #### Jangka Pendek
        - Integrasikan dengan real audio (bukan data simulasi)
        - Tambahkan microphone input langsung di web browser
        - Implementasikan Stacking Ensemble untuk akurasi lebih tinggi
        - Tambahkan tracking progres per sesi
        """)
    with rec2:
        st.markdown("""
        #### Jangka Panjang
        - Fine-tune Wav2Vec2 + LoRA dengan lebih banyak data audio
        - Dukung lebih banyak bahasa (multilingual CEFR)
        - Gamifikasi dengan sistem poin dan level
        - Dashboard untuk guru/instruktur
        """)


# ──────────────────────────────────────────────────────────────────────
# PAGE 5: DOKUMENTASI
# ──────────────────────────────────────────────────────────────────────

elif page == "Dokumentasi":
    st.markdown('<h1 class="hero-title">Dokumentasi</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Informasi tentang dataset, metodologi, fitur, dan cara penggunaan</p>', unsafe_allow_html=True)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    tab_d1, tab_d2, tab_d3, tab_d4 = st.tabs([
        "Dataset", "Metodologi", "Fitur", "Cara Penggunaan"
    ])

    with tab_d1:
        st.markdown("### Tentang Dataset")
        st.markdown("""
        #### Sumber Data
        Dataset dibangun dari **Oxford 3000 dan Oxford 5000** — daftar kosakata resmi yang dikategorikan
        berdasarkan level CEFR (A1-C2) oleh Oxford University Press.

        #### Proses Pembuatan
        1. **Parsing PDF** (`parse_oxford_pdf.py`): Mengekstrak 5,401+ kata dari dokumen PDF Oxford
           menggunakan `pdfplumber`, termasuk POS tag dan level CEFR.
        2. **Simulasi Dataset** (`generate_dataset.py`): Membuat 10,800+ sampel dari 150 pembicara
           simulasi di 35 skenario percakapan, dengan distribusi fitur berbeda per level CEFR.
        3. **Pembersihan Data** (`clean_data.py`): Menghapus duplikat, validasi batas logis, menangani
           outlier ekstrem dengan metode IQR (3x IQR), dan normalisasi teks.

        #### Statistik Dataset
        """)

        df = load_dataset()
        if df is not None:
            dsc1, dsc2, dsc3 = st.columns(3)
            with dsc1:
                st.metric("Total Sampel", f"{len(df):,}")
            with dsc2:
                st.metric("Kata Unik", f"{df['word'].nunique():,}")
            with dsc3:
                st.metric("Pembicara", f"{df['speaker_id'].nunique()}")

            st.markdown("#### Distribusi per Level CEFR")
            cefr_dist = df['cefr_level'].value_counts().reindex(CEFR_LEVELS)
            st.dataframe(
                pd.DataFrame({'Level': cefr_dist.index, 'Jumlah Sampel': cefr_dist.values,
                              'Persentase': (cefr_dist.values / len(df) * 100).round(1).astype(str) + '%'}),
                width="stretch", hide_index=True
            )

    with tab_d2:
        st.markdown("### Metodologi")
        st.markdown("""
        #### Pipeline 4 Langkah

        ```
        Oxford PDF --> parse_oxford_pdf.py --> oxford_vocabulary.csv (5,401 kata)
                                                      |
                                                      v
                      generate_dataset.py --> oxford_prosody_dataset.csv (10,802 sampel)
                                                      |
                                                      v
                           clean_data.py --> oxford_prosody_dataset_clean.csv (data bersih)
                                                      |
                                                      v
                               train.py --> best_cefr_classifier.pkl + prosody_regressor.pkl
        ```

        #### Algoritma Klasifikasi
        | Model | Tipe | Kelebihan |
        |---|---|---|
        | **Deep MLP** | Neural Network | Menangkap interaksi fitur kompleks non-linear |
        | Random Forest | Ensemble (Bagging) | Tahan overfitting, interpretable |
        | XGBoost | Ensemble (Boosting) | Cepat, regularisasi bawaan |
        | SVM (RBF) | Kernel Method | Akurat di dimensi tinggi |

        #### Metrik Evaluasi
        - **Klasifikasi**: Macro F1-Score (metrik seleksi utama), Precision, Recall
        - **Regresi**: RMSE (Root Mean Squared Error), R-squared
        - **Validasi**: GroupShuffleSplit (speaker-independent, anti-leakage)
        """)

    with tab_d3:
        st.markdown("### Daftar 19 Fitur Input Model")
        features_info = pd.DataFrame({
            'No': range(1, 20),
            'Nama Fitur': FEATURE_COLUMNS,
            'Kategori': [
                'Prosodi', 'Prosodi', 'Prosodi', 'Prosodi',
                'Prosodi', 'Prosodi', 'Prosodi',
                'Linguistik', 'Linguistik', 'Linguistik',
                'Linguistik', 'Linguistik', 'Linguistik',
                'Linguistik', 'Linguistik',
                'Deep Repr.', 'Deep Repr.', 'Deep Repr.',
                'Konteks'
            ],
            'Deskripsi': [
                'Rata-rata frekuensi dasar (F0) suara',
                'Variasi/deviasi standar pitch',
                'Kemiringan kontur nada bicara',
                'Intensitas energi suara (Root Mean Square)',
                'Total durasi rekaman suara (detik)',
                'Kecepatan bicara (kata per detik)',
                'Jeda sebelum menjawab (milidetik)',
                'Keberagaman kosakata (Type-Token Ratio)',
                'Rasio kesalahan tata bahasa',
                'Skor akurasi fonem pengucapan (0-100)',
                'Rasio jeda diam vs total durasi',
                'Frekuensi kata pengisi (uh, um)',
                'Konsistensi kecepatan bicara',
                'Keyakinan model ASR pada transkripsi',
                'Relevansi semantik jawaban (0-100)',
                'Embedding Whisper dimensi 1',
                'Embedding Whisper dimensi 2',
                'Embedding Whisper dimensi 3',
                'Skor riwayat kemampuan user (0-100)'
            ]
        })
        st.dataframe(features_info, width="stretch", hide_index=True)

    with tab_d4:
        st.markdown("### Cara Penggunaan Aplikasi")
        st.markdown("""
        #### Dashboard EDA
        Jelajahi data secara visual. Gunakan tab untuk berpindah antara distribusi CEFR, distribusi fitur,
        heatmap korelasi, box plot, dan scatter plot interaktif. Fitur yang ditampilkan bisa dipilih.

        #### Model Demo
        1. Pilih profil cepat (Pemula / Menengah / Mahir) atau Manual Input.
        2. Atur parameter menggunakan slider di setiap kategori fitur.
        3. Klik "Jalankan Prediksi" untuk melihat hasil.
        4. Hasil meliputi: level CEFR, skor prosodi, distribusi probabilitas, dan feedback.

        #### Evaluasi Model
        Perbandingan performa 4 model (Deep MLP, Random Forest, XGBoost, SVM).
        Tab berisi confusion matrix, feature importance, dan analisis regresi prosodi.

        #### Interpretasi Hasil
        Penjelasan cara kerja model, keputusan teknis (anti-leakage, temperature scaling),
        dan implikasi untuk pendidikan bahasa Inggris.

        #### Dokumentasi
        Halaman ini — informasi tentang dataset, metodologi, fitur, dan panduan penggunaan.

        ---

        #### Informasi Teknis
        | Komponen | Detail |
        |---|---|
        | Bahasa Pemrograman | Python 3.10+ |
        | Framework Dashboard | Streamlit |
        | ML Libraries | scikit-learn, XGBoost, imbalanced-learn |
        | Visualisasi | Plotly, Matplotlib, Seaborn |
        | Deployment | Streamlit Community Cloud |
        """)
