"""
Streamlit UI for Resume Screening System.
Provides a user-friendly interface for resume upload, classification, and job matching.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Set page configuration
st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# TRENDING 2024/2025 UI DESIGN - Dark Mode, Glassmorphism, Neon Accents
modern_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', 'Space Grotesk', sans-serif;
    }
    
    /* DARK THEME BACKGROUND */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        background-attachment: fixed;
    }
    
    /* Main container - Glassmorphism effect */
    .main .block-container {
        background: rgba(20, 20, 35, 0.7);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 24px;
        padding: 2.5rem;
        margin: 1.5rem auto;
        max-width: 1200px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
    }
    
    /* Animated gradient text for headers */
    .gradient-text {
        background: linear-gradient(135deg, #00f5ff 0%, #b829dd 50%, #ff006e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
    }
    
    /* NEON HERO SECTION */
    .hero-container {
        background: linear-gradient(135deg, rgba(0, 245, 255, 0.1) 0%, rgba(184, 41, 221, 0.1) 50%, rgba(255, 0, 110, 0.1) 100%);
        border: 1px solid rgba(0, 245, 255, 0.3);
        padding: 4rem 2rem;
        border-radius: 24px;
        text-align: center;
        color: white;
        margin-bottom: 2.5rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 0 60px rgba(0, 245, 255, 0.15),
                    inset 0 0 60px rgba(0, 245, 255, 0.05);
    }
    
    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(0, 245, 255, 0.1) 0%, transparent 70%);
        animation: rotate 20s linear infinite;
    }
    
    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #00f5ff 0%, #b829dd 50%, #ff006e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        position: relative;
        z-index: 1;
        letter-spacing: -1px;
    }
    
    .hero-subtitle {
        font-size: 1.3rem;
        color: rgba(255, 255, 255, 0.7);
        font-weight: 300;
        position: relative;
        z-index: 1;
    }
    
    /* GLASSMORPHISM CARDS */
    .feature-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2rem;
        border-radius: 20px;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        height: 100%;
        position: relative;
        overflow: hidden;
    }
    
    .feature-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        transition: left 0.5s;
    }
    
    .feature-card:hover::before {
        left: 100%;
    }
    
    .feature-card:hover {
        transform: translateY(-8px) scale(1.02);
        border-color: rgba(0, 245, 255, 0.5);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4),
                    0 0 40px rgba(0, 245, 255, 0.1);
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1.5rem;
        display: block;
        filter: drop-shadow(0 0 20px rgba(0, 245, 255, 0.5));
    }
    
    .feature-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #fff;
        margin-bottom: 0.75rem;
        font-family: 'Space Grotesk', sans-serif;
    }
    
    .feature-desc {
        font-size: 0.95rem;
        color: rgba(255, 255, 255, 0.6);
        line-height: 1.6;
    }
    
    /* NEON RESULT CARDS */
    .result-card {
        background: rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 245, 255, 0.3);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        margin: 1rem 0;
        position: relative;
        overflow: hidden;
        box-shadow: 0 0 30px rgba(0, 245, 255, 0.1);
    }
    
    .result-card::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, #00f5ff, #b829dd, #ff006e);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0%, 100% { opacity: 0.5; }
        50% { opacity: 1; }
    }
    
    .result-title {
        font-size: 0.85rem;
        color: rgba(255, 255, 255, 0.6);
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 0.75rem;
        font-weight: 500;
    }
    
    .result-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00f5ff 0%, #b829dd 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* SCORE CONTAINER */
    .score-container {
        background: rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 2.5rem;
        margin: 1.5rem 0;
    }
    
    .score-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.5rem;
    }
    
    .score-label {
        font-size: 1.1rem;
        font-weight: 500;
        color: rgba(255, 255, 255, 0.8);
    }
    
    .score-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00f5ff 0%, #ff006e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* NEON PROGRESS BAR */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #00f5ff 0%, #b829dd 50%, #ff006e 100%) !important;
        border-radius: 10px;
        box-shadow: 0 0 20px rgba(0, 245, 255, 0.5);
    }
    
    /* GLOWING BUTTONS */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #00f5ff 0%, #b829dd 100%);
        color: #0a0a0a;
        padding: 14px 28px;
        border-radius: 16px;
        border: none;
        font-weight: 700;
        font-size: 1.1rem;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
        box-shadow: 0 0 30px rgba(0, 245, 255, 0.3);
    }
    
    .stButton>button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.4), transparent);
        transition: left 0.5s;
    }
    
    .stButton>button:hover::before {
        left: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 40px rgba(0, 245, 255, 0.5);
    }
    
    /* DARK SIDEBAR WITH NEON ACCENTS */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0a0f 0%, #1a1a2e 100%);
        border-right: 1px solid rgba(0, 245, 255, 0.2);
    }
    
    [data-testid="stSidebar"] .css-1lcbmhc {
        color: white;
    }
    
    /* NEON RADIO BUTTONS */
    .stRadio > div {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 0.75rem;
        border: 1px solid rgba(0, 245, 255, 0.2);
    }
    
    .stRadio label {
        color: rgba(255, 255, 255, 0.8) !important;
        font-weight: 500;
        padding: 0.75rem 1rem;
        border-radius: 12px;
        transition: all 0.3s ease;
    }
    
    .stRadio label:hover {
        background: rgba(0, 245, 255, 0.1);
    }
    
    /* NEON UPLOAD AREA */
    .stFileUploader {
        background: rgba(0, 0, 0, 0.3);
        border: 2px dashed rgba(0, 245, 255, 0.5);
        border-radius: 20px;
        padding: 3rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: #00f5ff;
        background: rgba(0, 245, 255, 0.05);
        box-shadow: 0 0 40px rgba(0, 245, 255, 0.1);
    }
    
    /* RANK CARDS WITH NEON BORDERS */
    .rank-card {
        background: rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 0.75rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        display: flex;
        align-items: center;
        gap: 1.5rem;
        transition: all 0.3s ease;
    }
    
    .rank-card:hover {
        transform: translateX(10px);
        border-color: rgba(0, 245, 255, 0.5);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    }
    
    .rank-number {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00f5ff 0%, #b829dd 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        min-width: 60px;
        text-align: center;
    }
    
    .rank-content {
        flex: 1;
    }
    
    .rank-name {
        font-weight: 600;
        color: #fff;
        font-size: 1.2rem;
        margin-bottom: 0.25rem;
    }
    
    .rank-details {
        color: rgba(255, 255, 255, 0.6);
        font-size: 0.95rem;
    }
    
    .rank-score {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00f5ff 0%, #ff006e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* NEON BADGES */
    .badge {
        display: inline-block;
        padding: 0.5rem 1.25rem;
        border-radius: 30px;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        border: 1px solid;
    }
    
    .badge-excellent {
        background: rgba(16, 185, 129, 0.2);
        border-color: #10b981;
        color: #10b981;
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.3);
    }
    
    .badge-good {
        background: rgba(59, 130, 246, 0.2);
        border-color: #3b82f6;
        color: #3b82f6;
        box-shadow: 0 0 20px rgba(59, 130, 246, 0.3);
    }
    
    .badge-average {
        background: rgba(245, 158, 11, 0.2);
        border-color: #f59e0b;
        color: #f59e0b;
        box-shadow: 0 0 20px rgba(245, 158, 11, 0.3);
    }
    
    .badge-poor {
        background: rgba(239, 68, 68, 0.2);
        border-color: #ef4444;
        color: #ef4444;
        box-shadow: 0 0 20px rgba(239, 68, 68, 0.3);
    }
    
    /* DARK MODE TEXT */
    h1, h2, h3, h4, h5, h6 {
        color: #fff;
        font-family: 'Space Grotesk', sans-serif;
    }
    
    h2 {
        font-weight: 600;
        margin-top: 2.5rem;
        margin-bottom: 1.5rem;
        font-size: 1.8rem;
    }
    
    p, span, label {
        color: rgba(255, 255, 255, 0.8);
    }
    
    /* NEON TEXT AREAS */
    .stTextArea textarea {
        background: rgba(0, 0, 0, 0.3);
        border-radius: 16px;
        border: 1px solid rgba(0, 245, 255, 0.3);
        color: white;
        font-family: 'Inter', sans-serif;
        padding: 1rem;
    }
    
    .stTextArea textarea:focus {
        border-color: #00f5ff;
        box-shadow: 0 0 20px rgba(0, 245, 255, 0.2);
    }
    
    /* ANIMATIONS */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    @keyframes glow {
        0%, 100% { box-shadow: 0 0 20px rgba(0, 245, 255, 0.5); }
        50% { box-shadow: 0 0 40px rgba(0, 245, 255, 0.8); }
    }
    
    .animate-in {
        animation: fadeInUp 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .glow-effect {
        animation: glow 2s ease-in-out infinite;
    }
    
    /* NEON SCROLLBAR */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.3);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #00f5ff 0%, #b829dd 100%);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #00d4e0 0%, #9a20b0 100%);
    }
    
    /* SUCCESS MESSAGE */
    .success-message {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.2) 100%);
        border: 1px solid #10b981;
        border-radius: 16px;
        padding: 1.25rem;
        color: #10b981;
        text-align: center;
        font-weight: 600;
        box-shadow: 0 0 30px rgba(16, 185, 129, 0.2);
    }
    
    /* NEON PILL TAGS */
    .pill-tag {
        background: rgba(0, 245, 255, 0.1);
        border: 1px solid rgba(0, 245, 255, 0.5);
        color: #00f5ff;
        padding: 0.4rem 1rem;
        border-radius: 30px;
        font-size: 0.85rem;
        font-weight: 500;
        display: inline-block;
        margin: 0.25rem;
        transition: all 0.3s ease;
    }
    
    .pill-tag:hover {
        background: rgba(0, 245, 255, 0.2);
        box-shadow: 0 0 20px rgba(0, 245, 255, 0.3);
        transform: scale(1.05);
    }
    
    /* METRIC CARDS GRID */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.5rem;
        margin: 1.5rem 0;
    }
    
    /* DATA TABLE STYLING */
    .stDataFrame {
        background: rgba(0, 0, 0, 0.3);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* CHART CONTAINER */
    .stChart {
        background: rgba(0, 0, 0, 0.3);
        border-radius: 20px;
        padding: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* EXPANDER STYLING */
    .streamlit-expander {
        background: rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        margin: 1rem 0;
    }
    
    .streamlit-expanderHeader {
        color: white !important;
        font-weight: 600;
    }
    
    /* FILE UPLOADER LABEL */
    .stFileUploader > label {
        color: rgba(255, 255, 255, 0.9) !important;
        font-weight: 500;
    }
    
    /* ALERTS */
    .stAlert {
        border-radius: 16px;
        border: 1px solid;
        background: rgba(0, 0, 0, 0.3);
    }
    
    .stAlert[data-baseweb="notification"][kind="error"] {
        border-color: #ef4444;
        color: #ef4444;
    }
    
    .stAlert[data-baseweb="notification"][kind="warning"] {
        border-color: #f59e0b;
        color: #f59e0b;
    }
    
    .stAlert[data-baseweb="notification"][kind="info"] {
        border-color: #3b82f6;
        color: #3b82f6;
    }
</style>
"""

st.markdown(modern_css, unsafe_allow_html=True)

# Animated Hero Section with Neon Effects
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🎯 AI Resume Screener</div>
    <div class="hero-subtitle">Next-Gen Candidate Matching with Neural Intelligence</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<p style="text-align: center; color: rgba(255,255,255,0.7); font-size: 1.2rem; margin-bottom: 2.5rem; line-height: 1.8;">
    Experience the future of recruitment with <span class="gradient-text">AI-powered analysis</span>. 
    <br>Instant classification, intelligent matching, and precision ranking.
</p>
""", unsafe_allow_html=True)

# Sidebar Navigation with custom styling
st.sidebar.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <h2 style="color: white; font-size: 1.5rem; margin-bottom: 0.5rem;">🎯 Menu</h2>
    <p style="color: rgba(255,255,255,0.7); font-size: 0.9rem;">Select a mode to begin</p>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "",
    ["🏠 Home", "📤 Single Analysis", "📊 Batch Ranking", "📈 Performance"],
    label_visibility="collapsed"
)

# Initialize session state
if 'predictor' not in st.session_state:
    st.session_state.predictor = None

if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# Function to initialize predictor
def initialize_predictor():
    """Initialize the resume predictor."""
    try:
        from prediction import ResumePredictor
        import train_model
        
        model_dir = 'models'
        
        # Check if model exists, if not train it (for cloud deployment)
        if not os.path.exists(model_dir) or not os.path.exists(f'{model_dir}/classifier.joblib'):
            with st.spinner("🤖 First run - training model... This may take a minute..."):
                train_model.train_model(
                    dataset_path=train_model.download_sample_dataset(),
                    algorithm='logistic_regression',
                    vectorizer_method='tfidf'
                )
        
        st.session_state.predictor = ResumePredictor(model_dir)
        st.session_state.model_loaded = st.session_state.predictor.load_model()
        return st.session_state.model_loaded
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return False


# Home Page
if page == "🏠 Home":
    st.markdown('<div class="animate-in">', unsafe_allow_html=True)
    
    # Feature cards grid
    st.subheader("✨ Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">🤖</span>
            <div class="feature-title">AI Classification</div>
            <div class="feature-desc">Automatically categorize resumes into job roles using advanced ML algorithms</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">💼</span>
            <div class="feature-title">Smart Matching</div>
            <div class="feature-desc">Compare resumes against job descriptions with intelligent similarity scoring</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">🏆</span>
            <div class="feature-title">Batch Ranking</div>
            <div class="feature-desc">Rank multiple candidates instantly and identify top talent efficiently</div>
        </div>
        """, unsafe_allow_html=True)
    
    # How to use
    st.subheader("🚀 Quick Start")
    
    steps_col1, steps_col2, steps_col3 = st.columns(3)
    
    with steps_col1:
        st.markdown("""
        <div class="feature-card" style="text-align: center;">
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; width: 50px; height: 50px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 1rem; font-size: 1.5rem; font-weight: 700;">1</div>
            <div class="feature-title">Upload Resume</div>
            <div class="feature-desc">Select PDF or DOCX file</div>
        </div>
        """, unsafe_allow_html=True)
    
    with steps_col2:
        st.markdown("""
        <div class="feature-card" style="text-align: center;">
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; width: 50px; height: 50px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 1rem; font-size: 1.5rem; font-weight: 700;">2</div>
            <div class="feature-title">AI Analysis</div>
            <div class="feature-desc">Get instant classification</div>
        </div>
        """, unsafe_allow_html=True)
    
    with steps_col3:
        st.markdown("""
        <div class="feature-card" style="text-align: center;">
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; width: 50px; height: 50px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 1rem; font-size: 1.5rem; font-weight: 700;">3</div>
            <div class="feature-title">View Results</div>
            <div class="feature-desc">See scores and insights</div>
        </div>
        """, unsafe_allow_html=True)
    
    # System status
    st.subheader("⚡ System Status")
    
    status_col1, status_col2, status_col3 = st.columns(3)
    
    with status_col1:
        if initialize_predictor():
            st.markdown("""
            <div class="result-card" style="text-align: center;">
                <div class="result-title">Model Status</div>
                <div style="font-size: 2rem;">✅</div>
                <div class="result-value" style="font-size: 1.2rem;">Ready</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-card" style="text-align: center; background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);">
                <div class="result-title">Model Status</div>
                <div style="font-size: 2rem;">⚠️</div>
                <div class="result-value" style="font-size: 1.2rem;">Not Loaded</div>
            </div>
            """, unsafe_allow_html=True)
    
    with status_col2:
        st.markdown("""
        <div class="result-card" style="text-align: center;">
            <div class="result-title">Supported Formats</div>
            <div style="font-size: 1.5rem; margin: 0.5rem 0;">📄 📃</div>
            <div class="result-value" style="font-size: 1.2rem;">PDF & DOCX</div>
        </div>
        """, unsafe_allow_html=True)
    
    with status_col3:
        st.markdown("""
        <div class="result-card" style="text-align: center;">
            <div class="result-title">Categories</div>
            <div style="font-size: 1.5rem; margin: 0.5rem 0;">📊</div>
            <div class="result-value" style="font-size: 1.2rem;">6 Job Roles</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)


# Single Resume Analysis Page
elif page == "📤 Single Analysis":
    st.markdown('<div class="animate-in">', unsafe_allow_html=True)
    
    st.markdown("""
    <h2 style="display: flex; align-items: center; gap: 0.5rem;">
        <span style="font-size: 2rem;">📤</span> Single Resume Analysis
    </h2>
    """, unsafe_allow_html=True)
    
    # Initialize model
    if not st.session_state.model_loaded:
        if not initialize_predictor():
            st.stop()
    
    # Upload section with modern styling
    st.markdown("""
    <div class="feature-card" style="margin-bottom: 1.5rem;">
        <h4 style="margin-top: 0; color: #667eea;">📁 Upload Resume</h4>
        <p style="color: #666; margin-bottom: 0;">Supported formats: PDF, DOCX</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "",
        type=['pdf', 'docx'],
        label_visibility="collapsed"
    )
    
    # Job description input
    job_description = st.text_area(
        "Job Description (Optional)",
        height=150,
        placeholder="Paste job description here to calculate match score...",
        help="Enter a job description to match against the resume"
    )
    
    if uploaded_file is not None:
        # Determine file type
        file_type = uploaded_file.name.split('.')[-1].lower()
        
        # Process button
        if st.button("🔍 Analyze Resume", type="primary"):
            with st.spinner("Processing resume..."):
                try:
                    # Predict category
                    result = st.session_state.predictor.predict_category(
                        resume_path=uploaded_file,
                        file_type=file_type
                    )
                    
                    if 'error' in result:
                        st.error(f"❌ {result['error']}")
                    else:
                        # Success message with animation
                        st.markdown("""
                        <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 1rem; border-radius: 12px; text-align: center; margin-bottom: 1.5rem;">
                            <span style="font-size: 1.5rem;">✨</span> Analysis Complete!
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Results grid with modern cards
                        st.subheader("📊 Analysis Results")
                        
                        res_col1, res_col2, res_col3 = st.columns(3)
                        
                        with res_col1:
                            st.markdown(f"""
                            <div class="result-card" style="text-align: center;">
                                <div class="result-title">Predicted Role</div>
                                <div style="font-size: 2.5rem; margin: 0.5rem 0;">💼</div>
                                <div class="result-value" style="font-size: 1.3rem; text-transform: lowercase;">{result['predicted_category']}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with res_col2:
                            confidence = result['confidence'] * 100
                            st.markdown(f"""
                            <div class="result-card" style="text-align: center;">
                                <div class="result-title">Confidence</div>
                                <div style="font-size: 2.5rem; margin: 0.5rem 0;">🎯</div>
                                <div class="result-value" style="font-size: 1.3rem;">{confidence:.1f}%</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with res_col3:
                            if job_description.strip():
                                match = st.session_state.predictor.calculate_match_score(
                                    result['processed_text'],
                                    job_description
                                )
                                match_pct = match['match_score'] * 100
                                st.markdown(f"""
                                <div class="result-card" style="text-align: center;">
                                    <div class="result-title">Match Score</div>
                                    <div style="font-size: 2.5rem; margin: 0.5rem 0;">🤝</div>
                                    <div class="result-value" style="font-size: 1.3rem;">{match_pct:.1f}%</div>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div class="result-card" style="text-align: center; background: linear-gradient(135deg, #6b7280 0%, #4b5563 100%);">
                                    <div class="result-title">Match Score</div>
                                    <div style="font-size: 2.5rem; margin: 0.5rem 0;">🤝</div>
                                    <div class="result-value" style="font-size: 1.3rem;">N/A</div>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Confidence visualization
                        st.markdown("""
                        <div class="score-container">
                            <div class="score-header">
                                <span class="score-label">🎯 Classification Confidence</span>
                                <span class="score-value">{:.1f}%</span>
                            </div>
                        </div>
                        """.format(confidence), unsafe_allow_html=True)
                        st.progress(confidence / 100)
                        
                        # Category probabilities with modern chart
                        with st.expander("📈 View All Category Probabilities"):
                            probs = result['all_probabilities']
                            prob_df = pd.DataFrame({
                                'Category': list(probs.keys()),
                                'Probability': [p * 100 for p in probs.values()]
                            }).sort_values('Probability', ascending=False)
                            
                            # Styled bar chart
                            st.bar_chart(prob_df.set_index('Category'), use_container_width=True)
                            
                            # Show as styled table
                            st.markdown("<div style='margin-top: 1rem;'>", unsafe_allow_html=True)
                            for _, row in prob_df.iterrows():
                                pct = row['Probability']
                                color = "#10b981" if pct > 40 else "#3b82f6" if pct > 20 else "#6b7280"
                                st.markdown(f"""
                                <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.5rem; margin: 0.25rem 0; background: #f9fafb; border-radius: 8px;">
                                    <span style="font-weight: 500;">{row['Category']}</span>
                                    <span style="color: {color}; font-weight: 600;">{pct:.1f}%</span>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Job match results with modern design
                        if job_description.strip():
                            st.markdown("""
                            <h3 style="margin-top: 2rem; display: flex; align-items: center; gap: 0.5rem;">
                                <span>💼</span> Job Match Analysis
                            </h3>
                            """, unsafe_allow_html=True)
                            
                            match = st.session_state.predictor.calculate_match_score(
                                result['processed_text'],
                                job_description
                            )
                            
                            match_pct = match['match_score'] * 100
                            
                            # Determine badge style
                            if match_pct >= 80:
                                badge_class = "badge-excellent"
                                badge_text = "EXCELLENT MATCH"
                                emoji = "🌟"
                            elif match_pct >= 60:
                                badge_class = "badge-good"
                                badge_text = "GOOD MATCH"
                                emoji = "✅"
                            elif match_pct >= 40:
                                badge_class = "badge-average"
                                badge_text = "AVERAGE MATCH"
                                emoji = "🟡"
                            else:
                                badge_class = "badge-poor"
                                badge_text = "LOW MATCH"
                                emoji = "⚠️"
                            
                            # Match score card
                            st.markdown(f"""
                            <div class="score-container" style="text-align: center;">
                                <div style="margin-bottom: 1rem;">
                                    <span class="badge {badge_class}">{badge_text}</span>
                                </div>
                                <div class="score-value" style="font-size: 4rem; margin: 1rem 0;">{emoji} {match_pct:.1f}%</div>
                                <p style="color: #666; font-size: 1rem;">{match['interpretation']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Detailed scores
                            detail_col1, detail_col2 = st.columns(2)
                            with detail_col1:
                                st.markdown(f"""
                                <div class="feature-card" style="text-align: center;">
                                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">📄</div>
                                    <div style="color: #666; font-size: 0.9rem;">Text Similarity</div>
                                    <div style="font-size: 1.5rem; font-weight: 700; color: #667eea;">{match['similarity_score']*100:.1f}%</div>
                                </div>
                                """, unsafe_allow_html=True)
                            with detail_col2:
                                st.markdown(f"""
                                <div class="feature-card" style="text-align: center;">
                                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">🎯</div>
                                    <div style="color: #666; font-size: 0.9rem;">Category Match</div>
                                    <div style="font-size: 1.5rem; font-weight: 700; color: #667eea;">{match['category_match']*100:.1f}%</div>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Expandable sections with modern styling
                        st.markdown("""
                        <h3 style="margin-top: 2rem;">📄 Resume Content</h3>
                        """, unsafe_allow_html=True)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            with st.expander("� View Extracted Text"):
                                st.text_area(
                                    "",
                                    result['raw_text'][:2000] + ("..." if len(result['raw_text']) > 2000 else ""),
                                    height=200,
                                    disabled=True,
                                    label_visibility="collapsed"
                                )
                        with col2:
                            with st.expander("🔧 Processed Text"):
                                st.text_area(
                                    "",
                                    result['processed_text'][:2000] + ("..." if len(result['processed_text']) > 2000 else ""),
                                    height=200,
                                    disabled=True,
                                    label_visibility="collapsed"
                                )
                        
                        # Keywords with pills
                        with st.expander("🔑 Extracted Keywords"):
                            keywords = st.session_state.predictor.extract_keywords(
                                result['raw_text']
                            )
                            if keywords:
                                st.markdown("<div style='display: flex; flex-wrap: wrap; gap: 0.5rem;'>", unsafe_allow_html=True)
                                for kw in keywords[:15]:
                                    # Capitalize first letter for formal display
                                    formal_keyword = kw['keyword'].capitalize()
                                    st.markdown(f"""
                                    <span style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                                 color: white; padding: 0.4rem 1rem; border-radius: 20px; 
                                                 font-size: 0.85rem; font-weight: 500;">
                                        {formal_keyword}
                                    </span>
                                    """, unsafe_allow_html=True)
                                st.markdown("</div>", unsafe_allow_html=True)
                            else:
                                st.info("No keywords extracted")
                
                except Exception as e:
                    st.error(f"❌ Error processing resume: {e}")


# Batch Processing Page
elif page == "📊 Batch Ranking":
    st.markdown('<div class="animate-in">', unsafe_allow_html=True)
    
    st.markdown("""
    <h2 style="display: flex; align-items: center; gap: 0.5rem;">
        <span style="font-size: 2rem;">📊</span> Batch Resume Ranking
    </h2>
    """, unsafe_allow_html=True)
    
    # Initialize model
    if not st.session_state.model_loaded:
        if not initialize_predictor():
            st.stop()
    
    # Multiple file upload
    uploaded_files = st.file_uploader(
        "Upload Multiple Resumes (PDF or DOCX)",
        type=['pdf', 'docx'],
        accept_multiple_files=True,
        help="Upload multiple resumes to rank them"
    )
    
    # Job description (required for ranking)
    job_description = st.text_area(
        "Job Description *",
        height=150,
        placeholder="Enter job description to rank candidates...",
        help="Job description is required for ranking candidates"
    )
    
    if uploaded_files and job_description.strip():
        if st.button("🔍 Rank Candidates", type="primary"):
            with st.spinner("Processing resumes..."):
                try:
                    # Prepare resumes data
                    resumes_data = []
                    for uploaded_file in uploaded_files:
                        file_type = uploaded_file.name.split('.')[-1].lower()
                        resumes_data.append({
                            'name': uploaded_file.name,
                            'path': uploaded_file,
                            'file_type': file_type
                        })
                    
                    # Rank resumes
                    ranked_results = st.session_state.predictor.rank_resumes(
                        resumes_data,
                        job_description
                    )
                    
                    if not ranked_results:
                        st.warning("⚠️ No resumes could be processed")
                    else:
                        st.subheader("🏆 Ranked Candidates")
                        
                        # Display results with modern rank cards
                        for result in ranked_results:
                            rank = result['rank']
                            name = result['name']
                            score = result['match_score'] * 100
                            category = result['predicted_category'].lower()
                            interpretation = result['interpretation']
                            
                            # Medal for top 3
                            medal = ""
                            rank_emoji = ""
                            if rank == 1:
                                medal = "🥇"
                                rank_border = "border-left: 4px solid #FFD700;"
                            elif rank == 2:
                                medal = "🥈"
                                rank_border = "border-left: 4px solid #C0C0C0;"
                            elif rank == 3:
                                medal = "🥉"
                                rank_border = "border-left: 4px solid #CD7F32;"
                            else:
                                rank_border = "border-left: 4px solid #667eea;"
                            
                            # Badge based on score
                            if score >= 80:
                                badge_class = "badge-excellent"
                                badge_text = "Excellent"
                            elif score >= 60:
                                badge_class = "badge-good"
                                badge_text = "Good"
                            elif score >= 40:
                                badge_class = "badge-average"
                                badge_text = "Average"
                            else:
                                badge_class = "badge-poor"
                                badge_text = "Low"
                            
                            st.markdown(f"""
                            <div class="rank-card" style="{rank_border}">
                                <div class="rank-number">{medal or f'#{rank}'}</div>
                                <div class="rank-content">
                                    <div class="rank-name">{name}</div>
                                    <div class="rank-details">
                                        💼 {category} • <span class="badge {badge_class}">{badge_text}</span>
                                    </div>
                                </div>
                                <div class="rank-score">{score:.1f}%</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Summary table
                        st.subheader("📋 Summary Table")
                        summary_df = pd.DataFrame([
                            {
                                'Rank': r['rank'],
                                'Name': r['name'],
                                'Match Score': f"{r['match_score']*100:.1f}%",
                                'Category': r['predicted_category'].lower(),
                                'Similarity': f"{r['similarity_score']*100:.1f}%"
                            }
                            for r in ranked_results
                        ])
                        st.dataframe(summary_df, use_container_width=True)
                        
                        # Download results
                        csv = summary_df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Results (CSV)",
                            csv,
                            "resume_ranking_results.csv",
                            "text/csv"
                        )
                
                except Exception as e:
                    st.error(f"❌ Error processing resumes: {e}")
    elif uploaded_files and not job_description.strip():
        st.warning("⚠️ Please enter a job description to rank candidates")
    
    st.markdown('</div>', unsafe_allow_html=True)


# Model Performance Page
elif page == "📈 Performance":
    st.markdown('<div class="animate-in">', unsafe_allow_html=True)
    
    st.markdown("""
    <h2 style="display: flex; align-items: center; gap: 0.5rem;">
        <span style="font-size: 2rem;">📈</span> Model Performance
    </h2>
    """, unsafe_allow_html=True)
    
    model_dir = 'models'
    
    if not os.path.exists(model_dir):
        st.error("❌ No trained model found. Please train the model first.")
        st.code("python train_model.py", language="bash")
        st.stop()
    
    # Load metadata
    try:
        import joblib
        metadata = joblib.load(f'{model_dir}/metadata.joblib')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🤖 Model Information")
            st.write(f"**Algorithm:** {metadata['algorithm'].replace('_', ' ').lower()}")
            st.write(f"**Vectorizer:** {metadata['vectorizer_method'].upper()}")
            st.write(f"**Number of Categories:** {len(metadata['label_mapping'])}")
        
        with col2:
            st.subheader("📊 Categories")
            categories = list(metadata['label_mapping'].keys())
            for cat in categories:
                st.write(f"- {cat}")
        
        # Check for evaluation results
        if os.path.exists('evaluation_results.joblib'):
            results = joblib.load('evaluation_results.joblib')
            
            st.subheader("📈 Evaluation Metrics")
            
            metrics = results['metrics']
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
            with col2:
                st.metric("Precision", f"{metrics['precision']:.4f}")
            with col3:
                st.metric("Recall", f"{metrics['recall']:.4f}")
            with col4:
                st.metric("F1-Score", f"{metrics['f1_score']:.4f}")
            
            # Cross-validation
            if 'cross_validation' in results:
                cv = results['cross_validation']
                st.subheader("🔁 Cross-Validation Results")
                st.write(f"**Mean Accuracy:** {cv['mean_accuracy']:.4f}")
                st.write(f"**Std Deviation:** {cv['std_accuracy']:.4f}")
                
                # CV scores chart
                cv_df = pd.DataFrame({
                    'Fold': range(1, len(cv['scores']) + 1),
                    'Accuracy': cv['scores']
                })
                st.bar_chart(cv_df.set_index('Fold'))
            
            # Classification report
            with st.expander("📋 View Classification Report"):
                report = metrics['classification_report']
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df, use_container_width=True)
        else:
            st.info("ℹ️ No evaluation results found. Run training with evaluation to see metrics.")
    
    except Exception as e:
        st.error(f"❌ Error loading model metadata: {e}")
    
    st.markdown('</div>', unsafe_allow_html=True)


# Footer with modern styling
st.sidebar.markdown("""
<div style="position: fixed; bottom: 0; left: 0; right: 0; 
            background: linear-gradient(180deg, #1a1a2e 0%, #0f3460 100%);
            padding: 1rem; text-align: center; border-top: 1px solid rgba(255,255,255,0.1);">
    <p style="color: rgba(255,255,255,0.8); margin: 0; font-size: 0.9rem;">
        🎯 AI Resume Screener v2.0
    </p>
    <p style="color: rgba(255,255,255,0.5); margin: 0.25rem 0 0 0; font-size: 0.75rem;">
        Built with ❤️ using Streamlit & Scikit-learn
    </p>
</div>
""", unsafe_allow_html=True)
