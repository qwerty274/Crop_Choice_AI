"""
Crop Choice Intelligence - Streamlit Web Application
Designed for Streamlit Cloud Deployment
"""

import sys
import os
import streamlit as st
import pandas as pd
import numpy as np

# Ensure src directory is in module path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from predictor import CropPredictor
from train_model import train_and_evaluate

# Page Configuration
st.set_page_config(
    page_title="Crop Choice Intelligence AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Claymorphic Environmental CSS Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700;800&family=Plus+Jakarta+Sans:wght@400;600;700&display=swap');
    
    .stApp {
        background: radial-gradient(circle at 10% 10%, #15271d 0%, #0c140f 50%, #060a08 100%);
        color: #f0fdf4;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .hero-box {
        background: #14211a;
        box-shadow: 10px 10px 20px #070b09, -8px -8px 20px #21352a;
        border-radius: 24px;
        padding: 2rem;
        text-align: center;
        margin-bottom: 2rem;
        border: 1px solid rgba(34, 197, 94, 0.2);
    }
    
    .hero-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff 0%, #22c55e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .clay-card {
        background: #14211a;
        box-shadow: 10px 10px 20px #070b09, -8px -8px 20px #21352a;
        border-radius: 20px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    
    .crop-result {
        font-family: 'Outfit', sans-serif;
        font-size: 2.5rem;
        font-weight: 900;
        color: #22c55e;
        text-transform: capitalize;
    }
    
    .pill-good { background: #064e3b; color: #6ee7b7; padding: 0.3rem 0.8rem; border-radius: 12px; font-weight: 700; font-size: 0.85rem; }
    .pill-warn { background: #78350f; color: #fde68a; padding: 0.3rem 0.8rem; border-radius: 12px; font-weight: 700; font-size: 0.85rem; }
    .pill-bad  { background: #7f1d1d; color: #fca5a5; padding: 0.3rem 0.8rem; border-radius: 12px; font-weight: 700; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

# Initialize Predictor Engine
@st.cache_resource
def get_predictor():
    predictor = CropPredictor()
    if not predictor.loaded:
        train_and_evaluate()
        predictor._load_model()
    return predictor

predictor = get_predictor()

# App Header
st.markdown("""
<div class="hero-box">
    <div style="font-size: 3rem;">🌱</div>
    <div class="hero-title">Crop Choice Intelligence System</div>
    <p style="color: #a7f3d0; margin-top: 0.5rem;">
        Precision Agronomic Machine Learning Engine powered by Kaggle Dataset & Soil Health Advisory
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar Preset & Navigation
st.sidebar.title("🎛️ Navigation & Presets")
menu = st.sidebar.radio("Select Module", ["🌾 Crop Predictor", "📊 ML Benchmarks", "🌦️ Climate Simulator", "📚 Crop Library"])

st.sidebar.markdown("---")
st.sidebar.subheader("🌱 Quick Soil Presets")

preset = st.sidebar.selectbox("Load Preset Parameters", [
    "Custom Manual Sliders",
    "🌾 Alluvial Rice Belt",
    "☁️ Semi-Arid Cotton",
    "☕ Coffee Highlands",
    "🍎 Fruit Orchard Loam",
    "🫐 Acidic Soil"
])

# Default values
default_vals = {
    "Custom Manual Sliders": {"N": 90, "P": 48, "K": 40, "temp": 24.0, "humidity": 82.0, "ph": 6.5, "rainfall": 240.0},
    "🌾 Alluvial Rice Belt": {"N": 90, "P": 48, "K": 40, "temp": 24.0, "humidity": 82.0, "ph": 6.5, "rainfall": 240.0},
    "☁️ Semi-Arid Cotton": {"N": 120, "P": 45, "K": 20, "temp": 25.0, "humidity": 80.0, "ph": 7.2, "rainfall": 75.0},
    "☕ Coffee Highlands": {"N": 100, "P": 28, "K": 30, "temp": 25.0, "humidity": 58.0, "ph": 6.6, "rainfall": 160.0},
    "🍎 Fruit Orchard Loam": {"N": 30, "P": 130, "K": 200, "temp": 18.0, "humidity": 82.0, "ph": 6.0, "rainfall": 70.0},
    "🫐 Acidic Soil": {"N": 25, "P": 68, "K": 20, "temp": 20.0, "humidity": 22.0, "ph": 5.6, "rainfall": 110.0}
}[preset]

if menu == "🌾 Crop Predictor":
    col1, col2 = st.columns([1.1, 1.2])

    with col1:
        st.markdown("### 🧪 Soil & Climate Inputs")
        
        N = st.slider("Nitrogen (N) Content (mg/kg)", 0, 140, int(default_vals["N"]))
        P = st.slider("Phosphorus (P) Content (mg/kg)", 5, 145, int(default_vals["P"]))
        K = st.slider("Potassium (K) Content (mg/kg)", 5, 205, int(default_vals["K"]))
        temp = st.slider("Temperature (°C)", 8.0, 45.0, float(default_vals["temp"]), step=0.5)
        humidity = st.slider("Relative Humidity (%)", 14.0, 100.0, float(default_vals["humidity"]))
        ph = st.slider("Soil pH Level", 3.5, 10.0, float(default_vals["ph"]), step=0.1)
        rainfall = st.slider("Rainfall (mm)", 20.0, 300.0, float(default_vals["rainfall"]))
        
        inputs = {'N': N, 'P': P, 'K': K, 'temperature': temp, 'humidity': humidity, 'ph': ph, 'rainfall': rainfall}

    with col2:
        st.markdown("### 🏆 AI Recommendation Engine")
        
        result = predictor.predict(inputs, top_k=3)
        primary = result['primary_recommendation']
        top_recs = result['top_recommendations']
        soil = result['soil_health_analysis']
        
        st.markdown(f"""
        <div class="clay-card" style="text-align:center;">
            <div style="font-size:0.85rem; color:#84cc16; font-weight:700; text-transform:uppercase;">Recommended Optimal Crop</div>
            <div class="crop-result">{primary}</div>
            <div style="font-weight:700; color:#84cc16; margin-top:0.5rem;">{top_recs[0]['confidence']}% Match Confidence</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### Top-3 Probability Ranking")
        for idx, rec in enumerate(top_recs):
            st.write(f"**#{idx+1} {rec['crop'].capitalize()}** ({rec['confidence']}%)")
            st.progress(float(rec['confidence']) / 100.0)
            
        st.markdown("---")
        st.markdown("### 🩺 Soil Health & Agronomic Advisory")
        st.write(f"**NPK Ratio:** `{soil['npk_ratio']}` | **Soil pH Reaction:** `{soil['ph_status']}`")
        
        for diag in soil['diagnoses']:
            pill = "pill-good" if any(w in diag.lower() for w in ["optimal", "ideal", "healthy", "balanced"]) else ("pill-bad" if "low" in diag.lower() or "acidic" in diag.lower() else "pill-warn")
            st.markdown(f'<span class="{pill}">{diag}</span>', unsafe_allow_html=True)
            
        st.markdown("#### ⚡ Recommended Action Plan:")
        for advice in soil['fertilizer_recommendations']:
            st.write(f"🌱 {advice}")

elif menu == "📊 ML Benchmarks":
    st.markdown("### 🧠 Machine Learning Performance Benchmarks")
    st.info(f"🏆 **Selected Model:** {predictor.best_model_name}")
    
    bench_df = pd.DataFrame(predictor.benchmark_results)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Classifier Accuracy Comparison (%)")
        st.bar_chart(bench_df.set_index('model_name')[['accuracy', 'f1_score']])
        
    with col2:
        st.markdown("#### Relative Feature Importance Breakdown")
        imp_df = pd.DataFrame({
            'Feature': list(predictor.feature_importances.keys()),
            'Gini Importance (%)': [v * 100 for v in predictor.feature_importances.values()]
        })
        st.bar_chart(imp_df.set_index('Feature'))

elif menu == "🌦️ Climate Simulator":
    st.markdown("""
    <div class="clay-card" style="text-align: center; padding: 3rem 2rem;">
        <div style="font-size: 3.5rem;">🔒</div>
        <div style="background: rgba(245, 158, 11, 0.2); color: #f59e0b; padding: 0.4rem 1.2rem; border-radius: 20px; display: inline-block; font-weight: 700;">
            ⏳ Coming Soon — Scheduled for 2nd Evaluation Phase
        </div>
        <h2 style="font-family: 'Outfit'; font-size: 2rem; margin-top: 1rem; color: #f0fdf4;">Climate Sensitivity & Stress Simulator</h2>
        <p style="color: #a7f3d0; max-width: 600px; margin: 0 auto 1.5rem auto;">
            This module is reserved for the <strong>2nd Evaluation Phase</strong>. It will feature real-time global warming scenario modeling (+1°C to +5°C temperature shifts and ±50% seasonal rainfall fluctuations).
        </p>
    </div>
    """, unsafe_allow_html=True)

elif menu == "📚 Crop Library":
    st.markdown("### 📚 Crop Library Catalog (22 Supported Crops)")
    search = st.text_input("🔍 Search Crop Name", "")
    
    filtered_crops = [c for c in predictor.crop_classes if search.lower() in c.lower()]
    
    cols = st.columns(3)
    for idx, crop in enumerate(filtered_crops):
        stat = predictor.crop_stats.get(crop, {})
        with cols[idx % 3]:
            st.markdown(f"""
            <div class="clay-card">
                <h4 style="color:#22c55e; font-family:'Outfit'; text-transform:capitalize;">🌱 {crop}</h4>
                <p style="font-size:0.85rem; color:#a7f3d0;">
                    <b>Opt. Temp:</b> {stat.get('temperature', {}).get('mean', '-')} °C<br>
                    <b>Opt. Rainfall:</b> {stat.get('rainfall', {}).get('mean', '-')} mm<br>
                    <b>Opt. Humidity:</b> {stat.get('humidity', {}).get('mean', '-')} %<br>
                    <b>Opt. pH:</b> {stat.get('ph', {}).get('mean', '-')}<br>
                    <b>Target NPK:</b> {stat.get('N', {}).get('mean', '-')}:{stat.get('P', {}).get('mean', '-')}:{stat.get('K', {}).get('mean', '-')}
                </p>
            </div>
            """, unsafe_allow_html=True)
