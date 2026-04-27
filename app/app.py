# app/app.py

import sys
from pathlib import Path
import streamlit as st
from PIL import Image
import pandas as pd
import plotly.express as px
import os

# Add src to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "src"))

import importlib
import infer_manual
importlib.reload(infer_manual)
from infer_manual import infer

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Aid Sense | Premium AI Crop Advisor",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ASSETS ---
BANNER_PATH = list(BASE_DIR.glob("aid_sense_banner*.png"))
BANNER_IMAGE = BANNER_PATH[0] if BANNER_PATH else "https://cdn-icons-png.flaticon.com/512/2917/2917995.png"

# --- SESSION STATE RESET ---
if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None

# --- CUSTOM CSS (Premium Design) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;800&display=swap');
    
    :root {
        --primary: #10b981;
        --secondary: #0ea5e9;
        --accent: #f59e0b;
        --bg: #0f172a;
    }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at top right, #1e293b, #0f172a);
        color: #f8fafc;
    }
    
    /* Premium Glassmorphism */
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 24px;
        padding: 30px;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.3);
        margin-bottom: 25px;
        transition: transform 0.3s ease, border 0.3s ease;
    }
    
    .glass-card:hover {
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 16px;
        height: 3.5em;
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        font-weight: 700;
        letter-spacing: 0.5px;
        border: none;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        box-shadow: 0 10px 20px rgba(16, 185, 129, 0.2);
    }
    
    .stButton>button:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(16, 185, 129, 0.4);
        background: linear-gradient(135deg, #34d399 0%, #10b981 100%);
        color: white;
    }
    
    /* Typography */
    .hero-title {
        background: linear-gradient(to right, #f8fafc, #94a3b8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3.5rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .hero-subtitle {
        color: #94a3b8;
        text-align: center;
        font-size: 1.2rem;
        margin-bottom: 3rem;
    }
    
    .disease-title {
        color: #10b981;
        font-size: 28px;
        font-weight: 800;
        margin-bottom: 10px;
    }
    
    .confidence-badge {
        background: rgba(16, 185, 129, 0.1);
        color: #10b981;
        padding: 6px 16px;
        border-radius: 30px;
        font-size: 14px;
        font-weight: 700;
        border: 1px solid rgba(16, 185, 129, 0.2);
    }
    
    .cure-header {
        color: #f59e0b;
        font-weight: 700;
        margin-top: 25px;
        font-size: 1.1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .sidebar-content {
        padding: 20px;
        background: rgba(15, 23, 42, 0.5);
        border-radius: 20px;
    }

    /* Override Streamlit elements */
    .stSelectbox div[data-baseweb="select"] {
        background-color: rgba(30, 41, 59, 0.5);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    [data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.image(BANNER_IMAGE, width="stretch")
    st.markdown("<h2 style='text-align: center; color: #10b981;'>AID SENSE PRO</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; color: #94a3b8; font-size: 0.9rem; margin-bottom: 20px;'>
        Advanced Neural Architecture for Precision Agriculture
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    st.subheader("🌐 ENVIRONMENT CONTEXT")
    crop = st.selectbox("CROP VARIETY", ["Tomato", "Potato", "Apple", "Corn", "Grape", "Other"])
    region = st.selectbox("GEOGRAPHIC REGION", ["Tamil Nadu", "Karnataka", "Maharashtra", "Uttar Pradesh", "Punjab", "Other"])
    season = st.selectbox("ACTIVE SEASON", ["Kharif", "Rabi", "Summer", "Unknown"])
    
    st.divider()
    with st.expander("🛠️ SYSTEM SPECIFICATIONS"):
        st.markdown(f"""
        **CORE ARCHITECTURE:**  
        `MobileNetV3-Large-Hybrid`
        
        **ACCURACY:**  
        `95.8% (Targeted)`
        
        **LATENCY:**  
        `< 150ms per inference`
        
        **DATASET:**  
        `PlantVillage (54,305 samples)`
        """)

# --- MAIN PAGE ---
st.markdown("<div class='hero-title'>Smart Crop Guardian</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-subtitle'>Precision Diagnostics Powered by Advanced Neural Vision</div>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("### 📷 Vision Scan")
    uploaded_image = st.file_uploader(
        "DROP LEAF IMAGE HERE",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear, high-resolution image for analysis."
    )
    
    if uploaded_image:
        if st.session_state.last_uploaded_file != uploaded_image.name:
            st.session_state.last_uploaded_file = uploaded_image.name
            if 'result' in st.session_state:
                del st.session_state.result
        
        image = Image.open(uploaded_image)
        st.image(image, caption="Neural Scan Target", width="stretch")
        
        if st.button("⚡ EXECUTE NEURAL ANALYSIS"):
            with st.spinner("Processing plant biometric signals..."):
                try:
                    st.session_state.result = infer(
                        image,
                        crop=crop,
                        region=region,
                        season=season,
                    )
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
    else:
        st.info("Waiting for diagnostic input...")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    if 'result' in st.session_state:
        res = st.session_state.result
        
        st.markdown(f"""
            <div class="glass-card">
                <div style="display: flex; justify-content: space-between; align-items: start;">
                    <div>
                        <div class="disease-title">{res['issue']}</div>
                        <span class="confidence-badge">SCAN PRECISION: {res['confidence']*100:.2f}%</span>
                        <div style="font-size: 11px; color: #64748b; margin-top: 8px;">NEURAL SIGNATURE: <code>{res['label']}</code></div>
                    </div>
                    <div style="font-size: 40px; filter: drop-shadow(0 0 10px rgba(16,185,129,0.5));">🌱</div>
                </div>
                <div class="cure-header">🚨 ACTION PROTOCOL</div>
                <div style="color: #cbd5e1; line-height: 1.8; margin-top: 15px; background: rgba(0,0,0,0.2); padding: 15px; border-radius: 12px;">
                    {res['cure']}
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Top-3 Predictions Visualization
        with st.expander("📊 DETAILED PROBABILITY MATRIX"):
            top_df = pd.DataFrame(res['top_predictions'])
            top_df['confidence'] = top_df['confidence'] * 100
            
            fig = px.bar(
                top_df, 
                x='confidence', 
                y='issue', 
                orientation='h',
                color='confidence',
                color_continuous_scale=[[0, '#1e293b'], [1, '#10b981']],
                labels={'confidence': 'MATCH %', 'issue': 'CONDITION'},
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color="#94a3b8",
                height=300,
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=False)
            )
            st.plotly_chart(fig, width="stretch")
            
        st.success("Analysis report generated successfully.")
    else:
        st.markdown("""
            <div class="glass-card" style="text-align: center; color: #64748b; padding: 100px 30px;">
                <div style="font-size: 60px; margin-bottom: 20px; opacity: 0.3;">🔬</div>
                <p>Awaiting biometric data. Please upload a leaf image to proceed with the diagnostic cycle.</p>
            </div>
        """, unsafe_allow_html=True)

# --- FOOTER ---
st.divider()
st.markdown("""
<div style='text-align: center; color: #475569; font-size: 0.8rem; margin-bottom: 20px;'>
    AID SENSE PRO V3.0 | SECURE NEURAL DIAGNOSTICS FOR GLOBAL SUSTAINABILITY
</div>
""", unsafe_allow_html=True)
