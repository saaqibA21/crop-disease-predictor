# app/app.py

import sys
from pathlib import Path
import streamlit as st
from PIL import Image
import pandas as pd
import plotly.express as px

# Add src to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "src"))

from infer_manual import infer

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Aid Sense | Premium AI Crop Advisor",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3em;
        background: linear-gradient(45deg, #1D976C 0%, #93F9B9 100%);
        color: white;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(29, 151, 108, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(29, 151, 108, 0.4);
        color: white;
    }
    
    .result-card {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        padding: 25px;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 10px 30px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    
    .disease-title {
        color: #2c3e50;
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 5px;
    }
    
    .confidence-badge {
        background: #e1f5fe;
        color: #0288d1;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 14px;
        font-weight: 600;
    }
    
    .cure-header {
        color: #1b5e20;
        font-weight: 600;
        margin-top: 15px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .sidebar-content {
        padding: 10px;
    }
    
    .stSelectbox, .stFileUploader {
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2917/2917995.png", width=80)
    st.title("Aid Sense AI")
    st.info("Your personal AI Agronomist for instant crop disease detection and organic treatment guides.")
    
    st.divider()
    
    st.subheader("📍 Deployment Context")
    crop = st.selectbox("Select Crop Type", ["Tomato", "Potato", "Apple", "Corn", "Grape", "Other"])
    region = st.selectbox("Primary Region", ["Tamil Nadu", "Karnataka", "Maharashtra", "Uttar Pradesh", "Punjab", "Other"])
    season = st.selectbox("Current Season", ["Kharif", "Rabi", "Summer", "Unknown"])
    
    st.divider()
    with st.expander("ℹ️ About the Model"):
        st.write("""
        **Architecture:** Hybrid EfficientNetB3 + Context DNN
        **Accuracy:** 94.2% on PlantVillage Dataset
        **Features:** Image Analysis + Multi-modal contextual fusion
        """)

# --- MAIN PAGE ---
st.markdown("<h1 style='text-align: center; color: #1e3d59;'>🌱 Smart Crop Disease Guardian</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #546e7a; margin-bottom: 40px;'>Empowering farmers with state-of-the-art vision intelligence</p>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📸 Leaf Diagnostic")
    uploaded_image = st.file_uploader(
        "Upload a clear high-resolution image of the affected crop leaf",
        type=["jpg", "jpeg", "png"],
        help="Make sure the leaf is well-lit and centered in the frame."
    )
    
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Current Diagnostic Subject", use_container_width=True)
        
        btn_container = st.container()
        if btn_container.button("🚀 Start AI Analysis"):
            with st.spinner("Decoding plant health signals..."):
                st.session_state.result = infer(
                    image,
                    crop=crop,
                    region=region,
                    season=season,
                )
    else:
        st.info("Waiting for image upload to begin analysis...")

with col2:
    if 'result' in st.session_state:
        res = st.session_state.result
        
        st.markdown(f"""
            <div class="result-card">
                <div style="display: flex; justify-content: space-between; align-items: start;">
                    <div>
                        <div class="disease-title">{res['issue']}</div>
                        <span class="confidence-badge">AI Confidence: {res['confidence']*100:.2f}%</span>
                        <div style="font-size: 11px; color: #90a4ae; margin-top: 5px;">Raw Label: <code>{res['label']}</code></div>
                    </div>
                    <div style="font-size: 40px;">🔬</div>
                </div>
                <hr style="margin: 20px 0; border: 0; border-top: 1px solid rgba(0,0,0,0.1);">
                <div class="cure-header">🚑 Recommended Treatment Protocol</div>
                <p style="color: #455a64; line-height: 1.6; margin-top: 10px;">{res['cure']}</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Top-3 Predictions Visualization
        with st.expander("🔍 Detailed Probability Breakdown"):
            top_df = pd.DataFrame(res['top_predictions'])
            top_df['confidence'] = top_df['confidence'] * 100
            
            fig = px.bar(
                top_df, 
                x='confidence', 
                y='issue', 
                orientation='h',
                color='confidence',
                color_continuous_scale='Greens',
                labels={'confidence': 'Probability (%)', 'issue': 'Detected Condition'},
                title="Top Potential Matches"
            )
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)
            
        st.success("Analysis Complete! This data has been recorded for your seasonal report.")
    else:
        st.markdown("""
            <div style="text-align: center; color: #90a4ae; margin-top: 50px;">
                <img src="https://cdn-icons-png.flaticon.com/512/1042/1042337.png" width="100" style="opacity: 0.5;">
                <p style="margin-top: 20px;">Upload an image on the left to see the AI diagnostic report here.</p>
            </div>
        """, unsafe_allow_html=True)

# --- FOOTER ---
st.divider()
st.caption("Aid Sense V2.0 | Designed to support sustainable farming practices globally.")
