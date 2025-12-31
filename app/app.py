# app/app.py

import sys
from pathlib import Path

import streamlit as st
from PIL import Image

# Add src to sys.path so we can import our infer module
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "src"))

# 🔴 IMPORTANT: use infer_manual, NOT infer_tfds
from infer_manual import infer

st.set_page_config(
    page_title="Aid Sense – AI Crop Disease Advisor",
    layout="centered"
)

st.title("🌱 Aid Sense – AI Crop Disease & Cure Advisor")
st.write(
    "Upload a crop leaf image and select basic details. "
    "The AI will detect likely disease and give treatment advice."
)

uploaded_image = st.file_uploader(
    "Upload crop leaf image",
    type=["jpg", "jpeg", "png"]
)

col1, col2, col3 = st.columns(3)
with col1:
    crop = st.selectbox("Crop", ["Tomato", "Potato", "Apple", "Corn", "Grape", "Other"])
with col2:
    region = st.selectbox("Region", ["Tamil Nadu", "Karnataka", "Maharashtra", "Uttar Pradesh", "Punjab", "Other"])
with col3:
    season = st.selectbox("Season", ["Kharif", "Rabi", "Summer", "Unknown"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded image", use_column_width=True)

    if st.button("🔍 Analyse & Get Cure"):
        with st.spinner("Analysing disease…"):
            # new signature uses crop + region + season
            result = infer(
                image,
                crop=crop,
                region=region,
                season=season,
            )

        st.markdown("## 🧪 Detected Disease")
        st.success(f"**{result['issue']}**")
        st.write(f"**Raw label:** `{result['label']}`")
        st.write(f"**Model confidence:** {result['confidence']*100:.2f}%")

        st.markdown("## 🚑 Recommended Cure")
        st.warning(result["cure"])
else:
    st.info("Please upload a clear image of a single leaf.")
