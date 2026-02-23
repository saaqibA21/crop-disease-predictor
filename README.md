# 🌱 Aid Sense – AI Crop Disease & Cure Advisor

![Aid Sense Banner](https://cdn-icons-png.flaticon.com/512/2917/2917995.png)

Aid Sense is a high-performance, hybrid deep learning application designed to detect crop diseases from leaf images and provide actionable organic treatment advice. It combines **EfficientNetB3** computer vision with contextual data (Region, Season, Crop Type) to deliver state-of-the-art diagnostic accuracy.

## 🚀 Experience the App
**[View Live Demo](https://share.streamlit.io/saaqibA21/crop-disease-predictor/main/app/app.py)** *(Link works once deployed to Streamlit Cloud)*

---

## ✨ Key Features
- **Hybrid AI Analysis**: Combines image features with environmental context for superior precision.
- **Top-3 Diagnostic Probabilities**: Transparent breakdown of potential matches using interactive Plotly charts.
- **Organic Cure Guides**: Immediate treatment protocols for common diseases like Early Blight and Late Blight.
- **Premium UI**: Modern Glassmorphism design with responsive elements and custom typography.
- **Farmer-Centric**: Simplified workflows for field use.

## 🛠️ Technical Stack
- **Frontend**: Streamlit (Premium Custom CSS)
- **Deep Learning**: TensorFlow / Keras (EfficientNetB3)
- **Visualization**: Plotly Express
- **Data Handling**: Pandas & NumPy

## 📦 Installation & Local Run

1. **Clone the repository**:
   ```bash
   git clone https://github.com/saaqibA21/crop-disease-predictor.git
   cd crop-disease-predictor
   ```

2. **Set up Virtual Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Application**:
   ```bash
   streamlit run app/app.py
   ```

---

## 🏗️ Project Structure
```text
├── app/
│   └── app.py              # Main Streamlit Application
├── src/
│   ├── infer_manual.py     # AI Inference Engine
│   ├── cure_guide.py       # Knowledge base for disease treatments
│   └── train_*.py          # Model training scripts
├── models/
│   └── hybrid_*.h5         # Pre-trained Deep Learning Model
├── data/
│   └── dataset.csv         # Structured metadata
└── requirements.txt        # Production dependencies
```

## 🛡️ Disclaimer
*This tool is intended for advisory purposes. Always consult with a local agronomist for critical agricultural decisions.*

---
Created with ❤️ for sustainable farming.
