# src/cure_guide.py

CURE_GUIDE = {
    # --- TOMATO ---
    "Tomato___healthy": {
        "issue": "Tomato – Healthy",
        "cure": "No disease detected. Maintain regular irrigation and monitor for pests."
    },
    "Tomato___Early_blight": {
        "issue": "Tomato – Early Blight",
        "cure": "Remove infected lower leaves. Apply copper-based fungicides. Avoid overhead watering."
    },
    "Tomato___Late_blight": {
        "issue": "Tomato – Late Blight (Emergency)",
        "cure": "Destroy infected plants. Apply systemic fungicides like metalaxyl. Keep leaves dry."
    },
    "Tomato___Bacterial_spot": {
        "issue": "Tomato – Bacterial Spot",
        "cure": "Avoid handling plants when wet. Use copper-based sprays early in the season."
    },
    "Tomato___Leaf_Mold": {
        "issue": "Tomato – Leaf Mold",
        "cure": "Improve air circulation and reduce humidity in the canopy. Use resistant varieties."
    },
    "Tomato___Septoria_leaf_spot": {
        "issue": "Tomato – Septoria Leaf Spot",
        "cure": "Remove infected debris. Use fungicides containing chlorothalonil or mancozeb."
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "issue": "Tomato – Spider Mites",
        "cure": "Increase humidity and use neem oil or insecticidal soap to control mite populations."
    },
    "Tomato___Target_Spot": {
        "issue": "Tomato – Target Spot",
        "cure": "Protect with fungicides like azoxystrobin. Ensure proper row spacing for airflow."
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "issue": "Tomato – Yellow Leaf Curl Virus",
        "cure": "Control whiteflies using yellow sticky traps and neem oil. Remove infected plants."
    },
    "Tomato___Tomato_mosaic_virus": {
        "issue": "Tomato – Mosaic Virus",
        "cure": "No cure. Remove infected plants. Wash hands/tools to prevent spread via contact."
    },

    # --- POTATO ---
    "Potato___healthy": {
        "issue": "Potato – Healthy",
        "cure": "Regular hilling and balanced NPK fertilization. Monitor weekly."
    },
    "Potato___Early_blight": {
        "issue": "Potato – Early Blight",
        "cure": "Apply protective fungicides. Ensure crop rotation and remove old crop debris."
    },
    "Potato___Late_blight": {
        "issue": "Potato – Late Blight",
        "cure": "Treat with systemic fungicides. Ensure good field drainage and destroy infected tubers."
    },

    # --- APPLE ---
    "Apple___Apple_scab": {
        "issue": "Apple – Apple Scab",
        "cure": "Rake fallen leaves. Apply fungicides during the 'green tip' stage."
    },
    "Apple___Black_rot": {
        "issue": "Apple – Black Rot",
        "cure": "Prune out cankers and remove 'mummy' fruit. Apply lime-sulfur or captan."
    },
    "Apple___Cedar_apple_rust": {
        "issue": "Apple – Cedar Apple Rust",
        "cure": "Remove nearby juniper/cedar trees. Use resistant varieties or sulfur sprays."
    },
    "Apple___healthy": {
        "issue": "Apple – Healthy",
        "cure": "Standard pruning and pest monitoring. Ensure proper thinning of fruit."
    },

    # --- CORN ---
    "Corn_(maize)___Common_rust_": {
        "issue": "Corn – Common Rust",
        "cure": "Usually managed by resistant hybrids. Apply foliar fungicides if severe."
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "issue": "Corn – Gray Leaf Spot",
        "cure": "Tillage and crop rotation. Use fungicides if threshold is reached."
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "issue": "Corn – Northern Leaf Blight",
        "cure": "Maximize row spacing. Apply fungicides like pyraclostrobin if detected early."
    },
    "Corn_(maize)___healthy": {
        "issue": "Corn – Healthy",
        "cure": "Ensure adequate nitrogen and drought management."
    },

    # --- GRAPE ---
    "Grape___Black_rot": {
        "issue": "Grape – Black Rot",
        "cure": "Clean up vine debris. Apply fungicides from pre-bloom through berry touch."
    },
    "Grape___Esca_(Black_Measles)": {
        "issue": "Grape – Esca",
        "cure": "Prune away dead wood. Protect large pruning wounds with sealants."
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "issue": "Grape – Leaf Blight",
        "cure": "Apply copper or sulfur fungicides. Improve canopy ventilation."
    },
    "Grape___healthy": {
        "issue": "Grape – Healthy",
        "cure": "Monitor for mildews and maintain vine vigor."
    },

    # --- OTHER CROPS ---
    "Orange___Haunglongbing_(Citrus_greening)": {
        "issue": "Orange – Citrus Greening",
        "cure": "No cure. Control the Asian citrus psyllid vector. Remove infected trees immediately."
    },
    "Peach___Bacterial_spot": {
        "issue": "Peach – Bacterial Spot",
        "cure": "Use resistant cultivars. Apply copper or oxytetracycline sprays in spring."
    },
    "Peach___healthy": {
        "issue": "Peach – Healthy",
        "cure": "Maintain tree health with proper pruning and dormancy sprays."
    },
    "Pepper,_bell___Bacterial_spot": {
        "issue": "Pepper – Bacterial Spot",
        "cure": "Use certified disease-free seeds. Apply bactericides based on copper."
    },
    "Pepper,_bell___healthy": {
        "issue": "Pepper – Healthy",
        "cure": "Check for aphids and ensure consistent soil moisture."
    },
    "Blueberry___healthy": { "issue": "Blueberry – Healthy", "cure": "Acidify soil if needed. Mulch for moisture." },
    "Cherry_(including_sour)___Powdery_mildew": { "issue": "Cherry – Powdery Mildew", "cure": "Use sulfur sprays and improve airflow." },
    "Cherry_(including_sour)___healthy": { "issue": "Cherry – Healthy", "cure": "Regular pruning and fruit rot prevention." },
    "Raspberry___healthy": { "issue": "Raspberry – Healthy", "cure": "Prune out old canes. Monitor for rust." },
    "Soybean___healthy": { "issue": "Soybean – Healthy", "cure": "Monitor for bean leaf beetles and rust." },
    "Squash___Powdery_mildew": { "issue": "Squash – Powdery Mildew", "cure": "Apply neem oil or potassium bicarbonate." },
    "Strawberry___Leaf_scorch": { "issue": "Strawberry – Leaf Scorch", "cure": "Remove old leaves. Avoid excessive nitrogen." },
    "Strawberry___healthy": { "issue": "Strawberry – Healthy", "cure": "Mulch with straw. Monitor for grey mold." },
}

DEFAULT_CURE = {
    "issue": "Agricultural Condition Found",
    "cure": (
        "The AI has detected a pattern. Please review the breakdown chart to see potential candidates "
        "and consult with a local agronomist for a physical inspection."
    ),
}
