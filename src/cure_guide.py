# src/cure_guide.py

CURE_GUIDE = {
    "Tomato___healthy": {
        "issue": "Tomato – Healthy",
        "cure": (
            "No clear disease detected. Maintain normal care: consistent watering, balanced fertilization, "
            "remove old yellow leaves, and monitor weekly for new spots or lesions."
        )
    },
    "Tomato___Early_blight": {
        "issue": "Tomato – Early Blight",
        "cure": (
            "Remove the lowest infected leaves and avoid wetting foliage when watering. "
            "Apply a copper-based or mancozeb fungicide every 7–10 days as per label instructions. "
            "Rotate with non-tomato crops next season."
        )
    },
    "Tomato___Late_blight": {
        "issue": "Tomato – Late Blight (Emergency)",
        "cure": (
            "Immediately remove and destroy heavily infected plants (do not compost). "
            "Avoid overhead irrigation. Spray a systemic fungicide containing metalaxyl or similar "
            "according to local guidelines. Keep nearby potato and tomato plots under strict watch."
        )
    },
    "Potato___healthy": {
        "issue": "Potato – Healthy",
        "cure": (
            "No major problems detected. Maintain proper hilling, irrigation, and balanced fertilization. "
            "Inspect weekly for spots, wilting, or rotting lesions."
        )
    },
    "Potato___Early_blight": {
        "issue": "Potato – Early Blight",
        "cure": (
            "Remove old infected foliage. Avoid water stress. Apply a protectant fungicide like mancozeb "
            "at 7–10 day intervals, especially in warm and humid periods."
        )
    },
    "Potato___Late_blight": {
        "issue": "Potato – Late Blight (Emergency)",
        "cure": (
            "Destroy infected plants and avoid contact with healthy fields. "
            "Apply systemic fungicides recommended locally for late blight. "
            "Ensure good field drainage and avoid dense canopies that keep leaves wet."
        )
    },
    "Corn_(maize)___Common_rust_": {
        "issue": "Corn – Common Rust",
        "cure": (
            "Usually managed by planting resistant hybrids. If infection is early and severe, "
            "apply a foliar fungicide. Ensure good soil nitrogen and avoid overhead irrigation."
        )
    },
    "Grape___Black_rot": {
        "issue": "Grape – Black Rot",
        "cure": (
            "Prune infected vines and remove mummified berries from the ground. "
            "Apply fungicides early in the season (pre-bloom and post-bloom). "
            "Maintain an open canopy for better airflow and faster drying."
        )
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "issue": "Tomato – Yellow Leaf Curl Virus",
        "cure": (
            "This virus is spread by whiteflies. Use physical barriers like fine mesh nets. "
            "Control whitefly populations using yellow sticky traps or neem oil. "
            "Remove and destroy infected plants immediately to prevent further spread."
        )
    },
    "Apple___Apple_scab": {
        "issue": "Apple – Apple Scab",
        "cure": (
            "Rake and destroy fallen leaves in autumn to reduce spores. "
            "Apply sulfur or lime-sulfur sprays during the green tip phase and "
            "continue at recommended intervals during the growing season."
        )
    },
    # You can keep adding more mapping entries as needed for other PlantVillage classes
}

DEFAULT_CURE = {
    "issue": "Unknown / less common condition",
    "cure": (
        "The pattern does not strongly match a known disease in this version of the model. "
        "Isolate affected plants, remove the most damaged leaves, and consult a local agronomist or extension worker."
    ),
}
