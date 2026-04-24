import streamlit as st
import numpy as np
import cv2
import time
import tensorflow as tf
from tensorflow.keras.models import load_model

# ---------------- CONFIG ----------------
MODEL_PATH = "best_model.keras"
IMG_SIZE = (48, 48)

EMOTION_LABELS = {
    0: "ANGRY",
    1: "DISGUST",
    2: "FEAR",
    3: "HAPPY",
    4: "SAD",
    5: "SURPRISE",
    6: "NEUTRAL"
}

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Emotion Scan System",
    layout="wide"
)

# ---------------- STYLE ----------------
st.markdown("""
<style>
body {
    background-color: #000000;
    color: #00FFAA;
    font-family: monospace;
}
.block-container {
    background-color: #0a0a0a;
    border: 1px solid #00FFAA;
    padding: 20px;
}
.stButton>button {
    background-color: black;
    color: #00FFAA;
    border: 1px solid #00FFAA;
}
</style>
""", unsafe_allow_html=True)

# ---------------- MODEL LOADING (IMPORTANT FIX) ----------------
@st.cache_resource
def load_my_model():
    model = load_model(MODEL_PATH)
    return model

model = load_my_model()

# ---------------- PREPROCESS IMAGE ----------------
def preprocess_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return None, None

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(img, 1.1, 5)

    face_img = img.copy()

    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_img = img[y:y+h, x:x+w]

    resized = cv2.resize(face_img, IMG_SIZE)
    normalized = resized.astype("float32") / 255.0

    return normalized.reshape(1, 48, 48, 1), face_img

# ---------------- HEADER ----------------
st.markdown("## 🧠 AI EMOTION SCAN SYSTEM")
st.markdown("`>> Initialisation du module d’analyse faciale...`")

st.divider()

# ---------------- UPLOAD ----------------
uploaded_file = st.file_uploader(
    ">> Charger une image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    processed, face_img = preprocess_image(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("`[INPUT IMAGE]`")
        st.image(uploaded_file, use_container_width=True)

    with col2:
        st.markdown("`[FACE EXTRACTION]`")
        if face_img is not None:
            st.image(face_img, use_container_width=True)
        else:
            st.warning("No face detected")

    st.divider()

    if processed is not None:

        if st.button(">> LANCER ANALYSE"):

            progress = st.progress(0)

            # Animation scan
            for i in range(100):
                time.sleep(0.01)
                progress.progress(i + 1)

            preds = model.predict(processed)
            class_id = int(np.argmax(preds))
            confidence = float(np.max(preds))

            st.divider()
            st.markdown("### 📡 RAPPORT D’ANALYSE")

            st.code(f"""
STATUS        : SUCCESS
DETECTED      : {EMOTION_LABELS[class_id]}
CONFIDENCE    : {confidence*100:.2f}%
MODEL         : CNN Emotion AI
""")

            st.markdown("### 📊 DISTRIBUTION")

            for i, prob in enumerate(preds[0]):
                st.write(f"{EMOTION_LABELS[i]} : {prob:.3f}")
                st.progress(float(prob))

    else:
        st.error("Erreur de traitement de l'image")

# ---------------- FOOTER ----------------
st.divider()
st.markdown("`System ready | AI Module Active`")