import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras

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

# ---------------- MODEL LOADING (ROBUST VERSION) ----------------
@st.cache_resource
def load_my_model():
    try:
        model = keras.models.load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.error("Model loading failed. Vérifie compatibilité TensorFlow/Keras.")
        st.stop()

model = load_my_model()

# ---------------- IMAGE PREPROCESSING ----------------
def preprocess_image(uploaded_file):
    try:
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

        face_img = cv2.resize(face_img, IMG_SIZE)
        face_img = face_img.astype("float32") / 255.0

        return face_img.reshape(1, 48, 48, 1), face_img

    except Exception as e:
        st.error("Erreur lors du traitement de l'image.")
        return None, None

# ---------------- UI HEADER ----------------
st.markdown("## 🧠 AI EMOTION SCAN SYSTEM")
st.markdown("`>> Module facial actif`")
st.divider()

# ---------------- UPLOAD ----------------
uploaded_file = st.file_uploader("Charger une image", type=["jpg", "jpeg", "png"])

if uploaded_file:

    processed, face_img = preprocess_image(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(uploaded_file, caption="INPUT IMAGE", use_container_width=True)

    with col2:
        if face_img is not None:
            st.image(face_img, caption="FACE EXTRACTED", use_container_width=True)
        else:
            st.warning("Aucun visage détecté")

    st.divider()

    # ---------------- PREDICTION ----------------
    if processed is not None:

        if st.button("LANCER ANALYSE"):

            with st.spinner("Analyse en cours..."):
                preds = model.predict(processed)
                class_id = int(np.argmax(preds))
                confidence = float(np.max(preds))

            st.success("Analyse terminée")

            st.markdown("### 📡 RESULTAT")

            st.code(f"""
EMOTION      : {EMOTION_LABELS[class_id]}
CONFIDENCE   : {confidence * 100:.2f}%
MODEL        : CNN Emotion AI
""")

            st.markdown("### 📊 PROBABILITÉS")

            for i, prob in enumerate(preds[0]):
                st.write(f"{EMOTION_LABELS[i]} : {prob:.3f}")
                st.progress(float(prob))

    else:
        st.error("Impossible de traiter l'image")

# ---------------- FOOTER ----------------
st.divider()
st.markdown("System ready | AI module active")