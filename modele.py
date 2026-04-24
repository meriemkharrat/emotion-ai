import streamlit as st
import numpy as np

# ---------------- SAFE IMPORTS (IMPORTANT) ----------------
try:
    import cv2
except Exception:
    cv2 = None
    st.warning("OpenCV non disponible - certaines fonctions seront limitées.")

try:
    import tensorflow as tf
    from tensorflow import keras
except Exception as e:
    tf = None
    keras = None
    st.error("TensorFlow non disponible. Vérifie requirements.txt.")

# ---------------- CONFIG ----------------
MODEL_PATH = "best_model.keras"
IMG_SIZE = (48, 48)

EMOTION_LABELS = [
    "ANGRY",
    "DISGUST",
    "FEAR",
    "HAPPY",
    "SAD",
    "SURPRISE",
    "NEUTRAL"
]

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Emotion Scan System",
    layout="wide"
)

# ---------------- MODEL LOADING (ULTRA SAFE) ----------------
@st.cache_resource
def load_model_safe():
    if keras is None:
        return None

    try:
        model = keras.models.load_model(
            MODEL_PATH,
            compile=False,
            safe_mode=False
        )
        return model

    except Exception as e:
        st.error("Model loading failed (compatibilité TensorFlow/Keras).")
        st.code(str(e))
        return None

model = load_model_safe()

# ---------------- PREPROCESS ----------------
def preprocess_image(uploaded_file):
    if cv2 is None:
        return None, None

    try:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

        if img is None:
            return None, None

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        faces = face_cascade.detectMultiScale(img, 1.1, 5)

        face_img = img

        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_img = img[y:y+h, x:x+w]

        face_img = cv2.resize(face_img, IMG_SIZE)
        face_img = face_img.astype("float32") / 255.0

        return face_img.reshape(1, 48, 48, 1), face_img

    except Exception:
        return None, None

# ---------------- UI ----------------
st.markdown("## 🧠 AI EMOTION SCAN SYSTEM")
st.markdown("System initialized | Safe mode active")
st.divider()

uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded_file:

    processed, face_img = preprocess_image(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(uploaded_file, caption="INPUT", use_container_width=True)

    with col2:
        if face_img is not None:
            st.image(face_img, caption="FACE", use_container_width=True)
        else:
            st.warning("No face detected or OpenCV unavailable")

    st.divider()

    # ---------------- PREDICTION ----------------
    if st.button("RUN ANALYSIS"):

        if model is None:
            st.error("Model non chargé - vérifie TensorFlow/Keras versions")
            st.stop()

        if processed is None:
            st.error("Image non exploitable")
            st.stop()

        with st.spinner("Processing..."):
            preds = model.predict(processed)
            class_id = int(np.argmax(preds))
            confidence = float(np.max(preds))

        st.success("Analysis complete")

        st.markdown("### RESULT")

        st.code(f"""
EMOTION    : {EMOTION_LABELS[class_id]}
CONFIDENCE : {confidence*100:.2f}%
MODEL      : CNN Emotion AI
""")

        st.markdown("### PROBABILITIES")

        for i, p in enumerate(preds[0]):
            st.write(f"{EMOTION_LABELS[i]} : {p:.3f}")
            st.progress(float(p))

# ---------------- FOOTER ----------------
st.divider()
st.markdown("System stable | Safe execution mode enabled")