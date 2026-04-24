import os
import tensorflow as tf

# =========================
# CONFIG STABLE
# =========================
MODEL_PATH = "best_model.keras"
SAVEDMODEL_PATH = "saved_model_emotion"

# =========================
# LOAD MODEL ROBUST (SAFE)
# =========================
def load_emotion_model():
    """
    Charge le modèle de manière compatible multi-versions TF/Keras.
    """

    model = None

    # 1️⃣ Essai standard Keras (.keras)
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("✅ Modèle chargé depuis .keras")
        return model
    except Exception as e:
        print("⚠️ Échec chargement .keras :", e)

    # 2️⃣ Fallback SavedModel (recommandé production)
    try:
        if os.path.exists(SAVEDMODEL_PATH):
            model = tf.keras.models.load_model(SAVEDMODEL_PATH, compile=False)
            print("✅ Modèle chargé depuis SavedModel")
            return model
    except Exception as e:
        print("⚠️ Échec SavedModel :", e)

    # 3️⃣ Dernier fallback (TensorFlow direct load)
    try:
        model = tf.saved_model.load(SAVEDMODEL_PATH)
        print("✅ SavedModel TF natif chargé")
        return model
    except Exception as e:
        print("❌ Impossible de charger le modèle :", e)

    raise RuntimeError("🚨 Chargement du modèle impossible. Vérifie compatibilité TF/Keras.")


# =========================
# INITIALISATION
# =========================
model = load_emotion_model()


# =========================
# PREDICTION SAFE WRAPPER
# =========================
def predict_emotion(image_array):
    """
    Wrapper sécurisé pour prédiction.
    Compatible Keras Model ou SavedModel.
    """

    try:
        # cas Keras classique
        if hasattr(model, "predict"):
            return model.predict(image_array)

        # cas SavedModel
        infer = model.signatures["serving_default"]
        return infer(tf.constant(image_array))

    except Exception as e:
        print("❌ Erreur prédiction :", e)
        return None