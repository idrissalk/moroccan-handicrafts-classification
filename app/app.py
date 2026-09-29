# =====================================
# 🏺 Streamlit App - Classification de l'artisanat marocain
# Lancer avec : streamlit run streamlit_app.py
# =====================================
import os
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.convnext import preprocess_input

# =====================================
# ⚙️ Configuration de la page
# =====================================
st.set_page_config(
    page_title="Handcrafts Classifier",
    page_icon="🏺",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "convnext_model.keras")
IMG_SIZE = (224, 224)
CLASSES = ["Poufs", "Tapis", "Babouches", "Bottes", "Sacs", "Sandales", "Vestes"]

EMOJIS = {
    "Poufs": "🛋️", "Tapis": "🧶", "Babouches": "🥿", "Bottes": "🥾",
    "Sacs": "👜", "Sandales": "👡", "Vestes": "🧥",
}


# =====================================
# 📌 Chargement du modèle (une seule fois)
# =====================================
@st.cache_resource(show_spinner="Chargement du modèle ConvNeXt...")
def get_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Modèle introuvable : {MODEL_PATH}")
        st.stop()
    return load_model(MODEL_PATH)


# =====================================
# 🔍 Prédiction
# =====================================
def predict(model, img: Image.Image):
    img = img.convert("RGB").resize(IMG_SIZE)
    x = np.expand_dims(np.array(img, dtype="float32"), axis=0)
    x = preprocess_input(x)
    return model.predict(x, verbose=0)[0]


# =====================================
# 🎨 Sidebar
# =====================================
with st.sidebar:
    st.header("ℹ️ À propos")
    st.write(
        "Application de **classification d'images** de produits artisanaux "
        "marocains, basée sur un modèle **ConvNeXt** (Transfer Learning) "
        "entraîné avec TensorFlow / Keras."
    )
    st.subheader("Catégories")
    for c in CLASSES:
        st.write(f"{EMOJIS[c]} {c}")
    st.divider()
    top_k = st.slider("Nombre de prédictions à afficher", 1, len(CLASSES), 3)
    st.caption("Stack : Python · TensorFlow · ConvNeXt · Streamlit")

# =====================================
# 🖼️ Page principale
# =====================================
st.title("🏺 Classification de l'artisanat marocain")
st.write("Importez une photo d'un produit artisanal et le modèle identifie sa catégorie.")

model = get_model()

tab_upload, tab_camera = st.tabs(["📁 Importer une image", "📷 Prendre une photo"])

with tab_upload:
    uploaded = st.file_uploader("Choisir une image", type=["jpg", "jpeg", "png", "webp"])
with tab_camera:
    photo = st.camera_input("Prendre une photo")

source = uploaded or photo

if source is None:
    st.info("👆 Ajoutez une image pour lancer la prédiction.")
else:
    img = Image.open(source)
    col_img, col_res = st.columns([1, 1.2], gap="large")

    with col_img:
        st.image(img, caption="Image analysée", use_container_width=True)

    with st.spinner("Analyse en cours..."):
        probs = predict(model, img)

    order = np.argsort(probs)[::-1]
    best = order[0]

    with col_res:
        st.subheader("Résultat")
        st.success(f"{EMOJIS[CLASSES[best]]} **{CLASSES[best]}**")
        st.metric("Confiance", f"{probs[best] * 100:.1f} %")
        st.progress(float(probs[best]))

        if probs[best] < 0.5:
            st.warning("⚠️ Confiance faible : l'image est peut-être hors des catégories connues.")

        st.subheader(f"Top {top_k} prédictions")
        df = pd.DataFrame(
            {
                "Catégorie": [CLASSES[i] for i in order[:top_k]],
                "Probabilité": [float(probs[i]) for i in order[:top_k]],
            }
        ).set_index("Catégorie")
        st.bar_chart(df)