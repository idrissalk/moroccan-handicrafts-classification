# 🏺 Moroccan Handicrafts Classification

Application web de **classification d'images** de produits artisanaux marocains,
basée sur un modèle **ConvNeXt** (Transfer Learning) et déployée avec **Streamlit**.

## 📸 Aperçu

_Ajouter une capture d'écran de l'application ici :_
`[<img width="1270" height="842" alt="Capture d&#39;écran 2026-09-29 150100" src="https://github.com/user-attachments/assets/8d3fa736-8fde-499e-9a2a-6d300e9351dc" />]
`

## 🎯 Catégories

Poufs · Tapis · Babouches · Bottes · Sacs · Sandales · Vestes

## 🧠 Modèle

- Architecture : ConvNeXt (pré-entraîné sur ImageNet) + couches de classification
- Framework : TensorFlow / Keras
- Taille d'entrée : 224 × 224
- Accuracy sur le jeu de test : **XX %** _(à compléter)_

## 📁 Structure du projet

```
moroccan-handicrafts-classification/
├── app/
│   └── app.py               # Interface Streamlit
├── models/
│   └── convnext_model.keras # Modèle entraîné
├── notebooks/
│   └── handicrafts_classification.ipynb  # Entraînement et évaluation
├── requirements.txt
├── .gitignore
└── README.md
```

## 🚀 Lancer le projet

```bash
git clone https://github.com/<ton-user>/moroccan-handicrafts-classification.git
cd moroccan-handicrafts-classification
pip install -r requirements.txt
streamlit run app/app.py
```

## 🛠️ Stack

Python · TensorFlow/Keras · ConvNeXt · Streamlit · NumPy · Pandas · Pillow
