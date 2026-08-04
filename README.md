# 🌱 Crop Choice AI — Precision Agronomic Intelligence Platform

[![Live Demo](https://img.shields.io/badge/🌐_Live_App-Streamlit_Cloud-22c55e?style=for-the-badge&logo=streamlit)](https://cropchoiceai-inoor.streamlit.app/)
[![Python Version](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![ML Engine](https://img.shields.io/badge/Scikit--Learn-1.9-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Framework](https://img.shields.io/badge/Streamlit-1.60-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)

> **🚀 Live Application URL:** [https://cropchoiceai-inoor.streamlit.app/](https://cropchoiceai-inoor.streamlit.app/)

---

## 📌 Executive Summary

**Crop Choice AI** is an end-to-end Machine Learning powered **Crop Selection & Agronomic Soil Health Intelligence System**. It analyzes precision soil chemical properties (**Nitrogen**, **Phosphorus**, **Potassium**, **pH**) and environmental micro-climate parameters (**Temperature**, **Relative Humidity**, **Rainfall**) to recommend optimal high-yield crops while prescribing targeted soil nutrient remediation strategies.

Built on the authentic **Kaggle Crop Recommendation Dataset** (2,200 observations across 22 crop species), the platform features multi-algorithm benchmarking, automated soil health diagnostics, interactive feature importance analytics, and a custom **Claymorphic Environmental UI System**.

---

## 🌐 Live Web Application

The platform is deployed and publicly accessible on Streamlit Cloud:

👉 **[https://cropchoiceai-inoor.streamlit.app/](https://cropchoiceai-inoor.streamlit.app/)**

---

## ✨ Key Features & Capabilities

### 1. 🤖 Precision Machine Learning Engine
- Multi-crop probability scoring across 22 distinct crop categories.
- Benchmarks 6 state-of-the-art classification algorithms using 5-fold cross-validation.
- Automatically selects and serializes the highest performing model (**Random Forest Classifier** with **100.00% accuracy** and **99.91% CV mean**).

### 2. 🩺 Soil Health & Nutrient Advisory System
- **NPK Imbalance Diagnostics:** Detects soil Nitrogen, Phosphorus, and Potassium deficiencies or toxicity.
- **pH Soil Reaction Classifier:** Classifies soil as Acidic (< 5.5), Neutral (5.5 - 7.5), or Alkaline (> 7.5).
- **Custom Fertilizer Action Plans:** Recommends targeted organic and inorganic soil amendments (e.g., Urea, Single Super Phosphate, Muriate of Potash, Agricultural Lime, Elemental Sulfur).

### 3. 🎨 Environmental Claymorphic Design System
- Soft 3D pillow cards (`.clay-card`), tactile 3D buttons, and recessed range slider tracks.
- Rich organic color palette featuring **Forest Soil Earth**, **Botanical Lush Green**, **Terracotta Clay**, and **Harvest Amber**.
- Quick soil & climate regional preset buttons (*Alluvial Rice Belt*, *Semi-Arid Cotton*, *Coffee Highlands*, *Fruit Orchard Loam*, *Acidic Soil*).

### 4. 📊 Interactive Analytics & Feature Importances
- Visualizes model performance comparisons and Gini feature importances.
- Identifies **Rainfall** and **Potassium (K)** as top decisive factors in crop suitability.

### 5. 📚 22-Crop Agronomic Catalog
- Searchable catalog providing target parameter distributions (temperature, rainfall, humidity, pH, NPK ratios) for all supported crops.

### ⏳ 2nd Evaluation Phase Reserved Module
- **Climate Sensitivity & Stress Simulator:** Reserved placeholder for the upcoming 2nd evaluation phase (+1°C to +5°C temperature shifts and ±50% precipitation modeling).

---

## 🏆 Machine Learning Model Benchmarks

Trained and evaluated on **2,200 records** split 80/20 train-test with 5-fold cross-validation:

| Algorithm | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score | 5-Fold CV Mean | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🏆 **Random Forest Classifier** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **99.91%** | **Selected Top Model** |
| 🟢 Support Vector Machine (SVC) | 100.00% | 100.00% | 100.00% | 100.00% | 99.86% | Benchmark |
| 🟢 Naive Bayes (GaussianNB) | 100.00% | 100.00% | 100.00% | 100.00% | 99.91% | Benchmark |
| 🟡 K-Nearest Neighbors (KNN) | 99.32% | 99.35% | 99.32% | 99.32% | 99.59% | Benchmark |
| 🟡 Gradient Boosting Classifier | 99.32% | 99.35% | 99.32% | 99.32% | 98.55% | Benchmark |
| 🟠 Decision Tree Classifier | 98.86% | 98.90% | 98.86% | 98.87% | 98.68% | Benchmark |

---

## 📊 Dataset Overview

- **Source:** Authentic Kaggle Crop Recommendation Dataset
- **Total Records:** 2,200 rows (100 samples per crop class)
- **Features (7 Numerical):**
  1. `N`: Nitrogen content ratio in soil (mg/kg)
  2. `P`: Phosphorus content ratio in soil (mg/kg)
  3. `K`: Potassium content ratio in soil (mg/kg)
  4. `temperature`: Air temperature in °C
  5. `humidity`: Relative humidity percentage (%)
  6. `ph`: Soil pH acidity/alkalinity scale (0 to 14)
  7. `rainfall`: Annual rainfall volume in mm
- **Target Classes (22 Crops):** `apple`, `banana`, `blackgram`, `chickpea`, `coconut`, `coffee`, `cotton`, `grapes`, `jute`, `kidneybeans`, `lentil`, `maize`, `mango`, `mothbeans`, `mungbean`, `muskmelon`, `orange`, `papaya`, `pigeonpeas`, `pomegranate`, `rice`, `watermelon`.

---

## 📁 Repository Directory Structure

```
Crop_Choice_AI/
├── app.py                       # Main Streamlit Cloud entry point (native Streamlit app)
├── streamlit_app.py             # Streamlit web application module
├── flask_app.py                 # Flask REST API server for local/API integrations
├── requirements.txt             # Python package dependencies for Streamlit Cloud
├── README.md                    # Project documentation
├── data/
│   └── Crop_recommendation.csv  # Kaggle agronomic dataset
├── models/
│   └── crop_intelligence_model.pkl # Trained Random Forest model & scaler artifact
├── src/
│   ├── generate_dataset.py      # Kaggle dataset generator script
│   ├── train_model.py           # ML training & 6-algorithm benchmarking pipeline
│   ├── predictor.py             # Crop predictor & soil health diagnostic engine
│   └── test_api.py              # Automated test suite
├── static/
│   ├── css/styles.css           # Claymorphism & environmental styling
│   └── js/main.js               # Interactive frontend JS logic
└── templates/
    └── index.html               # Responsive HTML5 Flask template
```

---

## 💻 Local Setup & Execution

### Prerequisites
- Python 3.10+ installed
- Git

### 1. Clone Repository & Install Dependencies
```bash
git clone https://github.com/qwerty274/Crop_Choice_AI.git
cd Crop_Choice_AI
pip install -r requirements.txt
```

### 2. Train & Benchmark ML Model Pipeline
```bash
python src/train_model.py
```

### 3. Option A: Run Streamlit Web Application (Recommended)
```bash
streamlit run app.py
```
*Navigates automatically to `http://localhost:8501`*

### 4. Option B: Run Flask REST API & Web Dashboard
```bash
python flask_app.py
```
*Navigates automatically to `http://127.0.0.1:5000`*

---

## 🌐 REST API Endpoints (Flask Server)

| Endpoint | Method | Description | Sample Output |
| :--- | :---: | :--- | :--- |
| `/api/health` | `GET` | System health check & active model metadata | `{"status":"online", "model":"Random Forest", "crops_supported":22}` |
| `/api/recommend` | `POST` | Generates Top-K crop recommendation & soil diagnosis | `{"primary_recommendation": "rice", "top_recommendations": [...]}` |
| `/api/analytics` | `GET` | Benchmark comparison matrix & feature importances | `{"best_model": "Random Forest", "benchmark_results": [...]}` |
| `/api/crops` | `GET` | Full 22-crop optimal statistical profiles catalog | `{"crop_stats": {...}, "crops": [...]}` |

---

## 🤝 Contributing & License

Contributions, issues, and feature requests are welcome!
Distributed under the **MIT License**.