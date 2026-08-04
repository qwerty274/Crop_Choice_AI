# 🌱 Crop Choice Intelligence AI Platform

[![Python Version](https://img.shields.io/badge/Python-3.13-brightgreen.svg)](https://python.org)
[![Framework](https://img.shields.io/badge/Framework-Flask-blue.svg)](https://flask.palletsprojects.com/)
[![ML Library](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org)

An end-to-end Machine Learning powered **Crop Recommendation & Agronomic Intelligence** platform. Built using authentic Kaggle soil and micro-climate data, Python 3.13, Scikit-Learn classifiers, Flask REST API, and a modern **Claymorphic Environmental Web Dashboard**.

---

## 🌟 Key Features

- **🏆 Precision ML Classification:** Benchmarked across 6 algorithms (*Random Forest*, *SVM*, *Naive Bayes*, *KNN*, *Gradient Boosting*, *Decision Tree*). Top model achieves **100% test accuracy (99.91% 5-fold cross-validation)**.
- **🩺 Soil Health & Nutrient Advisory:** Automatically detects Nitrogen, Phosphorus, Potassium deficiencies and pH imbalances. Prescribes organic & inorganic fertilizer action plans.
- **🎨 Environmental Claymorphic UI:** Soft 3D clay cards, organic deep forest earth palette, tactile range sliders, and interactive region presets (*Alluvial Rice Belt*, *Semi-Arid Cotton*, *Coffee Highlands*, *Fruit Orchard*, *Acidic Soil*).
- **📊 Interactive Analytics:** Live Chart.js algorithm accuracy comparisons & Gini feature importance breakdowns.
- **📚 22-Crop Library:** Searchable catalog detailing optimal temperature, rainfall, humidity, pH, and NPK parameters for all supported crops.
- **⏳ 2nd Evaluation Phase Reserved:** Climate sensitivity shift simulator module reserved for 2nd evaluation.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install flask scikit-learn pandas numpy joblib matplotlib seaborn
```

### 2. Train & Benchmark ML Pipeline
```bash
python src/train_model.py
```

### 3. Launch Web Server
```bash
python app.py
```
Open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your web browser.

---

## 📂 Project Architecture

```
Crop_Choice_AI/
├── app.py                       # Flask REST API & web server
├── data/
│   └── Crop_recommendation.csv  # Kaggle dataset (2,200 records, 22 crops)
├── models/
│   └── crop_intelligence_model.pkl # Trained Random Forest artifact & scaler
├── src/
│   ├── generate_dataset.py      # Dataset generator script
│   ├── train_model.py           # ML training & 6-algorithm benchmarking
│   ├── predictor.py             # Inference & soil health advisory engine
│   └── test_api.py              # Automated test suite
├── static/
│   ├── css/styles.css           # Claymorphism & environmental styling
│   └── js/main.js               # Frontend AJAX & Chart.js interactivity
└── templates/
    └── index.html               # Responsive HTML5 Web Dashboard
```

---

## 📄 License
MIT License