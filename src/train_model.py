"""
Crop Choice Intelligence Model Training & Benchmarking Pipeline
Trains multiple classification models on Kaggle crop dataset, compares performance metrics,
evaluates feature importances, and saves the optimized model artifact.
"""

import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from generate_dataset import generate_dataset

DATA_PATH = os.path.join('data', 'Crop_recommendation.csv')
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'crop_intelligence_model.pkl')

def train_and_evaluate():
    print("=== Crop Choice Intelligence: Pipeline Initiation ===")
    
    # 1. Dataset Verification / Generation
    if not os.path.exists(DATA_PATH):
        print(f"Dataset missing at {DATA_PATH}. Generating default Kaggle crop recommendation dataset...")
        generate_dataset(samples_per_crop=100, output_file=DATA_PATH)
        
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns.")
    print(f"Crop classes ({df['label'].nunique()} total): {sorted(df['label'].unique())}")
    
    # 2. Data Splitting & Preprocessing
    X = df.drop('label', axis=1)
    y = df['label']
    
    feature_names = list(X.columns)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Model Benchmarking Suite
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Support Vector Machine': SVC(probability=True, random_state=42),
        'Naive Bayes': GaussianNB(),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
    }
    
    benchmark_results = []
    trained_models = {}
    
    print("\n--- Model Performance Comparison ---")
    for name, clf in models.items():
        # Train scaled or non-scaled based on algorithm preference
        if name in ['Support Vector Machine', 'K-Nearest Neighbors']:
            clf.fit(X_train_scaled, y_train)
            preds = clf.predict(X_test_scaled)
            cv_scores = cross_val_score(clf, scaler.transform(X), y, cv=5)
        else:
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            cv_scores = cross_val_score(clf, X, y, cv=5)
            
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average='macro', zero_division=0)
        rec = recall_score(y_test, preds, average='macro', zero_division=0)
        f1 = f1_score(y_test, preds, average='macro', zero_division=0)
        cv_mean = cv_scores.mean()
        
        benchmark_results.append({
            'model_name': name,
            'accuracy': round(acc * 100, 2),
            'precision': round(prec * 100, 2),
            'recall': round(rec * 100, 2),
            'f1_score': round(f1 * 100, 2),
            'cv_mean': round(cv_mean * 100, 2)
        })
        
        trained_models[name] = clf
        print(f"[{name}] Accuracy: {acc*100:.2f}% | F1-Score: {f1*100:.2f}% | 5-Fold CV: {cv_mean*100:.2f}%")
        
    # Sort benchmark results by accuracy
    benchmark_df = pd.DataFrame(benchmark_results).sort_values(by='accuracy', ascending=False)
    best_model_name = benchmark_df.iloc[0]['model_name']
    best_model = trained_models[best_model_name]
    
    print(f"\n>>> Selected Best Performing Model: '{best_model_name}' (Accuracy: {benchmark_df.iloc[0]['accuracy']}%)")
    
    # 4. Feature Importances (from Random Forest)
    rf_model = trained_models['Random Forest']
    feature_importances = dict(zip(feature_names, rf_model.feature_importances_))
    # Sort feature importances
    feature_importances = dict(sorted(feature_importances.items(), key=lambda item: item[1], reverse=True))
    
    # 5. Dataset Summary Stats per Crop
    crop_stats = {}
    for crop in df['label'].unique():
        sub_df = df[df['label'] == crop]
        crop_stats[crop] = {
            'N': {'mean': round(sub_df['N'].mean(), 1), 'min': round(sub_df['N'].min(), 1), 'max': round(sub_df['N'].max(), 1)},
            'P': {'mean': round(sub_df['P'].mean(), 1), 'min': round(sub_df['P'].min(), 1), 'max': round(sub_df['P'].max(), 1)},
            'K': {'mean': round(sub_df['K'].mean(), 1), 'min': round(sub_df['K'].min(), 1), 'max': round(sub_df['K'].max(), 1)},
            'temperature': {'mean': round(sub_df['temperature'].mean(), 1), 'min': round(sub_df['temperature'].min(), 1), 'max': round(sub_df['temperature'].max(), 1)},
            'humidity': {'mean': round(sub_df['humidity'].mean(), 1), 'min': round(sub_df['humidity'].min(), 1), 'max': round(sub_df['humidity'].max(), 1)},
            'ph': {'mean': round(sub_df['ph'].mean(), 1), 'min': round(sub_df['ph'].min(), 1), 'max': round(sub_df['ph'].max(), 1)},
            'rainfall': {'mean': round(sub_df['rainfall'].mean(), 1), 'min': round(sub_df['rainfall'].min(), 1), 'max': round(sub_df['rainfall'].max(), 1)}
        }
        
    # 6. Save Model Artifacts & Metadata
    os.makedirs(MODEL_DIR, exist_ok=True)
    artifact = {
        'model': best_model,
        'rf_model': rf_model,
        'scaler': scaler,
        'best_model_name': best_model_name,
        'feature_names': feature_names,
        'benchmark_results': benchmark_results,
        'feature_importances': feature_importances,
        'crop_stats': crop_stats,
        'crop_classes': sorted(list(df['label'].unique()))
    }
    
    joblib.dump(artifact, MODEL_PATH)
    print(f"Model artifact successfully saved to '{MODEL_PATH}'.")
    return artifact

if __name__ == '__main__':
    train_and_evaluate()
