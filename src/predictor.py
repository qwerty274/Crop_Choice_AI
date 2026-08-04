"""
Crop Predictor & Soil Health Intelligence Engine
Provides inference, top-k recommendations with probability distributions,
soil nutrient imbalance analyzer, and customized agronomic advice.
"""

import os
import joblib
import numpy as np
import pandas as pd

MODEL_PATH = os.path.join('models', 'crop_intelligence_model.pkl')

class CropPredictor:
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.loaded = False
        self._load_model()
        
    def _load_model(self):
        if os.path.exists(self.model_path):
            artifact = joblib.load(self.model_path)
            self.model = artifact['model']
            self.rf_model = artifact['rf_model']
            self.scaler = artifact['scaler']
            self.best_model_name = artifact['best_model_name']
            self.feature_names = artifact['feature_names']
            self.benchmark_results = artifact['benchmark_results']
            self.feature_importances = artifact['feature_importances']
            self.crop_stats = artifact['crop_stats']
            self.crop_classes = artifact['crop_classes']
            self.loaded = True
        else:
            self.loaded = False
            
    def predict(self, input_features, top_k=3):
        """
        input_features: dict containing 'N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'
        """
        if not self.loaded:
            self._load_model()
            if not self.loaded:
                raise FileNotFoundError(f"Model file not found at {self.model_path}. Please train the model first.")
                
        # Format input array
        df_input = pd.DataFrame([input_features])[self.feature_names]
        
        # Determine probability using random forest or active best model
        if hasattr(self.rf_model, 'predict_proba'):
            probabilities = self.rf_model.predict_proba(df_input)[0]
            classes = self.rf_model.classes_
        else:
            # Fallback
            scaled = self.scaler.transform(df_input)
            probabilities = self.model.predict_proba(scaled)[0]
            classes = self.model.classes_
            
        top_indices = np.argsort(probabilities)[::-1][:top_k]
        
        recommendations = []
        for idx in top_indices:
            crop_name = classes[idx]
            prob_percent = round(probabilities[idx] * 100, 2)
            
            # Fetch crop statistics & suitability details
            stats = self.crop_stats.get(crop_name, {})
            
            recommendations.append({
                'crop': crop_name,
                'confidence': prob_percent,
                'ideal_stats': stats
            })
            
        primary_crop = recommendations[0]['crop']
        
        # Soil & Agronomic Health Diagnosis
        soil_analysis = self._analyze_soil_health(input_features, primary_crop)
        
        return {
            'primary_recommendation': primary_crop,
            'top_recommendations': recommendations,
            'soil_health_analysis': soil_analysis,
            'input_summary': input_features
        }
        
    def _analyze_soil_health(self, input_data, recommended_crop):
        N = input_data['N']
        P = input_data['P']
        K = input_data['K']
        ph = input_data['ph']
        temp = input_data['temperature']
        humidity = input_data['humidity']
        rainfall = input_data['rainfall']
        
        ideal = self.crop_stats.get(recommended_crop, {})
        
        diagnoses = []
        fertilizer_advice = []
        
        # Nitrogen Check
        if N < 30:
            diagnoses.append("Soil Nitrogen (N) levels are low (< 30 mg/kg).")
            fertilizer_advice.append("Apply Nitrogen-rich fertilizers like Urea or Ammonium Nitrate, or incorporate leguminous cover crops.")
        elif N > 100:
            diagnoses.append("Soil Nitrogen (N) is high (> 100 mg/kg).")
            fertilizer_advice.append("Avoid excess Nitrogen inputs to prevent vegetative overgrowth and leaching.")
        else:
            diagnoses.append("Nitrogen (N) level is in optimal range.")
            
        # Phosphorus Check
        if P < 25:
            diagnoses.append("Phosphorus (P) is low (< 25 mg/kg). Root development may be stunted.")
            fertilizer_advice.append("Apply Single Super Phosphate (SSP) or Diammonium Phosphate (DAP).")
        elif P > 90:
            diagnoses.append("Phosphorus (P) is elevated (> 90 mg/kg).")
        else:
            diagnoses.append("Phosphorus (P) level is balanced.")
            
        # Potassium Check
        if K < 25:
            diagnoses.append("Potassium (K) is low (< 25 mg/kg), risking disease resistance.")
            fertilizer_advice.append("Apply Muriate of Potash (MOP) or Potassium Sulfate.")
        elif K > 180:
            diagnoses.append("Potassium (K) is high (> 180 mg/kg), suitable for potassium-hungry crops like Grapes or Apple.")
        else:
            diagnoses.append("Potassium (K) level is healthy.")
            
        # pH Check
        ph_status = "Neutral"
        if ph < 5.5:
            ph_status = "Strongly Acidic"
            diagnoses.append(f"Soil pH ({ph}) is acidic.")
            fertilizer_advice.append("Apply agricultural lime (calcium carbonate) or dolomite to neutralize acidity.")
        elif ph > 7.5:
            ph_status = "Alkaline"
            diagnoses.append(f"Soil pH ({ph}) is alkaline.")
            fertilizer_advice.append("Apply elemental sulfur or organic compost to gradually reduce pH.")
        else:
            diagnoses.append(f"Soil pH ({ph}) is ideal (neutral).")
            
        if not fertilizer_advice:
            fertilizer_advice.append("Soil nutrient balance is optimal! Use standard organic compost for maintenance.")
            
        return {
            'ph_status': ph_status,
            'diagnoses': diagnoses,
            'fertilizer_recommendations': fertilizer_advice,
            'npk_ratio': f"N:{round(N)} | P:{round(P)} | K:{round(K)}"
        }

    def simulate_climate_change(self, input_features, temp_shift=0.0, rainfall_shift_pct=0.0):
        """
        Simulates crop recommendation sensitivity under climate changes (+/- temp, +/- rainfall %)
        """
        simulated = dict(input_features)
        simulated['temperature'] = round(simulated['temperature'] + temp_shift, 2)
        simulated['rainfall'] = round(simulated['rainfall'] * (1 + rainfall_shift_pct / 100.0), 2)
        
        result = self.predict(simulated, top_k=3)
        result['simulation_params'] = {
            'temp_shift': temp_shift,
            'rainfall_shift_pct': rainfall_shift_pct,
            'simulated_temperature': simulated['temperature'],
            'simulated_rainfall': simulated['rainfall']
        }
        return result
