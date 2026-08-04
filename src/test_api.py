"""
Verification script for Crop Choice Intelligence Predictor Engine & APIs
"""
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__)))

from predictor import CropPredictor

def test_inference():
    print("--- Verifying CropPredictor Engine ---")
    predictor = CropPredictor()
    assert predictor.loaded, "Model should be loaded successfully"
    print(f"Loaded model: {predictor.best_model_name}")
    
    # Test 1: Rice parameters (high rainfall, high humidity, N=90)
    rice_input = {'N': 90, 'P': 48, 'K': 40, 'temperature': 24.0, 'humidity': 82.0, 'ph': 6.5, 'rainfall': 240.0}
    res_rice = predictor.predict(rice_input, top_k=3)
    print("Test 1 (Rice parameters):")
    print(f"  Primary Recommendation: {res_rice['primary_recommendation']}")
    print(f"  Top recommendations: {[r['crop'] + ' (' + str(r['confidence']) + '%)' for r in res_rice['top_recommendations']]}")
    print(f"  Soil NPK: {res_rice['soil_health_analysis']['npk_ratio']}")
    assert res_rice['primary_recommendation'] == 'rice', "Expected rice for rice input"
    
    # Test 2: Apple parameters (high K, high P, cool temp, low rainfall)
    apple_input = {'N': 20, 'P': 130, 'K': 200, 'temperature': 22.0, 'humidity': 92.0, 'ph': 6.0, 'rainfall': 110.0}
    res_apple = predictor.predict(apple_input, top_k=3)
    print("\nTest 2 (Apple/Grapes parameters):")
    print(f"  Primary Recommendation: {res_apple['primary_recommendation']}")
    print(f"  Top recommendations: {[r['crop'] + ' (' + str(r['confidence']) + '%)' for r in res_apple['top_recommendations']]}")
    
    # Test 3: Climate change simulation (+2C, -20% rain)
    sim_res = predictor.simulate_climate_change(rice_input, temp_shift=2.0, rainfall_shift_pct=-20.0)
    print("\nTest 3 (Climate Shift Simulation):")
    print(f"  Simulated Temp: {sim_res['simulation_params']['simulated_temperature']} C")
    print(f"  Simulated Rainfall: {sim_res['simulation_params']['simulated_rainfall']} mm")
    print(f"  New Primary Crop: {sim_res['primary_recommendation']}")
    
    print("\n[SUCCESS] All Predictor Engine Verification Tests Passed Successfully!")

if __name__ == '__main__':
    test_inference()
