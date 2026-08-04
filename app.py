"""
Crop Choice Intelligence - Dual Server Entry (Flask & Streamlit Cloud Support)
"""

import sys
import os

# Check if running under Streamlit (e.g. `streamlit run app.py` on Streamlit Cloud)
try:
    import streamlit as st
    is_streamlit_run = st.runtime.exists() if hasattr(st, 'runtime') else True
except Exception:
    is_streamlit_run = False

if is_streamlit_run:
    # Forward execution to streamlit_app.py
    import streamlit_app
else:
    # Flask Web Server Mode
    sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
    from flask import Flask, render_template, request, jsonify
    from predictor import CropPredictor
    from train_model import train_and_evaluate

    app = Flask(__name__)
    predictor = CropPredictor()

    def ensure_model_ready():
        if not predictor.loaded:
            print("Model artifact not found. Executing ML training pipeline...")
            train_and_evaluate()
            predictor._load_model()

    @app.route('/')
    def index():
        ensure_model_ready()
        return render_template('index.html')

    @app.route('/api/health', methods=['GET'])
    def health():
        ensure_model_ready()
        return jsonify({
            'status': 'online',
            'model': predictor.best_model_name if predictor.loaded else 'Not Loaded',
            'crops_supported': len(predictor.crop_classes) if predictor.loaded else 0
        })

    @app.route('/api/recommend', methods=['POST'])
    def recommend():
        ensure_model_ready()
        try:
            data = request.json or {}
            required_fields = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
            inputs = {f: float(data[f]) for f in required_fields if f in data}
            if len(inputs) < len(required_fields):
                return jsonify({'error': 'Missing required parameter(s)'}), 400
                
            top_k = int(data.get('top_k', 3))
            result = predictor.predict(inputs, top_k=top_k)
            return jsonify({'success': True, 'data': result})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/analytics', methods=['GET'])
    def analytics():
        ensure_model_ready()
        try:
            return jsonify({
                'success': True,
                'data': {
                    'best_model': predictor.best_model_name,
                    'benchmark_results': predictor.benchmark_results,
                    'feature_importances': predictor.feature_importances,
                    'total_crops': len(predictor.crop_classes),
                    'crop_classes': predictor.crop_classes
                }
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/crops', methods=['GET'])
    def crops_catalog():
        ensure_model_ready()
        try:
            return jsonify({
                'success': True,
                'data': {
                    'crop_stats': predictor.crop_stats,
                    'crops': predictor.crop_classes
                }
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/simulate', methods=['POST'])
    def simulate():
        ensure_model_ready()
        try:
            data = request.json or {}
            inputs = {
                'N': float(data.get('N', 50)),
                'P': float(data.get('P', 50)),
                'K': float(data.get('K', 50)),
                'temperature': float(data.get('temperature', 25)),
                'humidity': float(data.get('humidity', 70)),
                'ph': float(data.get('ph', 6.5)),
                'rainfall': float(data.get('rainfall', 100))
            }
            temp_shift = float(data.get('temp_shift', 0.0))
            rainfall_shift_pct = float(data.get('rainfall_shift_pct', 0.0))
            
            sim_result = predictor.simulate_climate_change(inputs, temp_shift=temp_shift, rainfall_shift_pct=rainfall_shift_pct)
            return jsonify({'success': True, 'data': sim_result})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    if __name__ == '__main__':
        ensure_model_ready()
        print("\n========================================================")
        print("  Crop Choice Intelligence Web App Running on http://127.0.0.1:5000")
        print("========================================================\n")
        app.run(host='127.0.0.1', port=5000, debug=True)
