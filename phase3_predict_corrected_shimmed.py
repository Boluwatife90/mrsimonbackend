import joblib
import os
import traceback

# 1. Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILENAME = 'stacked_ensemble_model.pkl'
MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILENAME)

stacked_model = None
scaler = None  # We are bypassing the scaler to avoid the corrupted file error

# 2. Attempt to load the model
try:
    print(f"🔍 Looking for model at: {MODEL_PATH}")
    if os.path.exists(MODEL_PATH):
        stacked_model = joblib.load(MODEL_PATH)
        print("✅ Model loaded successfully!")
    else:
        print(f"❌ ERROR: Model file not found at {MODEL_PATH}")
except Exception as e:
    print(f"❌ Detailed Error loading model:")
    traceback.print_exc()

# 3. Define the feature order (MUST match your training data columns exactly)
FEATURE_ORDER = [
    "soil_moisture", "EC", "N", "P", "K", "soil_temp", "pH", 
    "air_temp", "humidity", "rainfall", "ndvi_proxy", "growth_stage_encoded"
]

# 4. The prediction function
def predict_with_experts(payload, need_threshold=0.45):
    if stacked_model is None:
        raise Exception("Model failed to load. Check server terminal for details.")

    # Extract features in the exact order the model expects
    input_data = [[float(payload.get(feat, 0.0)) for feat in FEATURE_ORDER]]
    
    # Make prediction
    try:
        # Try standard predict_proba first
        probabilities = stacked_model.predict_proba(input_data)[0]
        need_proba = float(probabilities[1]) # Probability of class 1
    except AttributeError:
        # Fallback if model only has .predict() and not .predict_proba()
        need_pred = stacked_model.predict(input_data)[0]
        need_proba = 0.9 if need_pred == 1 else 0.1
        
    need_label = 1 if need_proba >= need_threshold else 0
    
    # Set a default rate (you can adjust this logic if your model predicts rate directly)
    rate_pred = 100.0 if need_label == 1 else 0.0
    
    return {
        "need_label": int(need_label),
        "need_proba": round(need_proba, 4),
        "rate_pred": round(float(rate_pred), 1),
        "timing": "24_48h" if need_label == 1 else "N/A",
        "expert": {
            "base": {
                "ts_pred_soil_moisture": float(payload.get('soil_moisture', 0)),
                "base_rate_raw": float(rate_pred),
                "base_need_proba": round(need_proba, 4)
            }
        }
    }