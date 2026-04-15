"""
================================================================================
 AI-Based Digital Burnout & Cognitive Load Detection System
 File: app.py
 Purpose: Flask web application — routes, API endpoints, and model integration
================================================================================
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

from flask import Flask, render_template, request, jsonify, redirect

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from models.realistic_train_model import train_realistic_model, REALISTIC_MODEL_PATH, REALISTIC_SCALER_PATH, MODEL_DIR
from models.train_model import predict_burnout, load_artifacts, DATA_PATH, MODEL_PATH
from utils.suggestions import generate_suggestions, generate_ai_report

app = Flask(__name__)


# ── Auto-bootstrap: generate dataset + train realistic model if artifacts missing ───────
def bootstrap():
    """Ensure dataset and realistic trained model exist before serving requests."""
    # Generate enhanced dataset if missing (needed for realistic model)
    from models.enhanced_train_model import ENHANCED_DATA_PATH
    if not ENHANCED_DATA_PATH.exists():
        print("[BOOTSTRAP] Generating enhanced dataset ")
        from data.enhanced_dataset import generate_enhanced_dataset, add_derived_features
        df = generate_enhanced_dataset(5000)
        df = add_derived_features(df)
        df.to_csv(ENHANCED_DATA_PATH, index=False)
        print(f"[BOOTSTRAP] Enhanced dataset saved  {ENHANCED_DATA_PATH}")

    # Train realistic model if missing
    if not REALISTIC_MODEL_PATH.exists():
        print("[BOOTSTRAP] Training realistic model ")
        train_realistic_model()
        print("[BOOTSTRAP] Realistic model ready.")
    
    # Fallback to original model if realistic not available
    elif not MODEL_PATH.exists():
        print("[BOOTSTRAP] Training original model as fallback ")
        from models.train_model import train
        train()
        print("[BOOTSTRAP] Original model ready.")


# ──# Realistic prediction function
def predict_burnout_realistic(raw: dict) -> dict:
    """Realistic prediction using the rule-based model"""
    try:
        import pickle
        import numpy as np
        from models.realistic_train_model import FEATURE_COLS
        
        # Load realistic model and scaler
        with open(REALISTIC_MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(REALISTIC_SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        with open(MODEL_DIR / "realistic_model_meta.json", "r") as f:
            meta = json.load(f)
        
        # Create feature vector in correct order
        feature_vector = []
        for col in FEATURE_COLS:
            feature_vector.append(raw[col])
        
        X = np.array([feature_vector])
        
        # Scale and predict
        X_scaled = scaler.transform(X)
        
        label_int = int(model.predict(X_scaled)[0])
        proba = model.predict_proba(X_scaled)[0].tolist()
        label_name = meta["label_map"][str(label_int)]
        label_color = meta["label_colors"][str(label_int)]
        
        # Realistic cognitive load calculation
        cognitive_load = round(
            (raw["pause_frequency"] / 20) * 40
            + (raw["backspace_rate"] / 30) * 25
            + max(0, (80 - raw["typing_speed_wpm"]) / 80) * 20
            + (raw["app_switch_rate"] / 60) * 15,
            1
        )
        cognitive_load = min(cognitive_load, 100)
        
        # Realistic productivity score
        productivity = round(
            max(0, 100
                - cognitive_load * 0.4
                - (raw["app_switch_rate"] / 60) * 20
                + (raw["focus_session_count"] / 8) * 20
                + (raw["break_frequency"] / 10) * 10
            ),
            1
        )
        
        # Distraction index
        distraction = round((raw["app_switch_rate"] / 60) * 100, 1)
        
        return {
            "burnout_level": label_name,
            "burnout_int": label_int,
            "burnout_color": label_color,
            "probabilities": {meta["label_map"][str(i)]: round(p * 100, 1) for i, p in enumerate(proba)},
            "cognitive_load": cognitive_load,
            "productivity": productivity,
            "distraction_index": distraction,
            "model_accuracy": meta["test_accuracy"],
            "model_type": "realistic"
        }
        
    except Exception as e:
        print(f"Realistic prediction failed: {e}")
        # Fallback to original prediction
        return predict_burnout(raw)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def root():
    """Redirect root to login."""
    return redirect("/login")

@app.route("/login")
def login():
    """Serve the login page."""
    return render_template("login.html")

@app.route("/dashboard")
def index():
    """Serve the main dashboard page."""
    _, _, meta = load_artifacts()
    return render_template("index.html", model_accuracy=round(meta["test_accuracy"] * 100, 2))


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST endpoint that accepts form data, runs prediction pipeline,
    generates suggestions, and returns a JSON response.
    """
    try:
        # Parse & validate inputs
        raw = {
            "screen_time_hours":  float(request.form.get("screen_time", 6)),
            "work_hours":         float(request.form.get("work_hours", 4)),
            "sleep_hours":        float(request.form.get("sleep_hours", 7)),
            "break_frequency":    int(request.form.get("break_frequency", 3)),
            "typing_speed_wpm":   float(request.form.get("typing_speed", 60)),
            "backspace_rate":     float(request.form.get("backspace_rate", 8)),
            "pause_frequency":    float(request.form.get("pause_frequency", 5)),
            "app_switch_rate":    float(request.form.get("app_switch_rate", 15)),
            "usage_type":         int(request.form.get("usage_type", 2)),
            "focus_session_count":int(request.form.get("focus_sessions", 3)),
            "hydration_level":    int(request.form.get("hydration", 3)),
            "physical_activity":  float(request.form.get("physical_activity", 30)),
        }

        # Clamp values to safe ranges
        raw["screen_time_hours"]  = max(0, min(raw["screen_time_hours"], 24))
        raw["sleep_hours"]        = max(0, min(raw["sleep_hours"], 12))
        raw["typing_speed_wpm"]   = max(1, min(raw["typing_speed_wpm"], 200))
        raw["backspace_rate"]     = max(0, min(raw["backspace_rate"], 50))

        # Run prediction with realistic model if available
        if REALISTIC_MODEL_PATH.exists():
            prediction = predict_burnout_realistic(raw)
        else:
            prediction = predict_burnout(raw)
        suggestions = generate_suggestions(raw, prediction)
        report      = generate_ai_report(raw, prediction, suggestions)

        response = {
            "status":       "success",
            "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M"),
            "prediction":   prediction,
            "suggestions":  suggestions,
            "report":       report,
            "input_echo":   raw,
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/model-info")
def model_info():
    """Return model metadata for display."""
    try:
        _, _, meta = load_artifacts()
        return jsonify(meta)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/sample-data")
def sample_data():
    """Return a few pre-canned scenarios for demo purposes."""
    scenarios = [
        {
            "label": "",
            "data": dict(screen_time=5, work_hours=4, sleep_hours=8, break_frequency=6,
                         typing_speed=75, backspace_rate=5, pause_frequency=3,
                         app_switch_rate=8, usage_type=0, focus_sessions=5,
                         hydration=4, physical_activity=60)
        },
        {
            "label": "",
            "data": dict(screen_time=9, work_hours=8, sleep_hours=6, break_frequency=2,
                         typing_speed=55, backspace_rate=14, pause_frequency=9,
                         app_switch_rate=30, usage_type=2, focus_sessions=3,
                         hydration=2, physical_activity=10)
        },
        {
            "label": "",
            "data": dict(screen_time=14, work_hours=11, sleep_hours=4, break_frequency=0,
                         typing_speed=35, backspace_rate=25, pause_frequency=17,
                         app_switch_rate=55, usage_type=1, focus_sessions=0,
                         hydration=1, physical_activity=0)
        },
    ]
    return jsonify(scenarios)


@app.route("/help")
def help():
    """Serve the help page."""
    return render_template("help.html")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    bootstrap()
    print("\n Starting Burnout Detection Server …")
    print("   Open: http://127.0.0.1:5000\n")
    app.run(debug=True, port=5000)
