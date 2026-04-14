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

from models.train_model import predict_burnout, load_artifacts, DATA_PATH, MODEL_PATH
from utils.suggestions import generate_suggestions, generate_ai_report

app = Flask(__name__)


# ── Auto-bootstrap: generate dataset + train model if artifacts missing ───────
def bootstrap():
    """Ensure dataset and trained model exist before serving requests."""
    if not DATA_PATH.exists():
        print("[BOOTSTRAP] Generating dataset …")
        from data.generate_dataset import generate_dataset
        df = generate_dataset()
        df.to_csv(DATA_PATH, index=False)
        print(f"[BOOTSTRAP] Dataset saved → {DATA_PATH}")

    if not MODEL_PATH.exists():
        print("[BOOTSTRAP] Training model …")
        from models.train_model import train
        train()
        print("[BOOTSTRAP] Model ready.")


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

        # Run prediction
        prediction  = predict_burnout(raw)
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
            "label": "🟢 Healthy Student",
            "data": dict(screen_time=5, work_hours=4, sleep_hours=8, break_frequency=6,
                         typing_speed=75, backspace_rate=5, pause_frequency=3,
                         app_switch_rate=8, usage_type=0, focus_sessions=5,
                         hydration=4, physical_activity=60)
        },
        {
            "label": "🟡 Overworked Professional",
            "data": dict(screen_time=9, work_hours=8, sleep_hours=6, break_frequency=2,
                         typing_speed=55, backspace_rate=14, pause_frequency=9,
                         app_switch_rate=30, usage_type=2, focus_sessions=3,
                         hydration=2, physical_activity=10)
        },
        {
            "label": "🔴 Severely Burned Out",
            "data": dict(screen_time=14, work_hours=11, sleep_hours=4, break_frequency=0,
                         typing_speed=35, backspace_rate=25, pause_frequency=17,
                         app_switch_rate=55, usage_type=1, focus_sessions=0,
                         hydration=1, physical_activity=0)
        },
    ]
    return jsonify(scenarios)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    bootstrap()
    print("\n🚀 Starting Burnout Detection Server …")
    print("   Open: http://127.0.0.1:5000\n")
    app.run(debug=True, port=5000)
