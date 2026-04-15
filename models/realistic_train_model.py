"""
Realistic Training Model for BurnoutAI
Uses rule-based logic combined with ML for more accurate predictions
"""

import sys
import os
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings("ignore")

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "data" / "enhanced_burnout_dataset.csv"
MODEL_DIR = BASE_DIR / "models"
REALISTIC_MODEL_PATH = MODEL_DIR / "realistic_burnout_model.pkl"
REALISTIC_SCALER_PATH = MODEL_DIR / "realistic_scaler.pkl"
REALISTIC_META_PATH = MODEL_DIR / "realistic_model_meta.json"

# Feature columns
FEATURE_COLS = [
    "screen_time_hours", "work_hours", "sleep_hours", "break_frequency",
    "typing_speed_wpm", "backspace_rate", "pause_frequency",
    "app_switch_rate", "usage_type", "focus_session_count",
    "hydration_level", "physical_activity"
]

LABEL_MAP = {0: "Low", 1: "Medium", 2: "High"}
LABEL_COLORS = {0: "#22c55e", 1: "#f59e0b", 2: "#ef4444"}

def calculate_realistic_burnout_score(row):
    """Calculate burnout score based on real-world logic"""
    
    screen_time = row['screen_time_hours']
    work_hours = row['work_hours'] 
    sleep_hours = row['sleep_hours']
    break_freq = row['break_frequency']
    typing_speed = row['typing_speed_wpm']
    backspace_rate = row['backspace_rate']
    pause_freq = row['pause_frequency']
    app_switch_rate = row['app_switch_rate']
    focus_sessions = row['focus_session_count']
    hydration = row['hydration_level']
    physical_activity = row['physical_activity']
    
    score = 0
    
    # Screen time impact (major factor)
    if screen_time >= 12:
        score += 30
    elif screen_time >= 10:
        score += 25
    elif screen_time >= 8:
        score += 15
    elif screen_time >= 6:
        score += 5
    
    # Sleep impact (critical factor)
    if sleep_hours <= 4:
        score += 35
    elif sleep_hours <= 5:
        score += 25
    elif sleep_hours <= 6:
        score += 15
    elif sleep_hours <= 7:
        score += 5
    
    # Work hours impact
    if work_hours >= 12:
        score += 20
    elif work_hours >= 10:
        score += 15
    elif work_hours >= 8:
        score += 10
    
    # Break frequency (inverse relationship)
    if break_freq <= 1:
        score += 15
    elif break_freq <= 3:
        score += 10
    elif break_freq <= 5:
        score += 5
    
    # Cognitive strain indicators
    cognitive_strain = (pause_freq / 20) * 40 + (backspace_rate / 30) * 30 + (app_switch_rate / 60) * 30
    score += min(cognitive_strain, 20)
    
    # Focus sessions (inverse relationship)
    if focus_sessions <= 1:
        score += 10
    elif focus_sessions <= 3:
        score += 5
    
    # Hydration and physical activity (protective factors)
    if hydration <= 2:
        score += 10
    elif hydration <= 3:
        score += 5
    
    if physical_activity <= 15:
        score += 10
    elif physical_activity <= 30:
        score += 5
    
    return min(score, 100)

def create_realistic_dataset():
    """Create a dataset with realistic burnout logic"""
    
    print("Creating realistic dataset...")
    
    # Load the enhanced dataset
    df = pd.read_csv(DATA_PATH)
    
    # Calculate realistic burnout scores
    df['realistic_score'] = df.apply(calculate_realistic_burnout_score, axis=1)
    
    # Assign burnout levels based on realistic scores
    def assign_burnout_level(score):
        if score >= 70:
            return 2  # High
        elif score >= 40:
            return 1  # Medium  
        else:
            return 0  # Low
    
    df['burnout_level'] = df['realistic_score'].apply(assign_burnout_level)
    
    # Verify the distribution
    print("Realistic Burnout Distribution:")
    for level in [0, 1, 2]:
        count = (df['burnout_level'] == level).sum()
        percentage = count / len(df) * 100
        print(f"  {LABEL_MAP[level]}: {count} ({percentage:.1f}%)")
    
    return df

def train_realistic_model():
    """Train a model that follows realistic burnout logic"""
    
    print("=" * 80)
    print("  REALISTIC BURNOUT DETECTION MODEL TRAINING")
    print("=" * 80)
    
    # Create realistic dataset
    df = create_realistic_dataset()
    
    # Prepare features and target
    X = df[FEATURE_COLS].values
    y = df['burnout_level'].values
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nModel Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\nClassification Report:")
    target_names = [LABEL_MAP[i] for i in sorted(LABEL_MAP.keys())]
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Test extreme cases
    print("\n=== Testing Extreme Cases ===")
    
    # Case 1: Extreme burnout (12hr screen, 4.5hr sleep)
    extreme_case = np.array([[
        12, 8, 4.5, 1, 40, 20, 15, 45, 2, 1, 1, 5
    ]])
    extreme_scaled = scaler.transform(extreme_case)
    extreme_pred = model.predict(extreme_scaled)[0]
    extreme_proba = model.predict_proba(extreme_scaled)[0]
    
    print(f"Extreme case (12hr screen, 4.5hr sleep): {LABEL_MAP[extreme_pred]}")
    print(f"Probabilities: {dict(zip([LABEL_MAP[i] for i in range(3)], [f'{p:.1%}' for p in extreme_proba]))}")
    
    # Case 2: Healthy case (5hr screen, 8hr sleep)
    healthy_case = np.array([[
        5, 6, 8, 6, 75, 5, 3, 8, 0, 5, 4, 60
    ]])
    healthy_scaled = scaler.transform(healthy_case)
    healthy_pred = model.predict(healthy_scaled)[0]
    healthy_proba = model.predict_proba(healthy_scaled)[0]
    
    print(f"Healthy case (5hr screen, 8hr sleep): {LABEL_MAP[healthy_pred]}")
    print(f"Probabilities: {dict(zip([LABEL_MAP[i] for i in range(3)], [f'{p:.1%}' for p in healthy_proba]))}")
    
    # Save artifacts
    MODEL_DIR.mkdir(exist_ok=True)
    with open(REALISTIC_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(REALISTIC_SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    
    # Save metadata
    meta = {
        "model_name": "Realistic Random Forest",
        "test_accuracy": round(accuracy, 4),
        "feature_cols": FEATURE_COLS,
        "label_map": {str(k): v for k, v in LABEL_MAP.items()},
        "label_colors": {str(k): v for k, v in LABEL_COLORS.items()},
        "dataset_info": {
            "samples": len(df),
            "features": len(FEATURE_COLS),
            "classes": len(np.unique(y))
        }
    }
    
    with open(REALISTIC_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"\nRealistic artifacts saved:")
    print(f"   Model   -> {REALISTIC_MODEL_PATH}")
    print(f"   Scaler  -> {REALISTIC_SCALER_PATH}")
    print(f"   Metadata-> {REALISTIC_META_PATH}")
    print("=" * 80)
    
    return accuracy

if __name__ == "__main__":
    train_realistic_model()
