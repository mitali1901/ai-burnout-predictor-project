"""
================================================================================
 AI-Based Digital Burnout & Cognitive Load Detection System
 File: models/train_model.py
 Purpose: Preprocess data, train ML models, evaluate, and persist artifacts
================================================================================
"""

import sys
import os
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent
DATA_PATH  = BASE_DIR / "data" / "burnout_dataset.csv"
MODEL_DIR  = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "burnout_model.pkl"
SCALER_PATH= MODEL_DIR / "scaler.pkl"
META_PATH  = MODEL_DIR / "model_meta.json"

# ── Feature columns used for training ────────────────────────────────────────
FEATURE_COLS = [
    "screen_time_hours", "work_hours", "sleep_hours", "break_frequency",
    "typing_speed_wpm", "backspace_rate", "pause_frequency",
    "app_switch_rate", "usage_type", "focus_session_count",
    "hydration_level", "physical_activity",
]
TARGET_COL   = "burnout_level"
LABEL_MAP    = {0: "Low", 1: "Medium", 2: "High"}
LABEL_COLORS = {0: "#22c55e", 1: "#f59e0b", 2: "#ef4444"}  # green / amber / red


# ── Preprocessing ─────────────────────────────────────────────────────────────
def load_and_preprocess(path: Path):
    """Load CSV, engineer features, split into X / y."""
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns.")

    # Derived features that improve signal
    df["work_to_break_ratio"]  = df["work_hours"] / (df["break_frequency"] + 1)
    df["sleep_deficit"]        = np.clip(8 - df["sleep_hours"], 0, 8)
    df["distraction_index"]    = (df["app_switch_rate"] / 60) * 100
    df["typing_efficiency"]    = df["typing_speed_wpm"] / (df["backspace_rate"] + 1)

    feature_cols_extended = FEATURE_COLS + [
        "work_to_break_ratio", "sleep_deficit",
        "distraction_index", "typing_efficiency",
    ]

    X = df[feature_cols_extended].values
    y = df[TARGET_COL].values
    return X, y, feature_cols_extended


# ── Model comparison ──────────────────────────────────────────────────────────
def compare_models(X_train, y_train):
    """Train multiple models and return the best one."""
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        "Decision Tree":       DecisionTreeClassifier(max_depth=8, min_samples_split=20, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    print("\n── Model Comparison (5-Fold CV) ──────────────────────────────────")
    best_name, best_score, best_model = None, 0, None
    scores_log = {}

    for name, model in candidates.items():
        scores = cross_val_score(model, X_scaled, y_train, cv=cv, scoring="accuracy")
        mean_acc = scores.mean()
        scores_log[name] = round(float(mean_acc), 4)
        marker = "  ◀ BEST" if mean_acc > best_score else ""
        print(f"  {name:<25}: {mean_acc:.4f} ± {scores.std():.4f}{marker}")
        if mean_acc > best_score:
            best_score, best_name, best_model = mean_acc, name, model

    print(f"\n  🏆 Selected: {best_name} (CV Accuracy: {best_score:.4f})")
    return best_model, scaler, best_name, scores_log


# ── Training pipeline ─────────────────────────────────────────────────────────
def train(data_path: Path = DATA_PATH):
    print("=" * 60)
    print("  BURNOUT DETECTION MODEL TRAINING")
    print("=" * 60)

    # 1. Load & preprocess
    X, y, feature_cols = load_and_preprocess(data_path)

    # 2. Train / test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

    # 3. Compare models, get best
    best_model, scaler, model_name, cv_scores = compare_models(X_train, y_train)

    # 4. Final fit on full training set
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    best_model.fit(X_train_scaled, y_train)

    # 5. Evaluation
    y_pred = best_model.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, y_pred)

    print("\n── Test Set Evaluation ───────────────────────────────────────────")
    print(f"  Accuracy  : {test_acc:.4f} ({test_acc*100:.2f}%)")
    print("\n  Classification Report:")
    target_names = [LABEL_MAP[i] for i in sorted(LABEL_MAP.keys())]
    print(classification_report(y_test, y_pred, target_names=target_names))

    # 6. Persist artifacts
    MODEL_DIR.mkdir(exist_ok=True)
    with open(MODEL_PATH,  "wb") as f: pickle.dump(best_model, f)
    with open(SCALER_PATH, "wb") as f: pickle.dump(scaler, f)

    meta = {
        "model_name":       model_name,
        "test_accuracy":    round(test_acc, 4),
        "cv_scores":        cv_scores,
        "feature_cols":     feature_cols,
        "label_map":        {str(k): v for k, v in LABEL_MAP.items()},
        "label_colors":     {str(k): v for k, v in LABEL_COLORS.items()},
    }
    with open(META_PATH, "w") as f: json.dump(meta, f, indent=2)

    print(f"\n✅ Artifacts saved:")
    print(f"   Model   → {MODEL_PATH}")
    print(f"   Scaler  → {SCALER_PATH}")
    print(f"   Metadata→ {META_PATH}")
    print("=" * 60)
    return test_acc


# ── Prediction helper (used by Flask app) ─────────────────────────────────────
def load_artifacts():
    """Load model, scaler, and metadata from disk."""
    with open(MODEL_PATH,  "rb") as f: model  = pickle.load(f)
    with open(SCALER_PATH, "rb") as f: scaler = pickle.load(f)
    with open(META_PATH,   "r") as f: meta   = json.load(f)
    return model, scaler, meta


def engineer_features(raw: dict) -> np.ndarray:
    """
    Accept a raw input dict and return the feature vector expected by the model.
    raw keys: screen_time_hours, work_hours, sleep_hours, break_frequency,
              typing_speed_wpm, backspace_rate, pause_frequency, app_switch_rate,
              usage_type, focus_session_count, hydration_level, physical_activity
    """
    s = raw
    row = [
        s["screen_time_hours"], s["work_hours"], s["sleep_hours"],
        s["break_frequency"],   s["typing_speed_wpm"], s["backspace_rate"],
        s["pause_frequency"],   s["app_switch_rate"],  s["usage_type"],
        s["focus_session_count"], s["hydration_level"], s["physical_activity"],
        # derived
        s["work_hours"] / (s["break_frequency"] + 1),
        max(0, 8 - s["sleep_hours"]),
        (s["app_switch_rate"] / 60) * 100,
        s["typing_speed_wpm"] / (s["backspace_rate"] + 1),
    ]
    return np.array(row).reshape(1, -1)


def predict_burnout(raw: dict) -> dict:
    """Full prediction pipeline for a single sample."""
    model, scaler, meta = load_artifacts()
    X = engineer_features(raw)
    X_scaled = scaler.transform(X)

    label_int   = int(model.predict(X_scaled)[0])
    proba       = model.predict_proba(X_scaled)[0].tolist()
    label_name  = meta["label_map"][str(label_int)]
    label_color = meta["label_colors"][str(label_int)]

    # Cognitive load: weighted heuristic (not the dataset column)
    cognitive_load = round(
        (raw["pause_frequency"] / 20) * 40
        + (raw["backspace_rate"] / 30) * 25
        + max(0, (80 - raw["typing_speed_wpm"]) / 80) * 20
        + (raw["app_switch_rate"] / 60) * 15,
        1
    )
    cognitive_load = min(cognitive_load, 100)

    # Productivity score
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
        "burnout_level":    label_name,
        "burnout_int":      label_int,
        "burnout_color":    label_color,
        "probabilities":    {meta["label_map"][str(i)]: round(p * 100, 1) for i, p in enumerate(proba)},
        "cognitive_load":   cognitive_load,
        "productivity":     productivity,
        "distraction_index":distraction,
        "model_accuracy":   meta["test_accuracy"],
    }


if __name__ == "__main__":
    # Auto-generate dataset if missing
    if not DATA_PATH.exists():
        print("Dataset not found — generating …")
        sys.path.insert(0, str(BASE_DIR / "data"))
        from generate_dataset import generate_dataset
        df = generate_dataset()
        df.to_csv(DATA_PATH, index=False)
        print(f"Dataset saved → {DATA_PATH}")

    train()
