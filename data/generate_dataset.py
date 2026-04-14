"""
================================================================================
 AI-Based Digital Burnout & Cognitive Load Detection System
 File: data/generate_dataset.py
 Purpose: Simulate a realistic dataset for training the burnout detection model
================================================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Reproducibility
np.random.seed(42)

N = 2000  # Total samples

def generate_dataset(n_samples: int = N) -> pd.DataFrame:
    """
    Generate a synthetic but realistic dataset for burnout detection.

    Feature engineering rationale:
    - screen_time_hours   : Daily total screen time (2–16 hrs)
    - work_hours          : Hours actively working/studying (1–12 hrs)
    - break_frequency     : Number of breaks taken per day (0–10)
    - typing_speed_wpm    : Average words per minute (20–120)
    - backspace_rate      : Backspace keypresses per 100 chars (0–30) — high = error-prone/distracted
    - pause_frequency     : Long pauses (>3s) per minute (0–20) — high = cognitive struggle
    - app_switch_rate     : App/tab switches per hour (0–60) — high = distraction
    - usage_type          : 0=Study, 1=Social Media, 2=Mixed
    - sleep_hours         : Previous night's sleep (3–9 hrs)
    - focus_session_count : Number of focused work sessions (0–8)
    - hydration_level     : Self-reported hydration (1–5 scale)
    - physical_activity   : Exercise minutes per day (0–120)
    """

    records = []

    for _ in range(n_samples):
        # ── Core behavioral signals ──────────────────────────────────────────
        screen_time   = np.random.uniform(2, 16)
        work_hours    = np.random.uniform(1, min(screen_time, 12))
        sleep_hours   = np.random.uniform(3, 9)
        break_freq    = np.random.randint(0, 11)
        usage_type    = np.random.choice([0, 1, 2], p=[0.4, 0.3, 0.3])

        # ── Typing & interaction signals ─────────────────────────────────────
        # Tired users type slower and make more errors
        base_wpm = np.random.uniform(40, 100)
        fatigue_penalty = max(0, (screen_time - 6) * 2)
        typing_speed  = max(20, base_wpm - fatigue_penalty + np.random.normal(0, 5))

        backspace_rate   = np.clip(np.random.normal(8, 4) + (screen_time - 6) * 0.5, 0, 30)
        pause_frequency  = np.clip(np.random.normal(5, 3) + (work_hours - 4) * 0.3, 0, 20)
        app_switch_rate  = np.clip(np.random.normal(15, 8) + (usage_type * 5), 0, 60)

        # ── Lifestyle signals ────────────────────────────────────────────────
        focus_sessions   = np.random.randint(0, 9)
        hydration        = np.random.randint(1, 6)
        physical_activity= np.random.uniform(0, 120)

        # ── Burnout Score Calculation (domain-expert heuristic) ───────────────
        # Higher → more burnout
        burnout_score = (
            (screen_time / 16) * 30        # Max 30 pts  — screen overuse
            + (work_hours / 12) * 20       # Max 20 pts  — overwork
            + ((9 - sleep_hours) / 6) * 25 # Max 25 pts  — sleep deprivation
            + ((10 - break_freq) / 10) * 10# Max 10 pts  — no breaks
            + (backspace_rate / 30) * 5    # Max 5 pts   — typing errors
            + (pause_frequency / 20) * 5   # Max 5 pts   — cognitive pauses
            + (app_switch_rate / 60) * 5   # Max 5 pts   — distraction
        )
        burnout_score = np.clip(burnout_score + np.random.normal(0, 3), 0, 100)

        # ── Burnout Label ────────────────────────────────────────────────────
        if burnout_score < 35:
            burnout_label = 0   # Low
        elif burnout_score < 65:
            burnout_label = 1   # Medium
        else:
            burnout_label = 2   # High

        # ── Cognitive Load Score (0–100) ─────────────────────────────────────
        cognitive_load = np.clip(
            (pause_frequency / 20) * 40
            + (backspace_rate / 30) * 25
            + ((16 - typing_speed) / 80) * 20
            + (app_switch_rate / 60) * 15
            + np.random.normal(0, 4),
            0, 100
        )

        records.append({
            "screen_time_hours":  round(screen_time, 2),
            "work_hours":         round(work_hours, 2),
            "sleep_hours":        round(sleep_hours, 2),
            "break_frequency":    break_freq,
            "typing_speed_wpm":   round(typing_speed, 1),
            "backspace_rate":     round(backspace_rate, 2),
            "pause_frequency":    round(pause_frequency, 2),
            "app_switch_rate":    round(app_switch_rate, 2),
            "usage_type":         usage_type,
            "focus_session_count":focus_sessions,
            "hydration_level":    hydration,
            "physical_activity":  round(physical_activity, 1),
            "cognitive_load_score": round(cognitive_load, 1),
            "burnout_level":      burnout_label,
        })

    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    df = generate_dataset()
    out_path = Path(__file__).parent / "burnout_dataset.csv"
    df.to_csv(out_path, index=False)

    print(f"✅ Dataset saved → {out_path}")
    print(f"   Samples : {len(df)}")
    print(f"   Features: {df.shape[1] - 2}")
    print("\nClass distribution:")
    labels = {0: "Low", 1: "Medium", 2: "High"}
    for k, v in labels.items():
        count = (df["burnout_level"] == k).sum()
        print(f"   {v:8s}: {count:5d} ({count/len(df)*100:.1f}%)")
    print("\nSample rows:")
    print(df.head(3).to_string())
