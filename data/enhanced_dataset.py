"""
Enhanced Dataset Generator for BurnoutAI
Creates more realistic and diverse burnout patterns with advanced features
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import random

# Reproducibility
np.random.seed(42)
random.seed(42)

def generate_enhanced_dataset(n_samples: int = 5000) -> pd.DataFrame:
    """
    Generate an enhanced, more realistic dataset for burnout detection.
    
    New features:
    - Stress indicators (heart rate variability proxy)
    - Social interaction patterns
    - Work-life balance metrics
    - Digital detox behaviors
    - Time-of-day patterns
    - Weekly patterns
    - Sleep quality metrics
    - Mental health indicators
    """
    
    records = []
    
    # Balanced class distribution targets
    target_distribution = {
        0: n_samples // 3,  # Low burnout
        1: n_samples // 3,  # Medium burnout
        2: n_samples // 3   # High burnout
    }
    
    # Define realistic user personas for each burnout level
    personas_by_level = {
        0: [  # Low burnout personas
            {
                'name': 'Healthy Professional',
                'weight': 0.4,
                'base_screen_time': (5, 8),
                'base_work_hours': (6, 8),
                'base_sleep': (7, 9),
                'base_breaks': (4, 8),
                'stress_base': (2, 4)
            },
            {
                'name': 'Balanced Student',
                'weight': 0.35,
                'base_screen_time': (6, 9),
                'base_work_hours': (4, 7),
                'base_sleep': (7, 9),
                'base_breaks': (3, 6),
                'stress_base': (2, 5)
            },
            {
                'name': 'Wellness-Conscious',
                'weight': 0.25,
                'base_screen_time': (4, 7),
                'base_work_hours': (5, 7),
                'base_sleep': (8, 10),
                'base_breaks': (5, 9),
                'stress_base': (1, 3)
            }
        ],
        1: [  # Medium burnout personas
            {
                'name': 'Overworked Student',
                'weight': 0.35,
                'base_screen_time': (9, 13),
                'base_work_hours': (7, 10),
                'base_sleep': (6, 8),
                'base_breaks': (2, 5),
                'stress_base': (5, 7)
            },
            {
                'name': 'Tech Worker',
                'weight': 0.35,
                'base_screen_time': (8, 12),
                'base_work_hours': (8, 10),
                'base_sleep': (6, 8),
                'base_breaks': (2, 4),
                'stress_base': (4, 6)
            },
            {
                'name': 'Busy Parent',
                'weight': 0.30,
                'base_screen_time': (7, 11),
                'base_work_hours': (8, 11),
                'base_sleep': (5, 7),
                'base_breaks': (1, 4),
                'stress_base': (6, 8)
            }
        ],
        2: [  # High burnout personas
            {
                'name': 'Burned Out Professional',
                'weight': 0.4,
                'base_screen_time': (12, 16),
                'base_work_hours': (10, 14),
                'base_sleep': (4, 6),
                'base_breaks': (0, 2),
                'stress_base': (7, 9)
            },
            {
                'name': 'Crisis Student',
                'weight': 0.35,
                'base_screen_time': (13, 16),
                'base_work_hours': (10, 14),
                'base_sleep': (3, 5),
                'base_breaks': (0, 2),
                'stress_base': (8, 10)
            },
            {
                'name': 'Overwhelmed Worker',
                'weight': 0.25,
                'base_screen_time': (11, 15),
                'base_work_hours': (11, 15),
                'base_sleep': (4, 6),
                'base_breaks': (0, 1),
                'stress_base': (7, 9)
            }
        ]
    }
    
    # Generate samples for each burnout level to ensure balance
    for target_level, target_count in target_distribution.items():
        personas = personas_by_level[target_level]
        
        for i in range(target_count):
            # Select persona for this burnout level
            persona = np.random.choice(personas, p=[p['weight'] for p in personas])
            
            # Generate base metrics with persona-specific ranges
            screen_time = np.random.uniform(*persona['base_screen_time'])
            work_hours = np.random.uniform(*persona['base_work_hours'])
            sleep_hours = np.random.uniform(*persona['base_sleep'])
            break_freq = np.random.randint(*persona['base_breaks'])
            stress_level = np.random.uniform(*persona['stress_base'])
            
            # Advanced behavioral patterns
            # Time-of-day patterns (0-24, peak hours)
            peak_activity_hour = np.random.choice([9, 10, 14, 15, 20, 21], p=[0.2, 0.15, 0.2, 0.15, 0.15, 0.15])
            late_night_activity = np.random.uniform(0, 4) if target_level == 2 else np.random.uniform(0, 2)
            
            # Sleep quality (1-10 scale) - varies by burnout level
            if target_level == 0:
                sleep_quality = np.clip(np.random.normal(8, 1), 5, 10)
            elif target_level == 1:
                sleep_quality = np.clip(np.random.normal(6, 1.5), 3, 9)
            else:  # High burnout
                sleep_quality = np.clip(np.random.normal(4, 1.5), 1, 7)
            
            # Social interaction metrics - vary by burnout level
            if target_level == 0:
                social_ratio = np.random.uniform(0.1, 0.4)
                detox_days = np.random.randint(1, 4)
            elif target_level == 1:
                social_ratio = np.random.uniform(0.2, 0.5)
                detox_days = np.random.randint(0, 2)
            else:  # High burnout
                social_ratio = np.random.uniform(0.3, 0.6)
                detox_days = np.random.randint(0, 1)
                
            social_screen_time = screen_time * social_ratio
            work_screen_time = screen_time - social_screen_time
            detox_days_per_week = detox_days
            screen_free_hours = np.random.uniform(2, 6) if detox_days > 0 else np.random.uniform(0, 2)
            
            # Weekly patterns (weekend vs weekday)
            weekend_screen_increase = np.random.uniform(-2, 2)  # Some people reduce, some increase
            weekend_work_decrease = max(0, work_hours - np.random.uniform(2, 6))
            
            # Cognitive patterns - vary by burnout level
            base_wpm = np.random.uniform(50, 80)
            cognitive_fatigue = (screen_time - 6) * 2 + (work_hours - 8) * 1.5 + (8 - sleep_hours) * 3
            
            if target_level == 0:
                typing_speed = base_wpm + np.random.normal(5, 10)
                backspace_rate = np.clip(np.random.normal(5, 3), 0, 20)
                pause_frequency = np.clip(np.random.normal(3, 2), 0, 15)
                focus_sessions = np.random.randint(4, 8)
                app_switch_rate = np.clip(np.random.normal(10, 5), 0, 30)
            elif target_level == 1:
                typing_speed = base_wpm - cognitive_fatigue * 0.5 + np.random.normal(0, 8)
                backspace_rate = np.clip(np.random.normal(10, 4), 0, 30)
                pause_frequency = np.clip(np.random.normal(6, 3), 0, 20)
                focus_sessions = np.random.randint(2, 6)
                app_switch_rate = np.clip(np.random.normal(20, 10), 5, 50)
            else:  # High burnout
                typing_speed = max(20, base_wpm - cognitive_fatigue + np.random.normal(-5, 10))
                backspace_rate = np.clip(np.random.normal(15, 5), 5, 40)
                pause_frequency = np.clip(np.random.normal(10, 4), 5, 25)
                focus_sessions = np.random.randint(0, 3)
                app_switch_rate = np.clip(np.random.normal(35, 15), 15, 80)
            
            # Lifestyle factors - vary by burnout level
            usage_type = np.random.choice([0, 1, 2], p=[0.3, 0.3, 0.4])  # Study, Social, Mixed
            
            if target_level == 0:
                hydration = np.clip(np.random.normal(4, 1), 2, 5)
                physical_activity = np.clip(np.random.normal(60, 20), 20, 120)
                anxiety_level = np.clip(np.random.normal(2, 1.5), 1, 6)
                motivation_level = np.clip(np.random.normal(8, 1.5), 5, 10)
            elif target_level == 1:
                hydration = np.clip(np.random.normal(3, 1), 1, 5)
                physical_activity = np.clip(np.random.normal(40, 20), 5, 90)
                anxiety_level = np.clip(np.random.normal(5, 2), 2, 8)
                motivation_level = np.clip(np.random.normal(5, 2), 2, 8)
            else:  # High burnout
                hydration = np.clip(np.random.normal(2, 1), 1, 4)
                physical_activity = np.clip(np.random.normal(20, 15), 0, 60)
                anxiety_level = np.clip(np.random.normal(7, 2), 4, 10)
                motivation_level = np.clip(np.random.normal(3, 2), 1, 6)
            
            # Work-life balance score
            work_life_balance = np.clip(10 - (work_hours / 16) * 10 - (screen_time / 16) * 5 + (sleep_hours / 8) * 5 + (physical_activity / 120) * 3, 0, 10)
            
            # Enhanced cognitive load calculation
            cognitive_load = np.clip(
                (pause_frequency / 25) * 30 +
                (backspace_rate / 40) * 20 +
                ((80 - typing_speed) / 60) * 20 +
                (app_switch_rate / 80) * 15 +
                (stress_level / 10) * 10 +
                ((10 - sleep_quality) / 10) * 5 +
                np.random.normal(0, 3),
                0, 100
            )
            
            # Use the target level directly for burnout label
            burnout_label = target_level
            
            records.append({
                # Core metrics
                "screen_time_hours": round(screen_time, 2),
                "work_hours": round(work_hours, 2),
                "sleep_hours": round(sleep_hours, 2),
                "break_frequency": break_freq,
                "typing_speed_wpm": round(typing_speed, 1),
                "backspace_rate": round(backspace_rate, 2),
                "pause_frequency": round(pause_frequency, 2),
                "app_switch_rate": round(app_switch_rate, 2),
                "usage_type": usage_type,
                "focus_session_count": focus_sessions,
                "hydration_level": int(hydration),
                "physical_activity": round(physical_activity, 1),
                
                # Enhanced features
                "sleep_quality": round(sleep_quality, 1),
                "stress_level": round(stress_level, 1),
                "work_life_balance": round(work_life_balance, 1),
                "anxiety_level": round(anxiety_level, 1),
                "motivation_level": round(motivation_level, 1),
                "peak_activity_hour": peak_activity_hour,
                "late_night_activity": round(late_night_activity, 1),
                "social_screen_time": round(social_screen_time, 2),
                "work_screen_time": round(work_screen_time, 2),
                "detox_days_per_week": detox_days_per_week,
                "screen_free_hours": round(screen_free_hours, 1),
                "weekend_screen_increase": round(weekend_screen_increase, 1),
                "weekend_work_decrease": round(weekend_work_decrease, 1),
                
                # Targets
                "cognitive_load_score": round(cognitive_load, 1),
                "burnout_level": burnout_label,
            })
    
    df = pd.DataFrame(records)
    return df

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add advanced derived features for better model performance"""
    
    # Work intensity metrics
    df['work_intensity'] = df['work_hours'] / (df['break_frequency'] + 1)
    df['screen_efficiency'] = df['work_screen_time'] / (df['screen_time_hours'] + 0.1)
    
    # Sleep metrics
    df['sleep_deficit'] = np.clip(8 - df['sleep_hours'], 0, 8)
    df['sleep_quality_ratio'] = df['sleep_quality'] / 10
    df['sleep_debt'] = df['sleep_deficit'] * (7 - df['sleep_quality'])
    
    # Cognitive metrics
    df['typing_efficiency'] = df['typing_speed_wpm'] / (df['backspace_rate'] + 1)
    df['cognitive_strain'] = (df['pause_frequency'] / 20) * 40 + (df['backspace_rate'] / 30) * 30
    df['focus_productivity'] = df['focus_session_count'] * df['typing_speed_wpm'] / (df['app_switch_rate'] + 1)
    
    # Lifestyle balance metrics
    df['digital_balance'] = (df['detox_days_per_week'] * 2 + df['screen_free_hours']) / 10
    df['work_life_ratio'] = df['work_hours'] / (24 - df['sleep_hours'] - df['physical_activity'] / 60)
    df['wellness_score'] = (df['sleep_quality'] + df['hydration_level'] + (df['physical_activity'] / 12)) / 3
    
    # Stress and mental health composites
    df['mental_health_index'] = (df['motivation_level'] + (10 - df['anxiety_level']) + df['work_life_balance']) / 3
    df['stress_burnout_risk'] = (df['stress_level'] + df['cognitive_strain']) / 2
    
    # Time-based patterns
    df['night_owl_score'] = df['late_night_activity'] / 4
    df['early_bird_score'] = (df['peak_activity_hour'] < 10).astype(int) * df['sleep_quality']
    
    # Social vs work balance
    df['social_work_ratio'] = df['social_screen_time'] / (df['work_screen_time'] + 0.1)
    
    return df

if __name__ == "__main__":
    print("Generating enhanced dataset...")
    df = generate_enhanced_dataset(5000)
    df = add_derived_features(df)
    
    # Save enhanced dataset
    out_path = Path(__file__).parent / "enhanced_burnout_dataset.csv"
    df.to_csv(out_path, index=False)
    
    print(f"Enhanced dataset saved: {out_path}")
    print(f"Samples: {len(df)}")
    print(f"Features: {df.shape[1] - 2}")
    
    print("\nEnhanced Class Distribution:")
    labels = {0: "Low", 1: "Medium", 2: "High"}
    for k, v in labels.items():
        count = (df["burnout_level"] == k).sum()
        print(f"   {v:8s}: {count:5d} ({count/len(df)*100:.1f}%)")
    
    print("\nNew Features Added:")
    new_features = [col for col in df.columns if col not in [
        'screen_time_hours', 'work_hours', 'sleep_hours', 'break_frequency',
        'typing_speed_wpm', 'backspace_rate', 'pause_frequency', 'app_switch_rate',
        'usage_type', 'focus_session_count', 'hydration_level', 'physical_activity',
        'cognitive_load_score', 'burnout_level'
    ]]
    for feature in new_features[:10]:  # Show first 10 new features
        print(f"   - {feature}")
    if len(new_features) > 10:
        print(f"   ... and {len(new_features) - 10} more features")
