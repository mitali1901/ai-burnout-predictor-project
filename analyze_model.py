"""
Model Analysis Script for BurnoutAI
Analyze current model performance and identify improvement areas
"""

import pandas as pd
import numpy as np
from models.train_model import load_artifacts, DATA_PATH
from sklearn.metrics import classification_report, confusion_matrix
import json

def analyze_current_model():
    """Analyze the current model performance"""
    print("=== CURRENT MODEL ANALYSIS ===")
    
    # Load current model and data
    model, scaler, meta = load_artifacts()
    df = pd.read_csv(DATA_PATH)
    
    print(f"Dataset size: {len(df)} samples")
    print(f"Features: {df.shape[1]-2} (excluding targets)")
    print(f"Current accuracy: {meta['test_accuracy']:.4f} ({meta['test_accuracy']*100:.2f}%)")
    
    # Check class distribution
    print("\n=== CLASS DISTRIBUTION ===")
    class_counts = df['burnout_level'].value_counts().sort_index()
    labels = {0: 'Low', 1: 'Medium', 2: 'High'}
    for cls, count in class_counts.items():
        print(f"{labels[cls]:8s}: {count:5d} ({count/len(df)*100:.1f}%)")
    
    # Check feature correlations with target
    print("\n=== FEATURE CORRELATIONS ===")
    feature_cols = [col for col in df.columns if col not in ['burnout_level', 'cognitive_load_score']]
    correlations = df[feature_cols + ['burnout_level']].corr()['burnout_level'].sort_values(key=abs, ascending=False)
    print("Top 10 most correlated features:")
    for feature, corr in correlations.head(11).items():
        if feature != 'burnout_level':
            print(f"{feature:25s}: {corr:7.4f}")
    
    # Check data quality
    print("\n=== DATA QUALITY ===")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    
    # Feature statistics
    print("\n=== FEATURE STATISTICS ===")
    numeric_features = df.select_dtypes(include=[np.number]).columns
    for col in numeric_features:
        if col not in ['burnout_level', 'cognitive_load_score']:
            print(f"{col:25s}: mean={df[col].mean():6.2f}, std={df[col].std():6.2f}, min={df[col].min():6.2f}, max={df[col].max():6.2f}")

if __name__ == "__main__":
    analyze_current_model()
