"""
Enhanced Training Model for BurnoutAI
Uses improved dataset, feature engineering, and model optimization for better accuracy
"""

import sys
import os
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

# Enhanced paths
BASE_DIR = Path(__file__).parent.parent
ENHANCED_DATA_PATH = BASE_DIR / "data" / "enhanced_burnout_dataset.csv"
MODEL_DIR = BASE_DIR / "models"
ENHANCED_MODEL_PATH = MODEL_DIR / "enhanced_burnout_model.pkl"
SCALER_PATH = MODEL_DIR / "enhanced_scaler.pkl"
META_PATH = MODEL_DIR / "enhanced_model_meta.json"

# Enhanced feature columns
CORE_FEATURES = [
    "screen_time_hours", "work_hours", "sleep_hours", "break_frequency",
    "typing_speed_wpm", "backspace_rate", "pause_frequency",
    "app_switch_rate", "usage_type", "focus_session_count",
    "hydration_level", "physical_activity"
]

ENHANCED_FEATURES = [
    "sleep_quality", "stress_level", "work_life_balance", "anxiety_level", 
    "motivation_level", "peak_activity_hour", "late_night_activity",
    "social_screen_time", "work_screen_time", "detox_days_per_week",
    "screen_free_hours", "weekend_screen_increase", "weekend_work_decrease"
]

ALL_FEATURES = CORE_FEATURES + ENHANCED_FEATURES

TARGET_COL = "burnout_level"
LABEL_MAP = {0: "Low", 1: "Medium", 2: "High"}
LABEL_COLORS = {0: "#22c55e", 1: "#f59e0b", 2: "#ef4444"}

def advanced_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Advanced feature engineering for better model performance"""
    
    # Work intensity and efficiency metrics
    df['work_intensity'] = df['work_hours'] / (df['break_frequency'] + 1)
    df['screen_efficiency'] = df['work_screen_time'] / (df['screen_time_hours'] + 0.1)
    df['productivity_score'] = (df['focus_session_count'] * df['typing_speed_wpm']) / (df['app_switch_rate'] + 1)
    
    # Sleep and recovery metrics
    df['sleep_deficit'] = np.clip(8 - df['sleep_hours'], 0, 8)
    df['sleep_quality_ratio'] = df['sleep_quality'] / 10
    df['sleep_debt'] = df['sleep_deficit'] * (10 - df['sleep_quality'])
    df['recovery_score'] = (df['sleep_quality'] + df['detox_days_per_week'] * 2) / 14
    
    # Cognitive and mental health metrics
    df['typing_efficiency'] = df['typing_speed_wpm'] / (df['backspace_rate'] + 1)
    df['cognitive_strain'] = (df['pause_frequency'] / 20) * 40 + (df['backspace_rate'] / 30) * 30
    df['mental_health_index'] = (df['motivation_level'] + (10 - df['anxiety_level']) + df['work_life_balance']) / 3
    df['stress_burnout_risk'] = (df['stress_level'] + df['cognitive_strain']) / 2
    
    # Digital wellness metrics
    df['digital_balance'] = (df['detox_days_per_week'] * 2 + df['screen_free_hours']) / 10
    df['work_life_ratio'] = df['work_hours'] / (24 - df['sleep_hours'] - df['physical_activity'] / 60)
    df['wellness_score'] = (df['sleep_quality'] + df['hydration_level'] + (df['physical_activity'] / 12)) / 3
    df['social_work_balance'] = df['social_screen_time'] / (df['work_screen_time'] + 0.1)
    
    # Time-based patterns
    df['night_owl_score'] = df['late_night_activity'] / 4
    df['early_bird_score'] = (df['peak_activity_hour'] < 10).astype(int) * df['sleep_quality']
    df['circadian_risk'] = df['night_owl_score'] * (10 - df['sleep_quality'])
    
    # Composite risk scores
    df['burnout_risk_composite'] = (
        df['stress_level'] / 10 * 0.3 +
        df['cognitive_strain'] / 50 * 0.2 +
        df['sleep_debt'] / 16 * 0.2 +
        (10 - df['work_life_balance']) / 10 * 0.15 +
        (10 - df['mental_health_index']) / 10 * 0.15
    )
    
    # Interaction terms
    df['stress_x_screen'] = df['stress_level'] * df['screen_time_hours']
    df['sleep_x_work'] = df['sleep_hours'] * df['work_hours']
    df['activity_x_focus'] = df['physical_activity'] * df['focus_session_count']
    
    return df

def load_and_preprocess_enhanced(path: Path):
    """Load enhanced dataset and apply advanced preprocessing"""
    print(f"Loading enhanced dataset from {path}")
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns.")
    
    # Apply advanced feature engineering
    df = advanced_feature_engineering(df)
    
    # Get all feature columns (excluding targets)
    feature_cols = [col for col in df.columns if col not in [TARGET_COL, 'cognitive_load_score']]
    
    # Handle any remaining missing values
    df = df.fillna(df.mean())
    
    X = df[feature_cols].values
    y = df[TARGET_COL].values
    
    print(f"Enhanced features created: {len(feature_cols)} total features")
    return X, y, feature_cols

def get_enhanced_models():
    """Get a comprehensive set of models for comparison"""
    
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=2000, C=1.0, random_state=42, 
            class_weight='balanced', solver='liblinear'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_split=10,
            min_samples_leaf=5, random_state=42, class_weight='balanced'
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=8,
            min_samples_split=10, random_state=42
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=8,
            min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
            random_state=42, eval_metric='mlogloss', use_label_encoder=False
        ),
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=8,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1
        ),
        'SVM': SVC(
            C=1.0, kernel='rbf', probability=True, random_state=42,
            class_weight='balanced'
        ),
        'KNN': KNeighborsClassifier(
            n_neighbors=7, weights='distance', algorithm='auto'
        ),
        'AdaBoost': AdaBoostClassifier(
            n_estimators=100, learning_rate=0.1, random_state=42
        )
    }
    
    return models

def comprehensive_model_comparison(X_train, y_train, X_test, y_test, feature_cols):
    """Comprehensive model comparison with detailed metrics"""
    
    print("\n=== Enhanced Model Comparison ===")
    
    models = get_enhanced_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Use RobustScaler for better handling of outliers
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    for name, model in models.items():
        print(f"\nEvaluating: {name}")
        
        try:
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            # Fit on full training data
            model.fit(X_train_scaled, y_train)
            
            # Test set predictions
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled) if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            test_accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Calculate AUC if probabilities available
            auc = 0
            if y_proba is not None:
                try:
                    auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
                except:
                    auc = 0
            
            results[name] = {
                'model': model,
                'cv_accuracy': cv_mean,
                'cv_std': cv_std,
                'test_accuracy': test_accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc
            }
            
            print(f"  CV Accuracy: {cv_mean:.4f} ± {cv_std:.4f}")
            print(f"  Test Accuracy: {test_accuracy:.4f}")
            print(f"  F1 Score: {f1:.4f}")
            print(f"  AUC: {auc:.4f}")
            
        except Exception as e:
            print(f"  Error with {name}: {str(e)}")
            continue
    
    # Find best model by test accuracy
    best_model_name = max(results.keys(), key=lambda k: results[k]['test_accuracy'])
    best_model = results[best_model_name]['model']
    
    print(f"\n=== Best Model: {best_model_name} ===")
    print(f"Test Accuracy: {results[best_model_name]['test_accuracy']:.4f}")
    print(f"F1 Score: {results[best_model_name]['f1']:.4f}")
    print(f"AUC: {results[best_model_name]['auc']:.4f}")
    
    return best_model, scaler, best_model_name, results

def create_ensemble_model(models_dict):
    """Create an ensemble of the best performing models"""
    
    # Select top 3 models by accuracy
    top_models = sorted(models_dict.items(), key=lambda x: x[1]['test_accuracy'], reverse=True)[:3]
    
    ensemble_models = [(name, model['model']) for name, model in top_models]
    
    # Create voting classifier
    ensemble = VotingClassifier(
        estimators=ensemble_models,
        voting='soft',
        weights=[model['test_accuracy'] for _, model in top_models]
    )
    
    return ensemble, [name for name, _ in ensemble_models]

def train_enhanced_model(data_path: Path = ENHANCED_DATA_PATH):
    """Train enhanced model with comprehensive approach"""
    
    print("=" * 80)
    print("  ENHANCED BURNOUT DETECTION MODEL TRAINING")
    print("=" * 80)
    
    # 1. Load and preprocess data
    X, y, feature_cols = load_and_preprocess_enhanced(data_path)
    
    # 2. Train/test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")
    
    # 3. Comprehensive model comparison
    best_model, scaler, model_name, all_results = comprehensive_model_comparison(
        X_train, y_train, X_test, y_test, feature_cols
    )
    
    # 4. Create and evaluate ensemble
    ensemble_model, ensemble_models = create_ensemble_model(all_results)
    
    # Scale data for ensemble
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train ensemble
    ensemble_model.fit(X_train_scaled, y_train)
    ensemble_pred = ensemble_model.predict(X_test_scaled)
    ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
    
    print(f"\n=== Ensemble Model ===")
    print(f"Models: {', '.join(ensemble_models)}")
    print(f"Ensemble Accuracy: {ensemble_accuracy:.4f}")
    
    # 5. Select final model (ensemble if better, otherwise best single)
    if ensemble_accuracy > all_results[model_name]['test_accuracy']:
        final_model = ensemble_model
        final_model_name = f"Ensemble({', '.join(ensemble_models)})"
        final_accuracy = ensemble_accuracy
        print("Using Ensemble model as final model")
    else:
        final_model = best_model
        final_model_name = model_name
        final_accuracy = all_results[model_name]['test_accuracy']
        print(f"Using {model_name} as final model")
    
    # 6. Final evaluation
    X_train_final_scaled = scaler.fit_transform(X_train)
    X_test_final_scaled = scaler.transform(X_test)
    
    final_model.fit(X_train_final_scaled, y_train)
    y_final_pred = final_model.predict(X_test_final_scaled)
    final_accuracy = accuracy_score(y_test, y_final_pred)
    
    print("\n=== Final Model Evaluation ===")
    print(f"Final Model: {final_model_name}")
    print(f"Final Test Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
    
    print("\nClassification Report:")
    target_names = [LABEL_MAP[i] for i in sorted(LABEL_MAP.keys())]
    print(classification_report(y_test, y_final_pred, target_names=target_names))
    
    # 7. Feature importance (if available)
    feature_importance = {}
    if hasattr(final_model, 'feature_importances_'):
        importance_scores = final_model.feature_importances_
        feature_importance = dict(zip(feature_cols, importance_scores))
        
        print("\nTop 10 Most Important Features:")
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        for feature, importance in sorted_features[:10]:
            print(f"  {feature:30s}: {importance:.4f}")
    
    # 8. Save artifacts
    MODEL_DIR.mkdir(exist_ok=True)
    with open(ENHANCED_MODEL_PATH, "wb") as f:
        pickle.dump(final_model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    
    # Enhanced metadata
    meta = {
        "model_name": final_model_name,
        "test_accuracy": round(final_accuracy, 4),
        "feature_cols": feature_cols,
        "feature_importance": feature_importance,
        "label_map": {str(k): v for k, v in LABEL_MAP.items()},
        "label_colors": {str(k): v for k, v in LABEL_COLORS.items()},
        "all_model_results": {
            name: {
                "test_accuracy": round(result["test_accuracy"], 4),
                "f1": round(result["f1"], 4),
                "auc": round(result["auc"], 4)
            } for name, result in all_results.items()
        },
        "dataset_info": {
            "samples": len(X),
            "features": len(feature_cols),
            "classes": len(np.unique(y))
        }
    }
    
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"\nEnhanced artifacts saved:")
    print(f"   Model   -> {ENHANCED_MODEL_PATH}")
    print(f"   Scaler  -> {SCALER_PATH}")
    print(f"   Metadata-> {META_PATH}")
    print("=" * 80)
    
    return final_accuracy

if __name__ == "__main__":
    # Install required packages if not available
    try:
        import xgboost as xgb
        import lightgbm as lgb
    except ImportError:
        print("Installing required packages...")
        os.system("pip install xgboost lightgbm")
        import xgboost as xgb
        import lightgbm as lgb
    
    # Train enhanced model
    accuracy = train_enhanced_model()
    print(f"\nEnhanced model training completed with accuracy: {accuracy:.4f}")
