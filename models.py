"""
models.py — Train emotional state (classification) and intensity (regression + classification) models.
Uses XGBoost as primary and RandomForest as baseline.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, mean_squared_error
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

SAVE_DIR = os.path.join(os.path.dirname(__file__), "saved_artifacts")


def train_emotional_state_model(X_train, y_train_raw, tfidf=None):
    """
    Train emotional state classifier using XGBoost and RandomForest.
    Returns: best_model, label_encoder, report_dict
    """
    print("\n" + "-" * 50)
    print("  PART 1: Emotional State Prediction (Classification)")
    print("-" * 50)

    os.makedirs(SAVE_DIR, exist_ok=True)

    # Encode target
    le_state = LabelEncoder()
    y_train = le_state.fit_transform(y_train_raw)
    n_classes = len(le_state.classes_)
    print(f"  Classes ({n_classes}): {list(le_state.classes_)}")

    # --- XGBoost Classifier ---
    print("\n  [*] Training XGBoost Classifier...")
    xgb_clf = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=None
    )
    xgb_clf.fit(X_train, y_train)
    xgb_preds = xgb_clf.predict(X_train)
    xgb_acc = accuracy_score(y_train, xgb_preds)
    xgb_f1 = f1_score(y_train, xgb_preds, average='macro')
    print(f"  [✓] XGBoost — Train Acc: {xgb_acc:.4f}, Macro F1: {xgb_f1:.4f}")

    # --- RandomForest Baseline ---
    print("\n  [*] Training RandomForest Baseline...")
    rf_clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=None
    )
    rf_clf.fit(X_train, y_train)
    rf_preds = rf_clf.predict(X_train)
    rf_acc = accuracy_score(y_train, rf_preds)
    rf_f1 = f1_score(y_train, rf_preds, average='macro')
    print(f"  [✓] RandomForest — Train Acc: {rf_acc:.4f}, Macro F1: {rf_f1:.4f}")

    # --- Cross-validation comparison ---
    print("\n  [*] Cross-validation (5-fold)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    xgb_cv_scores = cross_val_score(xgb_clf, X_train, y_train, cv=cv, scoring='f1_macro', n_jobs=None)
    rf_cv_scores = cross_val_score(rf_clf, X_train, y_train, cv=cv, scoring='f1_macro', n_jobs=None)

    print(f"  [✓] XGBoost CV F1:      {xgb_cv_scores.mean():.4f} ± {xgb_cv_scores.std():.4f}")
    print(f"  [✓] RandomForest CV F1:  {rf_cv_scores.mean():.4f} ± {rf_cv_scores.std():.4f}")

    # Select best
    if xgb_cv_scores.mean() >= rf_cv_scores.mean():
        best_model = xgb_clf
        best_name = "XGBoost"
    else:
        best_model = rf_clf
        best_name = "RandomForest"

    print(f"\n  ★ Best model: {best_name}")

    # Full classification report
    best_preds = best_model.predict(X_train)
    report = classification_report(y_train, best_preds, target_names=le_state.classes_, output_dict=True)
    print("\n  Classification Report (on training data):")
    print(classification_report(y_train, best_preds, target_names=le_state.classes_))

    # --- Feature Importance ---
    if best_name == "XGBoost":
        print("\n  [*] Extracting feature importances...")
        importances = best_model.feature_importances_
        
        # Build feature names
        text_features = tfidf.get_feature_names_out().tolist() if tfidf else [f"text_{i}" for i in range(300)]
        meta_features = ["ambience_type", "time_of_day", "face_emotion_hint", "previous_day_mood", "reflection_quality",
                         "sleep_hours", "energy_level", "stress_level", "duration_min"]
        
        all_features = text_features + meta_features
        
        # Create DataFrame
        fi_df = pd.DataFrame({
            'feature_name': all_features,
            'importance_score': importances,
            'feature_type': ['text'] * len(text_features) + ['metadata'] * len(meta_features)
        })
        
        fi_df = fi_df.sort_values(by='importance_score', ascending=False)
        fi_df.to_csv(os.path.join(SAVE_DIR, "feature_importances.csv"), index=False)
        
        print(f"\n  Top 10 features:")
        print(fi_df.head(10).to_string(index=False))
        
    joblib.dump(best_model, os.path.join(SAVE_DIR, "emotional_state_model.joblib"))
    joblib.dump(le_state, os.path.join(SAVE_DIR, "le_emotional_state.joblib"))
    print(f"  [✓] Saved best model & encoder")

    return best_model, le_state, report, {
        'xgb_cv_f1': xgb_cv_scores.mean(),
        'rf_cv_f1': rf_cv_scores.mean(),
        'best_model_name': best_name
    }


def train_intensity_model(X_train, y_train_raw):
    """
    Train intensity predictor using both regression and classification approaches.
    Returns: best_model, approach_name, metrics_dict
    """
    print("\n" + "-" * 50)
    print("  PART 2: Intensity Prediction (Regression vs Classification)")
    print("-" * 50)

    os.makedirs(SAVE_DIR, exist_ok=True)
    y_train = y_train_raw.values.astype(int)

    # ---- Approach 1: Regression (XGBRegressor → round to int) ----
    print("\n  [*] Approach 1: XGBoost Regressor...")
    xgb_reg = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=None
    )
    xgb_reg.fit(X_train, y_train)
    reg_preds_raw = xgb_reg.predict(X_train)
    reg_preds = np.clip(np.round(reg_preds_raw), 1, 5).astype(int)

    reg_rmse = np.sqrt(mean_squared_error(y_train, reg_preds_raw))
    reg_acc = accuracy_score(y_train, reg_preds)
    print(f"  [✓] Regressor — RMSE: {reg_rmse:.4f}, Rounded Acc: {reg_acc:.4f}")

    # ---- Approach 2: Classification (XGBClassifier with 5 classes) ----
    print("\n  [*] Approach 2: XGBoost Classifier (5 classes)...")
    le_intensity = LabelEncoder()
    y_train_cls = le_intensity.fit_transform(y_train)

    xgb_cls = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=None
    )
    xgb_cls.fit(X_train, y_train_cls)
    cls_preds = le_intensity.inverse_transform(xgb_cls.predict(X_train))
    cls_acc = accuracy_score(y_train, cls_preds)
    cls_rmse = np.sqrt(mean_squared_error(y_train, cls_preds))
    print(f"  [✓] Classifier — Acc: {cls_acc:.4f}, RMSE: {cls_rmse:.4f}")

    # ---- Compare and choose ----
    print("\n  Comparison:")
    print(f"  {'Approach':<25} {'RMSE':<10} {'Accuracy':<10}")
    print(f"  {'─' * 45}")
    print(f"  {'Regression→Round':<25} {reg_rmse:<10.4f} {reg_acc:<10.4f}")
    print(f"  {'Classification':<25} {cls_rmse:<10.4f} {cls_acc:<10.4f}")

    # Choose based on accuracy (primary) and RMSE (secondary)
    if cls_acc >= reg_acc:
        best_model = xgb_cls
        best_approach = "classification"
        best_metrics = {'rmse': cls_rmse, 'accuracy': cls_acc}
        joblib.dump(le_intensity, os.path.join(SAVE_DIR, "le_intensity.joblib"))
        print(f"\n  ★ Best approach: Classification (higher accuracy)")
    else:
        best_model = xgb_reg
        best_approach = "regression"
        best_metrics = {'rmse': reg_rmse, 'accuracy': reg_acc}
        le_intensity = None
        print(f"\n  ★ Best approach: Regression (higher accuracy after rounding)")

    # Also train a RandomForest for uncertainty estimation
    print("\n  [*] Training RandomForest Regressor for uncertainty estimation...")
    rf_reg = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=None
    )
    rf_reg.fit(X_train, y_train)
    joblib.dump(rf_reg, os.path.join(SAVE_DIR, "rf_intensity_regressor.joblib"))

    # Save best model
    joblib.dump(best_model, os.path.join(SAVE_DIR, "intensity_model.joblib"))
    print(f"  [✓] Saved intensity model ({best_approach})")

    return best_model, best_approach, le_intensity, best_metrics, rf_reg


def train_all_models(X_combined, train_df, tfidf=None):
    """
    Orchestrate training of all models.
    Returns dict with all models, encoders, and metrics.
    """
    print("\n" + "=" * 60)
    print("STEP 3: TRAINING MODELS")
    print("=" * 60)

    results = {}

    # Emotional state
    y_state = train_df['emotional_state']
    state_model, le_state, state_report, state_metrics = train_emotional_state_model(X_combined, y_state, tfidf=tfidf)
    results['state_model'] = state_model
    results['le_state'] = le_state
    results['state_report'] = state_report
    results['state_metrics'] = state_metrics

    # Intensity
    y_intensity = train_df['intensity']
    intensity_model, approach, le_intensity, int_metrics, rf_reg = train_intensity_model(X_combined, y_intensity)
    results['intensity_model'] = intensity_model
    results['intensity_approach'] = approach
    results['le_intensity'] = le_intensity
    results['intensity_metrics'] = int_metrics
    results['rf_intensity'] = rf_reg

    return results


if __name__ == "__main__":
    from data_loader import load_data
    from feature_engineering import build_features

    train_df, test_df = load_data()
    features = build_features(train_df, test_df)
    results = train_all_models(features['X_combined_train'], train_df)
