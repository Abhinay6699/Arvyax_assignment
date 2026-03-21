"""
ablation.py — Ablation study comparing text-only vs text+metadata models.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
import os
import warnings
warnings.filterwarnings("ignore")

SAVE_DIR = os.path.join(os.path.dirname(__file__), "saved_artifacts")


def run_ablation(X_text_only, X_combined, y_raw):
    """
    Compare text-only vs text+metadata models for emotional state prediction.

    Args:
        X_text_only: TF-IDF-only feature matrix
        X_combined: TF-IDF + metadata feature matrix
        y_raw: Raw emotional state labels (strings)

    Returns:
        DataFrame with comparison results
    """
    print("\n" + "=" * 60)
    print("STEP 6: ABLATION STUDY")
    print("=" * 60)

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = []

    # ---- Model 1: Text-Only ----
    print("\n  [*] Training text-only model (TF-IDF features)...")
    text_model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=None
    )

    text_acc_scores = cross_val_score(text_model, X_text_only, y, cv=cv, scoring='accuracy', n_jobs=None)
    text_f1_scores = cross_val_score(text_model, X_text_only, y, cv=cv, scoring='f1_macro', n_jobs=None)

    text_acc = text_acc_scores.mean()
    text_f1 = text_f1_scores.mean()
    text_acc_std = text_acc_scores.std()
    text_f1_std = text_f1_scores.std()

    print(f"  [✓] Text-Only — Acc: {text_acc:.4f} ± {text_acc_std:.4f}, F1: {text_f1:.4f} ± {text_f1_std:.4f}")

    results.append({
        'Model': 'Text-Only (TF-IDF)',
        'Accuracy': round(text_acc, 4),
        'Accuracy_Std': round(text_acc_std, 4),
        'F1_Macro': round(text_f1, 4),
        'F1_Std': round(text_f1_std, 4),
        'Features': 'TF-IDF (300 features)'
    })

    # ---- Model 2: Text + Metadata ----
    print("\n  [*] Training text+metadata model (TF-IDF + encoded metadata)...")
    combined_model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=None
    )

    comb_acc_scores = cross_val_score(combined_model, X_combined, y, cv=cv, scoring='accuracy', n_jobs=None)
    comb_f1_scores = cross_val_score(combined_model, X_combined, y, cv=cv, scoring='f1_macro', n_jobs=None)

    comb_acc = comb_acc_scores.mean()
    comb_f1 = comb_f1_scores.mean()
    comb_acc_std = comb_acc_scores.std()
    comb_f1_std = comb_f1_scores.std()

    print(f"  [✓] Text+Metadata — Acc: {comb_acc:.4f} ± {comb_acc_std:.4f}, F1: {comb_f1:.4f} ± {comb_f1_std:.4f}")

    results.append({
        'Model': 'Text + Metadata',
        'Accuracy': round(comb_acc, 4),
        'Accuracy_Std': round(comb_acc_std, 4),
        'F1_Macro': round(comb_f1, 4),
        'F1_Std': round(comb_f1_std, 4),
        'Features': 'TF-IDF + Metadata (300 + 9 features)'
    })

    # ---- Summary ----
    df_results = pd.DataFrame(results)

    acc_improvement = comb_acc - text_acc
    f1_improvement = comb_f1 - text_f1

    print("\n" + "=" * 60)
    print("  ABLATION STUDY RESULTS")
    print("=" * 60)
    print(f"\n  {'Model':<25} {'Accuracy':<15} {'F1 (Macro)':<15}")
    print(f"  {'─' * 55}")
    for _, row in df_results.iterrows():
        print(f"  {row['Model']:<25} {row['Accuracy']:.4f} ± {row['Accuracy_Std']:.4f}   {row['F1_Macro']:.4f} ± {row['F1_Std']:.4f}")

    print(f"\n  Improvement from adding metadata:")
    print(f"    Accuracy: {acc_improvement:+.4f}")
    print(f"    F1 Macro: {f1_improvement:+.4f}")

    if f1_improvement > 0:
        print(f"\n  ★ Metadata features IMPROVE the model by {f1_improvement:.4f} F1 points")
    else:
        print(f"\n  ★ Metadata features do NOT improve the model (F1 diff: {f1_improvement:.4f})")

    # Save results
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, "ablation_results.csv")
    df_results.to_csv(save_path, index=False)
    print(f"\n  [✓] Saved ablation results to {save_path}")

    return df_results


if __name__ == "__main__":
    from data_loader import load_data
    from feature_engineering import build_features

    train_df, test_df = load_data()
    features = build_features(train_df, test_df)
    run_ablation(features['X_text_only_train'], features['X_combined_train'], train_df['emotional_state'])
