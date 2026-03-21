"""
feature_engineering.py — Build TF-IDF text features and metadata features.
Combines them into a single sparse feature matrix.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack, csr_matrix
import joblib
import os

SAVE_DIR = os.path.join(os.path.dirname(__file__), "saved_artifacts")

NUMERICAL_COLS = ["sleep_hours", "energy_level", "stress_level", "duration_min"]
CATEGORICAL_COLS = ["ambience_type", "time_of_day", "face_emotion_hint", "previous_day_mood", "reflection_quality"]
TEXT_COL = "journal_text"


def build_features(train_df, test_df=None):
    """
    Build feature matrices from train (and optionally test) DataFrames.

    Returns:
        dict with keys:
            'X_text_only_train', 'X_combined_train',
            'X_text_only_test', 'X_combined_test' (if test_df provided),
            'tfidf', 'label_encoders'
    """
    print("\n" + "=" * 60)
    print("STEP 2: FEATURE ENGINEERING")
    print("=" * 60)

    os.makedirs(SAVE_DIR, exist_ok=True)
    result = {}

    # ==================== TF-IDF TEXT FEATURES ====================
    print("\n  [*] Building TF-IDF features...")
    tfidf = TfidfVectorizer(
        max_features=300,
        ngram_range=(1, 2),
        stop_words='english',
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )

    # Ensure text is clean
    train_text = train_df[TEXT_COL].fillna("no reflection").astype(str)
    X_text_train = tfidf.fit_transform(train_text)
    result['X_text_only_train'] = X_text_train
    print(f"  [✓] TF-IDF train shape: {X_text_train.shape}")

    if test_df is not None:
        test_text = test_df[TEXT_COL].fillna("no reflection").astype(str)
        X_text_test = tfidf.transform(test_text)
        result['X_text_only_test'] = X_text_test
        print(f"  [✓] TF-IDF test shape: {X_text_test.shape}")

    # Save TF-IDF vectorizer
    joblib.dump(tfidf, os.path.join(SAVE_DIR, "tfidf_vectorizer.joblib"))
    result['tfidf'] = tfidf

    # ==================== METADATA FEATURES ====================
    print("\n  [*] Building metadata features...")
    label_encoders = {}

    def encode_metadata(df, fit=True):
        """Encode categorical + numerical features into a sparse matrix."""
        parts = []

        # Categorical features via LabelEncoder
        for col in CATEGORICAL_COLS:
            if col in df.columns:
                vals = df[col].fillna("unknown").astype(str)
                if fit:
                    le = LabelEncoder()
                    encoded = le.fit_transform(vals)
                    label_encoders[col] = le
                else:
                    le = label_encoders[col]
                    # Handle unseen labels gracefully
                    encoded = []
                    for v in vals:
                        if v in le.classes_:
                            encoded.append(le.transform([v])[0])
                        else:
                            encoded.append(len(le.classes_))  # Assign unseen label a new index
                    encoded = np.array(encoded)
                parts.append(csr_matrix(encoded.reshape(-1, 1), dtype=np.float64))

        # Numerical features
        for col in NUMERICAL_COLS:
            if col in df.columns:
                vals = df[col].fillna(0).values.astype(np.float64)
                parts.append(csr_matrix(vals.reshape(-1, 1)))

        if parts:
            return hstack(parts)
        return csr_matrix(np.zeros((len(df), 1)))

    X_meta_train = encode_metadata(train_df, fit=True)
    print(f"  [✓] Metadata train shape: {X_meta_train.shape}")

    # Combined features
    X_combined_train = hstack([X_text_train, X_meta_train])
    result['X_combined_train'] = X_combined_train
    print(f"  [✓] Combined train shape: {X_combined_train.shape}")

    if test_df is not None:
        X_meta_test = encode_metadata(test_df, fit=False)
        X_combined_test = hstack([X_text_test, X_meta_test])
        result['X_combined_test'] = X_combined_test
        print(f"  [✓] Combined test shape: {X_combined_test.shape}")

    # Save encoders
    joblib.dump(label_encoders, os.path.join(SAVE_DIR, "label_encoders.joblib"))
    result['label_encoders'] = label_encoders

    print("\n  [✓] Feature engineering complete!")
    return result


if __name__ == "__main__":
    from data_loader import load_data
    train_df, test_df = load_data()
    features = build_features(train_df, test_df)
    print("\nFeature keys:", list(features.keys()))
