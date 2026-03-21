"""
data_loader.py — Download Google Sheets as CSV and preprocess the data.
Handles missing values for numerical, categorical, and text columns.
"""

import pandas as pd
import numpy as np
import requests
import io
import os


# Google Sheets export URLs
TRAIN_SHEET_ID = "1yLDum7yWr3IH0KivluCBEvqHGlfvFW_S"
TEST_SHEET_ID = "1lCvTufEhGgtDJp6b9oYyFXpCZqWPirSX"

TRAIN_URL = f"https://docs.google.com/spreadsheets/d/{TRAIN_SHEET_ID}/export?format=csv"
TEST_URL = f"https://docs.google.com/spreadsheets/d/{TEST_SHEET_ID}/export?format=csv"

# Column definitions
NUMERICAL_COLS = ["sleep_hours", "energy_level", "stress_level", "duration_min"]
CATEGORICAL_COLS = ["ambience_type", "time_of_day", "face_emotion_hint", "previous_day_mood"]
TEXT_COL = "journal_text"


def download_sheet(url, name="data"):
    """Download a Google Sheet as CSV and return a DataFrame."""
    print(f"  [*] Downloading {name}...")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))
        print(f"  [✓] Downloaded {name}: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"  [!] Failed to download {name}: {e}")
        # Fallback: try to load from local cache
        cache_path = os.path.join(os.path.dirname(__file__), f"{name}.csv")
        if os.path.exists(cache_path):
            print(f"  [*] Loading from local cache: {cache_path}")
            return pd.read_csv(cache_path)
        raise


def preprocess(df, is_train=True):
    """
    Handle missing values:
      - Numerical cols: fill with median
      - Categorical cols: fill with 'unknown'
      - Text col: fill with 'no reflection'
    """
    df = df.copy()

    # --- Numerical columns ---
    for col in NUMERICAL_COLS:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # --- Categorical columns ---
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("unknown").astype(str)

    # --- Text column ---
    if TEXT_COL in df.columns:
        df[TEXT_COL] = df[TEXT_COL].fillna("no reflection").astype(str)
        # Handle empty strings
        df[TEXT_COL] = df[TEXT_COL].replace("", "no reflection")

    # --- Target columns (train only) ---
    if is_train:
        if "emotional_state" in df.columns:
            df["emotional_state"] = df["emotional_state"].fillna("unknown").astype(str)
        if "intensity" in df.columns:
            df["intensity"] = df["intensity"].fillna(df["intensity"].median())

    print(f"  [✓] Preprocessing complete. Missing values remaining: {df.isnull().sum().sum()}")
    return df


def load_data():
    """Main entry point: download and preprocess train + test data."""
    print("\n" + "=" * 60)
    print("STEP 1: LOADING & PREPROCESSING DATA")
    print("=" * 60)

    train_df = download_sheet(TRAIN_URL, "training_data")
    test_df = download_sheet(TEST_URL, "test_data")

    # Cache locally for future runs
    cache_dir = os.path.dirname(__file__)
    train_df.to_csv(os.path.join(cache_dir, "training_data.csv"), index=False)
    test_df.to_csv(os.path.join(cache_dir, "test_data.csv"), index=False)

    train_df = preprocess(train_df, is_train=True)
    test_df = preprocess(test_df, is_train=False)

    print(f"\n  Train columns: {list(train_df.columns)}")
    print(f"  Train shape: {train_df.shape}")
    print(f"  Test shape:  {test_df.shape}")

    return train_df, test_df


if __name__ == "__main__":
    train_df, test_df = load_data()
    print("\nSample train data:")
    print(train_df.head())
