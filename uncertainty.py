"""
uncertainty.py — Confidence estimation and uncertain_flag computation.
Uses predict_proba for classification and tree-prediction std for regression.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor


def get_confidence_classification(model, X_sample, threshold=0.6):
    """
    Get confidence scores for classification model predictions.

    Args:
        model: Trained classifier with predict_proba method
        X_sample: Feature matrix
        threshold: Confidence threshold below which uncertain_flag = 1

    Returns:
        confidence (np.array): Max probability for each sample (0-1)
        uncertain_flag (np.array): 1 if uncertain, 0 if confident
    """
    try:
        proba = model.predict_proba(X_sample)
        max_proba = proba.max(axis=1)
        confidence = max_proba
        uncertain_flag = (max_proba < threshold).astype(int)
    except AttributeError:
        # Model doesn't support predict_proba — fallback
        confidence = np.ones(X_sample.shape[0]) * 0.5
        uncertain_flag = np.ones(X_sample.shape[0], dtype=int)

    return confidence, uncertain_flag


def get_confidence_regression(rf_model, X_sample, intensity_min=1, intensity_max=5):
    """
    Get confidence scores for intensity prediction using RandomForest tree variance.

    Uses the standard deviation of individual tree predictions as an uncertainty proxy.
    Normalizes to 0-1 range where 1 = high confidence, 0 = low confidence.

    Args:
        rf_model: Trained RandomForestRegressor
        X_sample: Feature matrix
        intensity_min: Minimum intensity value (for normalization)
        intensity_max: Maximum intensity value (for normalization)

    Returns:
        confidence (np.array): Confidence score (0-1), higher = more confident
        uncertain_flag (np.array): 1 if uncertain, 0 if confident
    """
    # Get individual tree predictions
    tree_preds = np.array([tree.predict(X_sample) for tree in rf_model.estimators_])
    # shape: (n_trees, n_samples)

    # Std across trees for each sample
    pred_std = tree_preds.std(axis=0)

    # Normalize: max possible std is (max - min) / 2
    max_std = (intensity_max - intensity_min) / 2.0
    normalized_std = np.clip(pred_std / max_std, 0, 1)

    # Confidence = 1 - normalized uncertainty
    confidence = 1.0 - normalized_std

    # Uncertain if confidence < 0.6
    uncertain_flag = (confidence < 0.6).astype(int)

    return confidence, uncertain_flag


def get_combined_confidence(state_model, rf_intensity_model, X_sample, threshold=0.6):
    """
    Get combined confidence from both emotional state and intensity models.

    Returns:
        state_confidence, state_uncertain,
        intensity_confidence, intensity_uncertain,
        overall_confidence, overall_uncertain
    """
    state_conf, state_unc = get_confidence_classification(state_model, X_sample, threshold)
    int_conf, int_unc = get_confidence_regression(rf_intensity_model, X_sample)

    # Overall: average of both, flag if either is uncertain
    overall_conf = (state_conf + int_conf) / 2.0
    overall_unc = ((state_unc == 1) | (int_unc == 1)).astype(int)

    return {
        'state_confidence': state_conf,
        'state_uncertain': state_unc,
        'intensity_confidence': int_conf,
        'intensity_uncertain': int_unc,
        'confidence': overall_conf,
        'uncertain_flag': overall_unc
    }


if __name__ == "__main__":
    print("Uncertainty module loaded successfully.")
    print("Use get_confidence_classification() for emotional state model.")
    print("Use get_confidence_regression() for intensity model.")
    print("Use get_combined_confidence() for both.")
