"""
predict.py — Generate final predictions.csv with all required columns:
id, predicted_state, predicted_intensity, confidence, uncertain_flag,
what_to_do, when_to_do, supportive_message
"""

import numpy as np
import pandas as pd
import joblib
import os

from decision_engine import decide
from uncertainty import get_combined_confidence

SAVE_DIR = os.path.join(os.path.dirname(__file__), "saved_artifacts")


def generate_supportive_message(predicted_state, what_to_do, when_to_do, intensity, confidence):
    """
    Generate a short 1-2 sentence human-like supportive message.
    """
    templates = {
        "calm": "You're in a good headspace right now. {action} could help you channel this energy well.",
        "focused": "You seem focused and clear-headed. This is a great time for {action}.",
        "neutral": "You seem balanced right now. Consider {action} to make the most of your day.",
        "restless": "You seem a bit unsettled. A short session of {action} can help bring some calm.",
        "overwhelmed": "It sounds like things feel heavy right now. Let's take it one step at a time — {action} might help you reset.",
        "mixed": "Your feelings seem a bit tangled right now. That's okay. Try {action} to help sort things out."
    }

    when_phrases = {
        "now": "Try to do this right now.",
        "within_15_min": "Try to find 15 minutes for this soon.",
        "later_today": "Set aside some time for this later today.",
        "tonight": "Wind down tonight with this.",
        "tomorrow_morning": "Start tomorrow morning with this."
    }

    state = str(predicted_state).lower().strip()
    template = templates.get(state, "Take things one step at a time. {action} can help you navigate your current state.")
    
    action_str = what_to_do.replace('_', ' ')
    message = template.format(action=action_str) + " " + when_phrases.get(when_to_do, "")

    if confidence < 0.5:
        message = "We're not entirely sure how you're feeling, but — " + message

    return message


def predict_test(test_df, X_combined_test, model_results):
    """
    Generate predictions for the test set.

    Args:
        test_df: Test DataFrame with original columns
        X_combined_test: Combined feature matrix for test
        model_results: Dict from train_all_models

    Returns:
        predictions DataFrame
    """
    print("\n" + "=" * 60)
    print("STEP 7: GENERATING PREDICTIONS")
    print("=" * 60)

    state_model = model_results['state_model']
    le_state = model_results['le_state']
    intensity_model = model_results['intensity_model']
    intensity_approach = model_results['intensity_approach']
    le_intensity = model_results.get('le_intensity')
    rf_intensity = model_results['rf_intensity']

    n_samples = X_combined_test.shape[0]
    print(f"\n  [*] Predicting for {n_samples} test samples...")

    # ---- Predict Emotional State ----
    state_preds_encoded = state_model.predict(X_combined_test)
    predicted_states = le_state.inverse_transform(state_preds_encoded)
    print(f"  [✓] Emotional states predicted")

    # ---- Predict Intensity ----
    if intensity_approach == "classification" and le_intensity is not None:
        int_preds_encoded = intensity_model.predict(X_combined_test)
        predicted_intensities = le_intensity.inverse_transform(int_preds_encoded)
    else:
        raw_preds = intensity_model.predict(X_combined_test)
        predicted_intensities = np.clip(np.round(raw_preds), 1, 5).astype(int)
    print(f"  [✓] Intensities predicted")

    # ---- Confidence & Uncertainty ----
    conf_results = get_combined_confidence(state_model, rf_intensity, X_combined_test)
    confidence = conf_results['confidence']
    uncertain_flag = conf_results['uncertain_flag']
    print(f"  [✓] Confidence scores computed")

    # ---- Decision Engine ----
    what_list = []
    when_list = []
    for i in range(n_samples):
        stress = test_df.iloc[i].get('stress_level', 2)
        energy = test_df.iloc[i].get('energy_level', 2)
        time = test_df.iloc[i].get('time_of_day', 'morning')

        what, when = decide(predicted_states[i], predicted_intensities[i], stress, energy, time)
        what_list.append(what)
        when_list.append(when)
    print(f"  [✓] Decision engine applied")

    # ---- Supportive Messages ----
    messages = []
    for i in range(n_samples):
        msg = generate_supportive_message(predicted_states[i], what_list[i], when_list[i], predicted_intensities[i], confidence[i])
        messages.append(msg)
    print(f"  [✓] Supportive messages generated")

    # ---- Build output DataFrame ----
    predictions = pd.DataFrame({
        'id': test_df['id'].values if 'id' in test_df.columns else range(n_samples),
        'predicted_state': predicted_states,
        'predicted_intensity': predicted_intensities,
        'confidence': np.round(confidence, 4),
        'uncertain_flag': uncertain_flag,
        'what_to_do': what_list,
        'when_to_do': when_list,
        'supportive_message': messages
    })

    # Save
    output_path = os.path.join(os.path.dirname(__file__), "predictions.csv")
    predictions.to_csv(output_path, index=False)
    print(f"\n  [✓] Saved predictions to {output_path}")

    # Summary
    print(f"\n  Prediction Summary:")
    print(f"    Total samples:         {n_samples}")
    print(f"    Unique states:         {predictions['predicted_state'].nunique()}")
    print(f"    Mean confidence:       {confidence.mean():.4f}")
    print(f"    Uncertain samples:     {uncertain_flag.sum()} ({uncertain_flag.mean()*100:.1f}%)")
    print(f"    State distribution:")
    for state, count in predictions['predicted_state'].value_counts().items():
        print(f"      {state:<20} {count:>4} ({count/n_samples*100:.1f}%)")

    return predictions


if __name__ == "__main__":
    from data_loader import load_data
    from feature_engineering import build_features
    from models import train_all_models

    train_df, test_df = load_data()
    features = build_features(train_df, test_df)
    model_results = train_all_models(features['X_combined_train'], train_df)
    predictions = predict_test(test_df, features['X_combined_test'], model_results)
    print("\nSample predictions:")
    print(predictions.head())
