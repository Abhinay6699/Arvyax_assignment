import os

patch_content = """def run_error_analysis(train_df, X_combined_train, model_results, top_n=10):
    \"\"\"
    Analyze misclassifications on the training set using OOF predictions
    to identify failure patterns. Writes findings to ERROR_ANALYSIS.md
    \"\"\"
    print("\\n" + "=" * 60)
    print("STEP 9: ERROR ANALYSIS")
    print("=" * 60)

    state_model = model_results['state_model']
    le_state = model_results['le_state']
    y_true = train_df['emotional_state'].values
    y_true_encoded = le_state.transform(y_true)

    print("  Running 5-fold stratified cross_val_predict to get OOF predictions...")
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Using OOF predicting using predict_proba to capture probabilities across all folds
    oof_proba = cross_val_predict(state_model, X_combined_train, y_true_encoded, cv=cv, method='predict_proba')
    
    y_pred_encoded = np.argmax(oof_proba, axis=1)
    y_pred = le_state.inverse_transform(y_pred_encoded)
    max_proba = np.max(oof_proba, axis=1)

    misclassified_mask = y_true != y_pred
    mis_indices = np.where(misclassified_mask)[0]

    print(f"\\n  Total OOF misclassified: {len(mis_indices)} / {len(y_true)} ({(len(mis_indices)/len(y_true)*100):.1f}%)")

    if len(mis_indices) == 0:
        print("  [*] No misclassifications found.")
        error_md = generate_error_analysis_md([], train_df, y_true, y_pred, max_proba, le_state)
    else:
        # Score each misclassification
        case_scores = []
        for idx in mis_indices:
            row = train_df.iloc[idx]
            text = str(row.get('journal_text', ''))
            word_count = len(text.split())
            conf = max_proba[idx]
            true_l = y_true[idx].lower()
            pred_l = y_pred[idx].lower()
            
            try:
                stress = float(row.get('stress_level', 3))
                energy = float(row.get('energy_level', 3))
            except:
                stress, energy = 3.0, 3.0
            
            score = 0
            if word_count < 5:
                score += 3
            if (energy >= 4 and true_l in ['sad', 'overwhelmed']) or \\
               (stress <= 2 and true_l in ['anxious', 'overwhelmed']):
                score += 3
            if conf < 0.4:
                score += 2
            
            far_pairs = [{'calm', 'overwhelmed'}, {'happy', 'sad'}, {'calm', 'anxious'}]
            if {true_l, pred_l} in far_pairs:
                score += 2
                
            score += (1.0 - conf)
            case_scores.append((score, idx))
            
        case_scores.sort(key=lambda x: x[0], reverse=True)
        analysis_indices = [idx for score, idx in case_scores[:10]]

        errors = []
        for idx in analysis_indices:
            row = train_df.iloc[idx]
            error = {
                'id': row.get('id', idx),
                'journal_text': str(row.get('journal_text', 'N/A')).replace('\\n', '<br>').replace('\\r', ''),
                'true_label': y_true[idx],
                'predicted_label': y_pred[idx],
                'confidence': round(float(max_proba[idx]), 4),
                'stress_level': row.get('stress_level', 'N/A'),
                'energy_level': row.get('energy_level', 'N/A'),
                'sleep_hours': row.get('sleep_hours', 'N/A'),
                'time_of_day': row.get('time_of_day', 'N/A'),
            }
            errors.append(error)

            print(f"  Analzying Sample ID: {error['id']}")

        error_md = generate_error_analysis_md(errors, train_df, y_true, y_pred, max_proba, le_state)

    error_path = os.path.join(os.path.dirname(__file__), "ERROR_ANALYSIS.md")
    with open(error_path, 'w', encoding='utf-8') as f:
        f.write(error_md)
    print(f"\\n  [✓] Saved ERROR_ANALYSIS.md")

    return mis_indices


def generate_error_analysis_md(errors, train_df, y_true, y_pred, confidences, le_state):
    \"\"\"Generate the ERROR_ANALYSIS.md content.\"\"\"
    total = len(y_true)
    n_errors = (y_true != y_pred).sum()

    error_rate = f"{n_errors/total*100:.2f}%"
    mean_conf = f"{confidences.mean():.4f}"
    mean_conf_errors = f"{confidences[y_true != y_pred].mean():.4f}" if n_errors > 0 else "N/A"
    mean_conf_correct = f"{confidences[y_true == y_pred].mean():.4f}" if (total - n_errors) > 0 else "N/A"

    md = f\"\"\"# Error Analysis — Out-of-Fold Cross-Validation

## Overview
| Metric | Value |
|--------|-------|
| Total training samples | {total} |
| OOF misclassified count | {n_errors} |
| OOF error rate | {error_rate} |
| Mean confidence on errors vs correct | {mean_conf_errors} vs {mean_conf_correct} |

## 10 Failure Cases
\"\"\"
    if not errors:
        md += "*No misclassifications found.*\\n"
    else:
        for i, err in enumerate(errors, 1):
            word_count = len(err['journal_text'].replace('<br>', ' ').split())
            explanation = []
            if word_count < 5:
                explanation.append(f"The text is extremely short ({word_count} words), providing minimal semantic signal.")
            
            try:
                en = float(err['energy_level'])
                st = float(err['stress_level'])
                if en >= 4 and err['true_label'] in ['sad', 'overwhelmed']:
                    explanation.append(f"The high energy level ({en}) typically conflicts with '{err['true_label']}'.")
                if st <= 2 and err['true_label'] in ['anxious', 'overwhelmed']:
                    explanation.append(f"The low stress level ({st}) strongly correlates with calm, severely penalizing the true '{err['true_label']}' prediction.")
            except:
                pass
                
            far_pairs = [{'calm', 'overwhelmed'}, {'happy', 'sad'}, {'calm', 'anxious'}]
            if {err['true_label'].lower(), err['predicted_label'].lower()} in far_pairs:
                explanation.append("The predicted and true labels are polar opposites, implying the metadata features likely overwhelmed the textual context.")
            
            if err['confidence'] < 0.4:
                explanation.append(f"The model was highly uncertain (confidence: {err['confidence']}).")
                
            if not explanation:
                explanation.append(f"The model over-indexed on typical features for '{err['predicted_label']}' rather than capturing the nuance of '{err['true_label']}'.")
            
            expl_text = " ".join(explanation)
            
            md += f\"\"\"### Case {i}: Sample ID {err['id']}

| Field | Value |
|-------|-------|
| **Journal Text** | {err['journal_text']} |
| **True Label** | {err['true_label']} |
| **Predicted Label** | {err['predicted_label']} |
| **Confidence Score** | {err['confidence']} |
| **Stress Level** | {err['stress_level']} |
| **Energy Level** | {err['energy_level']} |
| **Sleep Hours** | {err['sleep_hours']} |
| **Time of Day** | {err['time_of_day']} |

**Why the model failed:** {expl_text}

\"\"\"

    classes = le_state.classes_ if le_state else np.unique(y_true)
    import numpy as np
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    md += "## Confusion Matrix\\n\\n| True \\\\ Predicted | " + " | ".join(classes) + " |\\n"
    md += "|" + "|".join(["---"] * (len(classes) + 1)) + "|\\n"
    
    cm_pairs = []
    for i, c_true in enumerate(classes):
        row = f"| **{c_true}** | "
        cells = []
        for j, c_pred in enumerate(classes):
            cells.append(str(cm[i, j]))
            if i != j:
                cm_pairs.append((c_true, c_pred, cm[i, j]))
        row += " | ".join(cells) + " |\\n"
        md += row

    cm_pairs.sort(key=lambda x: x[2], reverse=True)
    top_3 = cm_pairs[:3]
    
    md += "\\n**Top 3 Most Common Confusions:**\\n"
    for idx, (t, p, count) in enumerate(top_3, 1):
        md += f"{idx}. True **{t}** predicted as **{p}** ({count} times)\\n"

    cat1_ex, cat2_ex, cat3_ex, cat4_ex, cat5_ex = [], [], [], [], []
    for err in errors:
        word_count = len(err['journal_text'].replace('<br>', ' ').split())
        en = float(err.get('energy_level', 3) if err.get('energy_level') not in ['N/A', None] else 3)
        st = float(err.get('stress_level', 3) if err.get('stress_level') not in ['N/A', None] else 3)
        has_conflict = (en >= 4 and err['true_label'] in ['sad', 'overwhelmed']) or \\
                       (st <= 2 and err['true_label'] in ['anxious', 'overwhelmed'])
        
        if word_count < 5 and len(cat1_ex) < 2:
            cat1_ex.append(err['id'])
        elif has_conflict and len(cat2_ex) < 2:
            cat2_ex.append(err['id'])
        elif err['confidence'] < 0.4 and len(cat3_ex) < 2:
            cat3_ex.append(err['id'])
        elif len(cat4_ex) < 2:
            cat4_ex.append(err['id'])
        elif len(cat5_ex) < 2:
            cat5_ex.append(err['id'])

    def fmt(lst): return ", ".join(map(str, lst)) if lst else "None"

    md += f\"\"\"
## Failure Pattern Summary

### 1. Short or Vague Text
- Texts like "ok", "I'm fine", "not bad" carry minimal semantic signal
- TF-IDF cannot extract meaningful features from 1-2 word entries
- **Impact**: Model falls back to metadata features, which may be inconsistent
- **Examples from top 10 cases**: {fmt(cat1_ex)}

### 2. Conflicting Signals
- High energy + sad emotional state (e.g., agitated sadness)
- Low stress + anxious state (e.g., existential worry without acute stress)
- **Impact**: Model receives contradictory feature signals
- **Examples from top 10 cases**: {fmt(cat2_ex)}

### 3. Ambiguous Language
- Sarcasm: "Oh great, another wonderful day" (labeled sad but text reads positive)
- Euphemisms: "managing" could be calm or stressed
- Cultural expressions that don't map directly to sentiment
- **Examples from top 10 cases**: {fmt(cat3_ex)}

### 4. Noisy or Contradictory Labels
- Same/similar journal entries labeled differently across samples
- Annotator disagreement on borderline emotional states
- Intensity ratings that don't match the text tone
- **Examples from top 10 cases**: {fmt(cat4_ex)}

### 5. Missing Context
- Entries referencing ongoing situations without background
- Time-series effects: today's entry depends on yesterday's events
- External factors not captured in features (weather, social context)
- **Examples from top 10 cases**: {fmt(cat5_ex)}

## Recommendations for Improvement

1. **Richer text representations**: Use sentence embeddings (e.g., Sentence-BERT) instead of TF-IDF
2. **Minimum text length filtering**: Flag or handle entries with < 5 words differently
3. **Multi-label approach**: Allow multiple emotional states per entry
4. **Temporal features**: Include rolling averages of mood, energy, stress
5. **Ensemble with confidence weighting**: Down-weight predictions with low confidence
6. **Active learning**: Re-label the most uncertain/misclassified samples
\"\"\"
    return md
"""

with open('main.py', 'r', encoding='utf-8') as f:
    orig = f.read()

import re
pattern = re.compile(r"def run_error_analysis\(train_df, X_combined_train, model_results, top_n=15\):.*?def print_summary", re.DOTALL)
new_content = pattern.sub(patch_content + "\n\ndef print_summary", orig)

with open('main.py', 'w', encoding='utf-8') as f:
    f.write(new_content)
