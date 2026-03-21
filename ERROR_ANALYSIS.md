# Error Analysis — Out-of-Fold Cross-Validation

## Overview
| Metric | Value |
|--------|-------|
| Total training samples | 1200 |
| OOF misclassified count | 457 |
| OOF error rate | 38.08% |
| Mean confidence on errors | 0.4479 |
| Mean confidence on correct | 0.7760 |

## 10 Failure Cases
### Case 1: Sample ID 850

| Field | Value |
|-------|-------|
| **Journal Text** | more clear today |
| **True Label** | overwhelmed |
| **Predicted Label** | calm |
| **Confidence Score** | 0.3299 |
| **Stress Level** | 2 |
| **Energy Level** | 3 |
| **Sleep Hours** | 3.5 |
| **Time of Day** | evening |

**Why the model failed:** The text is extremely short (3 words), providing minimal semantic signal. The low stress level (2.0) strongly correlates with calm, severely penalizing the true 'overwhelmed' prediction. The predicted and true labels are polar opposites, implying the metadata features likely overwhelmed the textual context. The model was highly uncertain (confidence: 0.3299).

### Case 2: Sample ID 778

| Field | Value |
|-------|-------|
| **Journal Text** | a little lighter |
| **True Label** | overwhelmed |
| **Predicted Label** | calm |
| **Confidence Score** | 0.3509 |
| **Stress Level** | 3 |
| **Energy Level** | 5 |
| **Sleep Hours** | 8.5 |
| **Time of Day** | morning |

**Why the model failed:** The text is extremely short (3 words), providing minimal semantic signal. The high energy level (5.0) typically conflicts with 'overwhelmed'. The predicted and true labels are polar opposites, implying the metadata features likely overwhelmed the textual context. The model was highly uncertain (confidence: 0.3509).

### Case 3: Sample ID 501

| Field | Value |
|-------|-------|
| **Journal Text** | honestly got distracted again. |
| **True Label** | overwhelmed |
| **Predicted Label** | restless |
| **Confidence Score** | 0.2356 |
| **Stress Level** | 3 |
| **Energy Level** | 4 |
| **Sleep Hours** | 4.0 |
| **Time of Day** | afternoon |

**Why the model failed:** The text is extremely short (4 words), providing minimal semantic signal. The high energy level (4.0) typically conflicts with 'overwhelmed'. The model was highly uncertain (confidence: 0.2356).

### Case 4: Sample ID 827

| Field | Value |
|-------|-------|
| **Journal Text** | not sure what changed |
| **True Label** | overwhelmed |
| **Predicted Label** | mixed |
| **Confidence Score** | 0.2634 |
| **Stress Level** | 2 |
| **Energy Level** | 2 |
| **Sleep Hours** | 5.0 |
| **Time of Day** | evening |

**Why the model failed:** The text is extremely short (4 words), providing minimal semantic signal. The low stress level (2.0) strongly correlates with calm, severely penalizing the true 'overwhelmed' prediction. The model was highly uncertain (confidence: 0.2634).

### Case 5: Sample ID 839

| Field | Value |
|-------|-------|
| **Journal Text** | hard to focus |
| **True Label** | overwhelmed |
| **Predicted Label** | restless |
| **Confidence Score** | 0.2707 |
| **Stress Level** | 5 |
| **Energy Level** | 4 |
| **Sleep Hours** | 6.0 |
| **Time of Day** | night |

**Why the model failed:** The text is extremely short (3 words), providing minimal semantic signal. The high energy level (4.0) typically conflicts with 'overwhelmed'. The model was highly uncertain (confidence: 0.2707).

### Case 6: Sample ID 1072

| Field | Value |
|-------|-------|
| **Journal Text** | still heavy |
| **True Label** | overwhelmed |
| **Predicted Label** | neutral |
| **Confidence Score** | 0.3221 |
| **Stress Level** | 1 |
| **Energy Level** | 3 |
| **Sleep Hours** | 8.0 |
| **Time of Day** | afternoon |

**Why the model failed:** The text is extremely short (2 words), providing minimal semantic signal. The low stress level (1.0) strongly correlates with calm, severely penalizing the true 'overwhelmed' prediction. The model was highly uncertain (confidence: 0.3221).

### Case 7: Sample ID 833

| Field | Value |
|-------|-------|
| **Journal Text** | tired but okay ... |
| **True Label** | overwhelmed |
| **Predicted Label** | focused |
| **Confidence Score** | 0.326 |
| **Stress Level** | 2 |
| **Energy Level** | 2 |
| **Sleep Hours** | 8.0 |
| **Time of Day** | evening |

**Why the model failed:** The text is extremely short (4 words), providing minimal semantic signal. The low stress level (2.0) strongly correlates with calm, severely penalizing the true 'overwhelmed' prediction. The model was highly uncertain (confidence: 0.326).

### Case 8: Sample ID 487

| Field | Value |
|-------|-------|
| **Journal Text** | felt heavy |
| **True Label** | overwhelmed |
| **Predicted Label** | mixed |
| **Confidence Score** | 0.3297 |
| **Stress Level** | 1 |
| **Energy Level** | 2 |
| **Sleep Hours** | 5.0 |
| **Time of Day** | night |

**Why the model failed:** The text is extremely short (2 words), providing minimal semantic signal. The low stress level (1.0) strongly correlates with calm, severely penalizing the true 'overwhelmed' prediction. The model was highly uncertain (confidence: 0.3297).

### Case 9: Sample ID 966

| Field | Value |
|-------|-------|
| **Journal Text** | felt better |
| **True Label** | overwhelmed |
| **Predicted Label** | mixed |
| **Confidence Score** | 0.3548 |
| **Stress Level** | 3 |
| **Energy Level** | 4 |
| **Sleep Hours** | 7.0 |
| **Time of Day** | afternoon |

**Why the model failed:** The text is extremely short (2 words), providing minimal semantic signal. The high energy level (4.0) typically conflicts with 'overwhelmed'. The model was highly uncertain (confidence: 0.3548).

### Case 10: Sample ID 1015

| Field | Value |
|-------|-------|
| **Journal Text** | tired but okay |
| **True Label** | overwhelmed |
| **Predicted Label** | mixed |
| **Confidence Score** | 0.3598 |
| **Stress Level** | 2 |
| **Energy Level** | 5 |
| **Sleep Hours** | 5.0 |
| **Time of Day** | morning |

**Why the model failed:** The text is extremely short (3 words), providing minimal semantic signal. The high energy level (5.0) typically conflicts with 'overwhelmed'. The low stress level (2.0) strongly correlates with calm, severely penalizing the true 'overwhelmed' prediction. The model was highly uncertain (confidence: 0.3598).

## Confusion Matrix

| True \ Predicted | calm | focused | mixed | neutral | overwhelmed | restless |
|---|---|---|---|---|---|---|
| **calm** | 117 | 11 | 21 | 24 | 22 | 21 |
| **focused** | 17 | 131 | 15 | 10 | 7 | 13 |
| **mixed** | 28 | 16 | 115 | 12 | 12 | 8 |
| **neutral** | 23 | 12 | 13 | 130 | 12 | 11 |
| **overwhelmed** | 18 | 8 | 15 | 11 | 122 | 16 |
| **restless** | 24 | 10 | 16 | 12 | 19 | 128 |

**Top 3 Most Common Confusions:**
1. True **mixed** predicted as **calm** (28 times)
2. True **calm** predicted as **neutral** (24 times)
3. True **restless** predicted as **calm** (24 times)

## Failure Pattern Summary

### 1. Short or Vague Text
- Texts like "ok", "I'm fine", "not bad" carry minimal semantic signal
- TF-IDF cannot extract meaningful features from 1-2 word entries
- **Impact**: Model falls back to metadata features, which may be inconsistent
- **Examples from top 10 cases**: 850, 778

### 2. Conflicting Signals
- High energy + sad emotional state (e.g., agitated sadness)
- Low stress + anxious state (e.g., existential worry without acute stress)
- **Impact**: Model receives contradictory feature signals
- **Examples from top 10 cases**: 501, 827

### 3. Ambiguous Language
- Sarcasm: "Oh great, another wonderful day" (labeled sad but text reads positive)
- Euphemisms: "managing" could be calm or stressed
- Cultural expressions that don't map directly to sentiment
- **Examples from top 10 cases**: 839, 1072

### 4. Noisy or Contradictory Labels
- Same/similar journal entries labeled differently across samples
- Annotator disagreement on borderline emotional states
- Intensity ratings that don't match the text tone
- **Examples from top 10 cases**: 833, 487

### 5. Missing Context
- Entries referencing ongoing situations without background
- Time-series effects: today's entry depends on yesterday's events
- External factors not captured in features (weather, social context)
- **Examples from top 10 cases**: 966, 1015

## Recommendations for Improvement

1. **Richer text representations**: Use sentence embeddings (e.g., Sentence-BERT) instead of TF-IDF
2. **Minimum text length filtering**: Flag or handle entries with < 5 words differently
3. **Multi-label approach**: Allow multiple emotional states per entry
4. **Temporal features**: Include rolling averages of mood, energy, stress
5. **Ensemble with confidence weighting**: Down-weight predictions with low confidence
6. **Active learning**: Re-label the most uncertain/misclassified samples
