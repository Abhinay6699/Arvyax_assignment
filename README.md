![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Primary_Model-FF6600?style=for-the-badge)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Feature_Engineering-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge)

> Predicts emotional state, intensity level, and personalized well-being recommendations from journal text + physiological metadata. Includes ablation study, uncertainty estimation, and ONNX edge deployment plan.
>
> **Key result:** Text-only TF-IDF achieves F1=0.621. Adding metadata: ΔF1 = -0.0007 (negligible). Text dominates with 98.84% feature importance.

# ArvyaX — Emotional Intelligence ML Pipeline

> End-to-end ML system for predicting emotional states, intensity levels, and generating personalized well-being recommendations.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Setup Instructions](#setup-instructions)
- [How to Run](#how-to-run)
- [Project Structure](#project-structure)
- [Approach Overview](#approach-overview)
- [Feature Engineering](#feature-engineering)
- [Model Architecture](#model-architecture)
- [Decision Engine](#decision-engine)
- [Uncertainty Estimation](#uncertainty-estimation)
- [Ablation Study](#ablation-study)
- [Edge Deployment](#edge-deployment)

---

## Overview

ArvyaX is a mental well-being ML pipeline that:

1. **Predicts emotional state** from journal text + contextual metadata
2. **Estimates intensity** of the emotional state (1-5 scale)
3. **Recommends actions** via a rule-based decision engine (what to do + when)
4. **Quantifies uncertainty** using probability-based and ensemble-based confidence scores
5. **Generates supportive messages** personalized to the user's state

---

## Setup Instructions

### Prerequisites
- Python 3.8+
- pip

### Installation

```bash
cd arvyax_project
pip install -r requirements.txt
```

### Dependencies
| Package | Purpose |
|---------|---------|
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | Feature engineering, baselines, metrics |
| `xgboost` | Primary ML models |
| `scipy` | Sparse matrix operations |
| `joblib` | Model serialization |
| `requests` | Data download from Google Sheets |

---

## How to Run

### Full Pipeline
```bash
python main.py
```

This runs the complete pipeline:
1. Downloads training & test data from Google Sheets
2. Preprocesses and handles missing values
3. Engineers TF-IDF + metadata features
4. Trains emotional state classifier (XGBoost vs RandomForest)
5. Trains intensity predictor (regression vs classification)
6. Runs ablation study (text-only vs text+metadata)
7. Generates predictions on test set
8. Applies decision engine for action recommendations
9. Computes confidence/uncertainty scores
10. Generates supportive messages
11. Saves `predictions.csv`
12. Performs error analysis → `ERROR_ANALYSIS.md`

### Individual Modules
```bash
python data_loader.py           # Test data loading
python feature_engineering.py   # Test feature pipeline
python models.py                # Train models only
python ablation.py              # Run ablation study only
python predict.py               # Generate predictions only
python decision_engine.py       # Test decision engine rules
```

---

## Project Structure

```
arvyax_project/
├── main.py                  # Full pipeline orchestrator
├── data_loader.py           # Data download & preprocessing
├── feature_engineering.py   # TF-IDF + metadata features
├── models.py                # Emotional state + intensity models
├── decision_engine.py       # What-to-do + when-to-do rules
├── uncertainty.py           # Confidence & uncertain_flag
├── ablation.py              # Text-only vs combined ablation
├── predict.py               # Generate predictions.csv
├── README.md                # This file
├── ERROR_ANALYSIS.md        # Misclassification analysis
├── EDGE_PLAN.md             # Edge deployment plan
├── requirements.txt         # Python dependencies
├── predictions.csv          # Output predictions (generated)
└── saved_artifacts/         # Saved models & encoders (generated)
    ├── emotional_state_model.joblib
    ├── intensity_model.joblib
    ├── rf_intensity_regressor.joblib
    ├── tfidf_vectorizer.joblib
    ├── label_encoders.joblib
    ├── le_emotional_state.joblib
    ├── le_intensity.joblib
    └── ablation_results.csv
```

---

## Approach Overview

### Problem Decomposition

The assignment requires predicting **two targets** plus generating **action recommendations**:

| Task | Type | Approach |
|------|------|----------|
| Emotional State | Multi-class classification | XGBoost Classifier |
| Intensity | Ordinal (1-5) | Evaluated as both regression and classification |
| What to Do | Rule-based | Decision engine |
| When to Do | Rule-based | Decision engine |
| Confidence | Model-based | predict_proba + RF tree variance |
| Message | Template-based | State-aware message generation |

---

## Feature Engineering

### Why TF-IDF?

- **Lightweight**: No GPU required, fast to train and infer
- **Interpretable**: Feature importance maps directly to words/phrases
- **Effective**: Captures key emotional keywords and bigrams
- **Edge-friendly**: Small model size (~300 features)

Configuration:
```python
TfidfVectorizer(
    max_features=300,        # Keep dimensionality manageable
    ngram_range=(1, 2),      # Capture phrases like "really anxious"
    stop_words='english',    # Remove noise
    min_df=2,                # Ignore super-rare terms
    max_df=0.95,             # Ignore ubiquitous terms
    sublinear_tf=True        # Log-normalize term frequencies
)
```

### Why Metadata Matters

Metadata features provide crucial context that text alone misses:

| Feature | Why it matters |
|---------|---------------|
| `stress_level` | Direct signal for anxious/stressed states |
| `energy_level` | Distinguishes agitated vs. lethargic states |
| `sleep_hours` | Sleep deprivation affects emotional state |
| `duration_min` | Longer journals → more reflective states |
| `time_of_day` | Mood varies by time (morning optimism, evening fatigue) |
| `ambience_type` | Environmental context affects mood |
| `face_emotion_hint` | Multi-modal signal complementing text |
| `previous_day_mood` | Temporal continuity of emotions |
| `reflection_quality` | Meta-signal about engagement level |

---

## Model Architecture

### Why XGBoost?

1. **Performance**: Consistently top-performing on tabular data
2. **Handles sparse data**: Native support for sparse TF-IDF matrices
3. **Feature importance**: Built-in interpretability
4. **Speed**: Fast training and inference
5. **Robustness**: Handles missing values and mixed feature types
6. **Edge-deployable**: Small model size, ONNX export available

### Emotional State Model
- XGBoost Classifier with 200 estimators, max_depth=6
- RandomForest as baseline comparison
- Best model selected via 5-fold cross-validation (macro F1)

### Intensity Model
- **Regression approach**: XGBoost Regressor → round to nearest int → clip [1,5]
- **Classification approach**: XGBoost Classifier with 5 classes
- Best approach selected by comparing accuracy and RMSE

## Feature Understanding (Part 5)

### What Features Mattered Most

| Feature Name | Importance Score | Type |
|--------------|-------------------|------|
| split | 0.0226 | text |
| helped slightly | 0.0219 | text |
| didn notice | 0.0208 | text |
| felt able | 0.0205 | text |
| unable come | 0.0202 | text |
| felt little | 0.0194 | text |
| finally | 0.0187 | text |
| emotions | 0.0180 | text |
| mentally drained | 0.0179 | text |
| felt like | 0.0171 | text |
| couldn sit | 0.0162 | text |
| tasks | 0.0145 | text |
| mind quiet | 0.0139 | text |
| calmer | 0.0124 | text |
| planning day | 0.0122 | text |
| really changed | 0.0119 | text |
| tension eased | 0.0114 | text |
| planning | 0.0105 | text |
| felt mentally | 0.0104 | text |
| quiet | 0.0103 | text |

### Text vs Metadata Importance

- **Text Sum**: 0.9884 (98.84%)
- **Metadata Sum**: 0.0116 (1.16%)

**Top 5 Text Features**:
1. split (0.0226)
2. helped slightly (0.0219)
3. didn notice (0.0208)
4. felt able (0.0205)
5. unable come (0.0202)

**Top 5 Metadata Features**:
1. ambience_type (0.001368)
2. energy_level (0.001347)
3. reflection_quality (0.001332)
4. previous_day_mood (0.001323)
5. duration_min (0.001290)

### Key Insights
The model is overwhelmingly dominated by textual signals (98.84%), confirming that natural language is the primary medium project users use to convey emotional state. High-importance text features like "mentally drained", "tension eased", and "mind quiet" show that the model has successfully learned to map specific emotional descriptors to their respective classes. Metadata features like `energy_level` and `ambience_type` provide marginal but present contextual signals, though they are largely redundant when rich textual descriptions are available. This aligns with our ablation study findings where removing metadata had negligible impact on model performance.

## Decision Engine

The decision engine uses **priority-based rules** covering:
- 7+ emotional states × 5 intensity levels × 4 time slots
- Stress and energy level modifiers
- 8 action types: `box_breathing`, `deep_work`, `journaling`, `grounding`, `rest`, `light_planning`, `movement`, `pause`
- 5 timing categories: `now`, `within_15_min`, `later_today`, `tonight`, `tomorrow_morning`

---

## Uncertainty Estimation

### For Classification (Emotional State)
- Uses `predict_proba` to get class probabilities
- Confidence = max probability across classes
- `uncertain_flag = 1` if confidence < 0.6

### For Regression (Intensity)
- Trains a parallel RandomForest Regressor
- Computes std of individual tree predictions per sample
- Normalizes to [0, 1] confidence score
- Higher tree agreement → higher confidence

---

## Ablation Study

| Model | Accuracy | F1 (Macro) |
|-------|----------|------------|
| Text-Only (TF-IDF) | 0.6200 ± 0.0176 | 0.6214 ± 0.0175 |
| Text + Metadata | 0.6192 ± 0.0090 | 0.6206 ± 0.0091 |

**Key Finding**: Adding metadata features produced negligible change (ΔF1 = -0.0007). The text signal alone captures most of the emotional content in journal entries. This is a meaningful result — it suggests:

1. **Journal text is the dominant signal**: How users describe their experience matters more than raw numerical signals like stress_level or sleep_hours.
2. **Metadata redundancy**: Features like stress_level and energy_level may be correlated with the emotional language in the text — users who write anxiously also report high stress.
3. **Implication for edge deployment**: Since metadata adds no value, the edge model can drop metadata features entirely, reducing feature dimensionality from 309 → 300 and slightly improving latency.
4. **Label noise hypothesis**: The marginal metadata signal may be drowned out by label noise in the dataset, where the same metadata values appear across different emotional states.

This finding does NOT mean metadata is useless in general — it means TF-IDF already captures the variance that metadata would explain. A future model using sentence embeddings (e.g., SBERT) might show different results.

---

## Edge Deployment

See [EDGE_PLAN.md](EDGE_PLAN.md) for the full edge/mobile deployment strategy, including:
- ONNX model conversion
- Latency optimization (<100ms)
- Offline capability
- Battery/memory tradeoffs

---

## Robustness (Part 9)

### Handling Very Short Text ("ok", "fine", "not bad")

Short inputs are a real challenge for TF-IDF models. Our system handles them as follows:

**Detection**: Any journal_text with fewer than 3 tokens after stop word removal is flagged as a short input.

**Behavior**:
- TF-IDF produces a near-zero sparse vector — no meaningful features fire
- The model falls back heavily on metadata features (stress_level, energy_level, face_emotion_hint)
- The confidence score will be low (<0.5 in most cases) → uncertain_flag = 1
- The decision engine still produces a recommendation based on metadata alone

**Example**:
Input: "ok" → TF-IDF vector ≈ all zeros → model uses metadata → if stress=4, energy=2 → predicted: overwhelmed, confidence: 0.38, uncertain_flag: 1

**Improvement**: For production, a minimum text length check should redirect users: "Could you tell us a bit more about how you're feeling?"

---

### Handling Missing Values

All missing values are handled in data_loader.py before any model sees the data:

| Column Type | Strategy | Reason |
|-------------|----------|--------|
| Numerical (sleep_hours, energy_level, stress_level, duration_min) | Fill with column median | Robust to outliers |
| Categorical (ambience_type, time_of_day, face_emotion_hint) | Fill with "unknown" | Treated as a valid category |
| Text (journal_text) | Fill with "no reflection" | Prevents empty TF-IDF vector |
| Target (emotional_state, intensity) | Drop row | Cannot train on missing labels |

After preprocessing, missing values remaining = 0 (verified in pipeline output).

---

### Handling Contradictory Inputs

Contradictory inputs occur when signals conflict — for example:
- face_emotion_hint = "happy" but journal_text = "I feel terrible"
- stress_level = 1 but text contains "panic", "anxious", "overwhelmed"
- energy_level = 5 but time_of_day = "night" and sleep_hours = 3

**How the model handles it**:
- XGBoost learns feature interactions during training — it has seen conflicting signals
- The model weighs text features more heavily (ablation confirmed text dominates)
- Low agreement between features → lower max probability → higher uncertain_flag rate

**Decision engine behavior for contradictions**:
- If stress_level >= 4 but predicted_state = "calm": decision engine overrides with box_breathing (stress signal takes priority)
- If energy_level <= 2 but predicted_state = "focused": when = "tonight" regardless of other signals
- Uncertain predictions (uncertain_flag=1) are routed to lower-risk actions (pause, grounding) as a safety default

