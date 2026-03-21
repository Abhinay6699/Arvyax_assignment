# Edge Deployment Plan — ArvyaX Emotional Intelligence Pipeline

## Overview

This document outlines the strategy for deploying the ArvyaX emotional state prediction models on edge devices (mobile phones, tablets) for real-time, offline inference.

---

## 1. Model Compression & Conversion

### XGBoost → ONNX Conversion

```python
# Conversion example
import onnxmltools
from onnxmltools.convert.xgboost import convert as convert_xgboost
from onnxconverter_common.data_types import FloatTensorType

# Define input shape
initial_type = [('features', FloatTensorType([None, n_features]))]

# Convert
onnx_model = convert_xgboost(xgb_model, initial_types=initial_type)
onnxmltools.utils.save_model(onnx_model, 'emotional_state_model.onnx')
```

### Model Size Comparison

| Format | Emotional State Model | Intensity Model | Total |
|--------|----------------------|-----------------|-------|
| XGBoost (joblib) | ~3-5 MB | ~2-4 MB | ~5-9 MB |
| ONNX | ~1-2 MB | ~1-2 MB | ~2-4 MB |
| Quantized ONNX (INT8) | ~0.5-1 MB | ~0.5-1 MB | ~1-2 MB |

### Additional Compression Techniques
- **Tree pruning**: Reduce max_depth from 6 → 4, reduce n_estimators from 200 → 100
- **Feature selection**: Use top-50 TF-IDF features instead of 300
- **Quantization**: INT8 quantization via ONNX Runtime
- **Dead tree removal**: Remove trees that contribute < 0.1% to predictions

---

## 2. Latency Targets

### Target: < 100ms total inference on mobile CPU

| Stage | Target Latency | Notes |
|-------|---------------|-------|
| Text preprocessing | < 5 ms | String operations |
| TF-IDF vectorization | < 10 ms | Pre-fitted vocabulary lookup |
| Metadata encoding | < 2 ms | Label encoding lookup |
| Feature concatenation | < 3 ms | Sparse matrix creation |
| Emotional state inference | < 30 ms | ONNX Runtime |
| Intensity inference | < 20 ms | ONNX Runtime |
| Decision engine | < 5 ms | Rule-based (pure logic) |
| Uncertainty estimation | < 15 ms | RF tree prediction std |
| Message generation | < 5 ms | Template substitution |
| **Total** | **< 95 ms** | **Within target** |

### Optimizations
- **Batch processing**: Process daily journal entries in batch during idle CPU time
- **Warm-up**: Pre-load models into memory on app launch
- **Lazy loading**: Load intensity model only when needed
- **Caching**: Cache TF-IDF vocabulary as a flat array for O(1) lookup

---

## 3. Offline Capability

### Architecture

```
┌─────────────────────────────────────────┐
│              Mobile App                  │
│                                          │
│  ┌──────────┐  ┌──────────┐             │
│  │  Journal  │──│ TF-IDF   │             │
│  │  Input    │  │ Encoder  │             │
│  └──────────┘  └────┬─────┘             │
│                     │                    │
│  ┌──────────┐  ┌────▼─────┐             │
│  │ Metadata │──│ Feature  │             │
│  │ Input    │  │ Matrix   │             │
│  └──────────┘  └────┬─────┘             │
│                     │                    │
│         ┌───────────┴───────────┐       │
│    ┌────▼────┐          ┌───────▼──┐    │
│    │ State   │          │ Intensity│    │
│    │ Model   │          │ Model    │    │
│    │ (ONNX)  │          │ (ONNX)   │    │
│    └────┬────┘          └────┬─────┘    │
│         │                    │           │
│    ┌────▼────────────────────▼─────┐    │
│    │      Decision Engine          │    │
│    │   (what_to_do + when_to_do)   │    │
│    └──────────────┬────────────────┘    │
│                   │                      │
│    ┌──────────────▼────────────────┐    │
│    │    Supportive Message Gen     │    │
│    └──────────────┬────────────────┘    │
│                   │                      │
│    ┌──────────────▼────────────────┐    │
│    │         Local Storage         │    │
│    │  (SQLite / SharedPreferences) │    │
│    └───────────────────────────────┘    │
│                                          │
│  ★ NO INTERNET REQUIRED                 │
└─────────────────────────────────────────┘
```

### Files Required on Device

| File | Size | Purpose |
|------|------|---------|
| `emotional_state_model.onnx` | ~1-2 MB | State prediction |
| `intensity_model.onnx` | ~1-2 MB | Intensity prediction |
| `tfidf_vocab.json` | ~50 KB | TF-IDF vocabulary |
| `tfidf_idf_weights.npy` | ~2 KB | IDF weights |
| `label_mappings.json` | ~1 KB | Categorical encoders |
| `decision_rules.json` | ~2 KB | Decision engine config |
| **Total** | **~2-4 MB** | **Fully offline** |

---

## 4. TF-IDF Vectorizer Serialization

### For Python (Server/Development)
```python
import joblib
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.joblib')
```

### For Mobile (Lightweight)
```python
import json
import numpy as np

# Export vocabulary
vocab = tfidf_vectorizer.vocabulary_
with open('tfidf_vocab.json', 'w') as f:
    json.dump(vocab, f)

# Export IDF weights
np.save('tfidf_idf_weights.npy', tfidf_vectorizer.idf_)

# Export parameters
params = {
    'max_features': 300,
    'ngram_range': [1, 2],
    'sublinear_tf': True,
    'norm': 'l2'
}
with open('tfidf_params.json', 'w') as f:
    json.dump(params, f)
```

### Mobile Implementation (Pseudocode)
```kotlin
// Android (Kotlin)
class TfidfEncoder(context: Context) {
    private val vocab: Map<String, Int> = loadVocab()
    private val idfWeights: FloatArray = loadIdfWeights()

    fun encode(text: String): FloatArray {
        val tokens = tokenize(text)           // Split + lowercase + ngrams
        val tfVector = computeTF(tokens)       // Term frequency
        val tfidfVector = applyIdf(tfVector)   // TF * IDF
        return normalize(tfidfVector)          // L2 normalize
    }
}
```

---

## 5. Mobile Framework Options

### Option A: ONNX Runtime Mobile (Recommended)

| Aspect | Details |
|--------|---------|
| Android | `onnxruntime-mobile` (~5 MB AAR) |
| iOS | `onnxruntime-mobile` (~3 MB framework) |
| Pros | Native XGBoost support, optimized for mobile CPUs |
| Cons | Larger binary than TFLite |

```gradle
// Android build.gradle
dependencies {
    implementation 'com.microsoft.onnxruntime:onnxruntime-mobile:1.16.0'
}
```

### Option B: TensorFlow Lite

| Aspect | Details |
|--------|---------|
| Convert | XGBoost → sklearn wrapper → tf.lite |
| Android | `tensorflow-lite` (~3 MB AAR) |
| iOS | `TensorFlowLiteSwift` (~2 MB) |
| Pros | Smaller binary, better ecosystem |
| Cons | XGBoost conversion is indirect, potential accuracy loss |

### Recommendation: **ONNX Runtime Mobile**
- Direct XGBoost → ONNX conversion without intermediate steps
- Better numerical fidelity
- Active development and community support

---

## 6. Fallback Strategy

```
┌─────────────────┐
│  Input Features  │
└────────┬────────┘
         │
    ┌────▼────┐     Success
    │  ONNX   │─────────────► Prediction
    │  Model   │
    └────┬────┘
         │ Failure (crash, OOM, timeout)
    ┌────▼────────────┐
    │  Rule-Based      │
    │  Decision Engine │────► Fallback Prediction
    └─────────────────┘
```

### Fallback Rules
```python
def fallback_predict(stress_level, energy_level, face_emotion_hint, time_of_day):
    """Rule-based prediction when ML model fails."""
    if stress_level >= 4:
        state = "stressed"
        intensity = min(5, stress_level)
    elif energy_level <= 2:
        state = "low"
        intensity = 3
    elif face_emotion_hint == "happy":
        state = "happy"
        intensity = 3
    elif face_emotion_hint == "sad":
        state = "sad"
        intensity = 3
    else:
        state = "neutral"
        intensity = 2
    return state, intensity
```

### When to Trigger Fallback
- Model loading failure (corrupted file, insufficient memory)
- Inference timeout (> 200ms)
- ONNX Runtime crash
- Out-of-memory exception

---

## 7. Battery & Memory Tradeoffs

### Memory Profile

| Component | Memory (RAM) | Notes |
|-----------|-------------|-------|
| ONNX Runtime | ~15 MB | Shared runtime |
| State model | ~2 MB | Loaded into memory |
| Intensity model | ~2 MB | Lazy-loaded |
| TF-IDF vocab | ~0.5 MB | Constant |
| Feature buffers | ~1 MB | Per-inference |
| **Total** | **~20 MB** | **Acceptable for modern phones** |

### Battery Optimization

1. **Batch processing**: Don't run inference on every keystroke. Process when user submits journal entry.
2. **Background scheduling**: Use Android WorkManager / iOS BackgroundTasks for batch predictions.
3. **Model warm-up**: Load model once on app start, keep in memory pool.
4. **CPU affinity**: Pin inference to efficiency cores (ARM big.LITTLE).
5. **Avoid GPU**: For small models like XGBoost, CPU inference is faster due to GPU launch overhead.

### Storage Impact

| Item | Size |
|------|------|
| App binary + ONNX Runtime | ~8 MB |
| Models (ONNX) | ~2-4 MB |
| TF-IDF data | ~0.1 MB |
| Local prediction history | ~1 MB/year |
| **Total** | **~12 MB** |

---

## 8. Update Strategy

### Model Updates
1. **OTA model download**: Periodically check for new model versions
2. **Delta updates**: Only download changed model files
3. **A/B testing**: Roll out new models to 10% of users first
4. **Rollback**: Keep previous model version as backup

### Version Management
```json
{
  "model_version": "1.2.0",
  "min_app_version": "2.0.0",
  "state_model": "emotional_state_v1.2.onnx",
  "intensity_model": "intensity_v1.2.onnx",
  "tfidf_vocab_hash": "sha256:abc123...",
  "release_date": "2026-03-18"
}
```

---

## Summary

| Requirement | Solution | Status |
|-------------|----------|--------|
| < 100ms inference | ONNX Runtime + optimized pipeline | ✅ |
| < 5 MB model size | ONNX + quantization | ✅ |
| Fully offline | All models local, no API calls | ✅ |
| Cross-platform | ONNX Runtime Mobile (Android + iOS) | ✅ |
| Fallback strategy | Rule-based decision engine | ✅ |
| Battery efficient | Batch processing + lazy loading | ✅ |
| Updatable | OTA model downloads | ✅ |
