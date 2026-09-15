# Deep Learning Models for Acute Kidney Injury Prediction

**Deep Learning Models for Acute Kidney Injury Prediction: Multi-Center External Validation and Evaluation Under Simulated Continuous Monitoring Conditions**

This repository contains the code for training, evaluating, and analyzing deep learning and machine learning models for Acute Kidney Injury (AKI) prediction using electronic health record (EHR) time-series data. The models are developed on a single-center dataset and externally validated across three hospitals. A key contribution is the **simulated continuous monitoring (online simulation)** framework that evaluates model performance under realistic deployment conditions, where predictions are generated continuously as new data arrives.

## Models

| Model | Type | Description |
|-------|------|-------------|
| **Masked CNN** | Deep Learning | 1D CNN with RMSNorm and masked operations for variable-length sequences |
| **LSTM-Attention** | Deep Learning | LSTM encoder with attention mechanism, operating on Initial Triplet Embedding (ITE) inputs |
| **ITE Transformer** | Deep Learning | Initial Triplet Embedding (ITE) Transformer for irregular time-series |
| **XGBoost** | ML Baseline | Gradient boosting on flattened time-series features |
| **Logistic Regression** | ML Baseline | L2-regularized logistic regression on flattened features |

## Datasets

- **NHIS Ilsan Hospital** (Internal development & test)
- **Chuncheon Sacred Heart Hospital (CSHH)** (External validation)
- **MIMIC-IV** (External validation)

## Repository Structure

```
.
├── shared/                     # Shared infrastructure (callbacks, losses, optimizers, utils)
├── train_cnn/                  # CNN model training & inference
│   ├── models/                 # Model architectures (vanilla, RMS, masked RMS CNN)
│   ├── data/                   # DataModule, collate functions, normalization
│   ├── lightning_modules/      # PyTorch Lightning training module
│   └── scripts/                # Training, inference, online inference scripts
├── train_lstm_attention/       # LSTM-Attention model training & inference
│   ├── src/                    # Model, data, lightning modules, scripts
│   └── experiment_*.py         # Entry-point experiment scripts
├── train_transformer/          # ITE Transformer model training & inference
│   ├── src/                    # Model, data, lightning modules, scripts
│   └── experiment_*.py         # Entry-point experiment scripts
├── train_ml/                   # XGBoost & Logistic Regression baselines
├── experiments/                # 8-step analysis pipeline
│   └── clinical_faithfulness/  # Clinical trend analysis
├── timeseries/                 # Performance evaluation & statistical analysis
├── cohort_selection/           # Patient cohort selection
└── utils/                      # Central configuration, data loaders, metrics
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
# CNN (Masked RMS)
python train_cnn/scripts/experiment_holdout_setting.py \
    --develop_set ilsan --prediction_window_size 0 \
    --input_seq_len 256 --apply_prob 0.0

# LSTM-Attention
python train_lstm_attention/experiment_train.py \
    --develop_set ilsan --prediction_window_size 0

# ITE Transformer
python train_transformer/experiment_train.py \
    --develop_set ilsan --prediction_window_size 0

# ML Baselines (XGBoost, Logistic Regression)
python train_ml/scripts/experiment_holdout.py \
    --develop_set ilsan --prediction_window_size 0
```

### Inference (Holdout Evaluation)

```bash
python train_cnn/scripts/experiment_holdout_inference.py \
    --develop_set ilsan --prediction_window_size 0 \
    --input_seq_len 256 --apply_prob 0.0

python train_lstm_attention/experiment_inference.py \
    --develop_set ilsan --prediction_window_size 0

python train_transformer/experiment_inference.py \
    --develop_set ilsan --prediction_window_size 0
```

### Online Simulation (Continuous Monitoring)

```bash
python train_cnn/scripts/experiment_online_inference.py \
    --develop_set ilsan --prediction_window_size 0

python train_lstm_attention/experiment_online_inference.py \
    --develop_set ilsan --prediction_window_size 0

python train_transformer/experiment_online_inference.py \
    --develop_set ilsan --prediction_window_size 0
```

### Experiment Pipeline

Run the full 8-step analysis pipeline:

```bash
python experiments/run_all.py
```

| Step | Description |
|------|-------------|
| 1 | Reference time matching for non-AKI patients |
| 2 | Baseline model evaluation (discrimination, calibration) |
| 3 | Permutation feature importance (single-point) |
| 4 | Online feature importance (temporal) |
| 5 | Feature subset analysis |
| 6 | Sensitivity analysis |
| 7 | Alert burden assessment |
| 8 | Missingness impact analysis |

## Features

51 input features (48 time-series + 3 demographics):

- **Demographics (3)**: Age, Sex, BMI
- **Vital signs (6)**: SBP, DBP, Heart rate, Respiratory rate, SpO2, Temperature
- **Laboratory (42)**: Albumin, ALP, ALT, aPTT, AST, Bilirubin, BNP, BUN, Calcium, Chloride, Cholesterol, CK, Creatinine, CRP, ESR, GGT, Glucose, HbA1c, Bicarbonate, HDL, Haematocrit, Haemoglobin, Lactate, LDH, Magnesium, pCO2, PDW, pH, Phosphate, Platelet, pO2, Potassium, Procalcitonin, Total protein, PT INR, RBC, RDW, Sodium, tCO2, Triglyceride, Uric acid, WBC

## Prediction Horizons

Models are trained and evaluated at three prediction horizons: **0h**, **48h**, and **72h** before AKI onset.

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT). See the [LICENSE](LICENSE) file for details.
