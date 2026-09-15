#!/bin/bash
# =============================================================================
# Run full ML baseline pipeline: training + holdout inference + online inference
#
# Usage:
#   cd code/train_ml/scripts
#   bash experiment.sh [device]
#
# Example:
#   bash experiment.sh 0        # Use GPU 0
#   bash experiment.sh cpu      # Use CPU
# =============================================================================

DEVICE=${1:-0}

echo "=============================================="
echo "ML Baselines (XGBoost + Logistic Regression)"
echo "Device: $DEVICE"
echo "=============================================="

# --- Holdout Training + Inference ---
for PW in 0 48 72; do
    echo ""
    echo ">>> Holdout: develop_set=ilsan, prediction_window=${PW}"
    python experiment_holdout.py \
        --develop_set ilsan \
        --prediction_window_size $PW \
        --device $DEVICE
done

# --- Online Inference ---
for PW in 0 48 72; do
    echo ""
    echo ">>> Online Inference: develop_set=ilsan, prediction_window=${PW}"
    python experiment_online_inference.py \
        --develop_set ilsan \
        --prediction_window_size $PW
done

echo ""
echo "=============================================="
echo "All ML baseline experiments completed!"
echo "=============================================="
