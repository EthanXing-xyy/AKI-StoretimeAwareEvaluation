#!/bin/bash

# Online Inference Script for 1D CNN Model
# 
# This script runs online inference to simulate real-time predictions
# at each timestamp as new data arrives.
# Runs for all develop sets (ilsan, cchlmc) and prediction horizons (0, 48, 72 hours)
#
# Usage:
#   ./online_inference.sh [device]
#
# Examples:
#   ./online_inference.sh 0
#   ./online_inference.sh cpu

set -e

# Default parameters
DEVICE=${1:-"0"}
TIME_RESOLUTION=${2:-1.0}
SEQ_LEN=${3:-256}

# DataLoader settings (optimized for parallel data loading)
NUM_WORKERS=4
DATALOADER_BATCH_SIZE=128    # Patients per DataLoader batch
GPU_BATCH_SIZE=512           # Max samples per GPU forward pass (controls memory)
PREFETCH_FACTOR=4

# Develop sets and prediction horizons to evaluate
DEVELOP_SETS=("ilsan" "cchlmc")
PREDICTION_WINDOWS=(0 48 72)

echo "=============================================="
echo "Online Inference - 1D CNN Model"
echo "=============================================="
echo "Device: $DEVICE"
echo "Time Resolution: $TIME_RESOLUTION hours"
echo "Sequence Length: $SEQ_LEN"
echo "Develop Sets: ${DEVELOP_SETS[*]}"
echo "Prediction Windows: ${PREDICTION_WINDOWS[*]} hours"
echo ""
echo "DataLoader Settings:"
echo "  num_workers: $NUM_WORKERS"
echo "  dataloader_batch_size: $DATALOADER_BATCH_SIZE"
echo "  gpu_batch_size: $GPU_BATCH_SIZE"
echo "  prefetch_factor: $PREFETCH_FACTOR"
echo "=============================================="

for DEVELOP_SET in "${DEVELOP_SETS[@]}"; do
    for PREDICTION_WINDOW_SIZE in "${PREDICTION_WINDOWS[@]}"; do
        echo ""
        echo "=============================================="
        echo "Running: $DEVELOP_SET / prediction horizon $PREDICTION_WINDOW_SIZE hours"
        echo "=============================================="
        
        python experiment_online_inference.py \
            --develop_set "$DEVELOP_SET" \
            --prediction_window_size "$PREDICTION_WINDOW_SIZE" \
            --device "$DEVICE" \
            --time_resolution "$TIME_RESOLUTION" \
            --min_observations 1 \
            --seq_len "$SEQ_LEN" \
            --num_workers "$NUM_WORKERS" \
            --dataloader_batch_size "$DATALOADER_BATCH_SIZE" \
            --gpu_batch_size "$GPU_BATCH_SIZE" \
            --prefetch_factor "$PREFETCH_FACTOR"
        
        echo "Completed: $DEVELOP_SET / prediction horizon $PREDICTION_WINDOW_SIZE hours"
    done
done

echo ""
echo "=============================================="
echo "All experiments completed!"
echo "  Develop sets: ${DEVELOP_SETS[*]}"
echo "  Prediction windows: ${PREDICTION_WINDOWS[*]} hours"
echo "=============================================="
