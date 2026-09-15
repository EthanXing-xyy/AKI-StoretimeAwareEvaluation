#!/bin/bash

# Online Inference Script for Transformer Model
# 
# This script runs online inference to simulate real-time predictions
# at each timestamp as new data arrives.
#
# Usage:
#   ./online_inference.sh [develop_set] [device]
#
# Examples:
#   ./online_inference.sh ilsan 0          # Run all horizons on GPU 0
#   ./online_inference.sh cchlmc 1         # Run all horizons on GPU 1
#   ./online_inference.sh ilsan cpu        # Run on CPU

set -e

# Default parameters
DEVELOP_SET=${1:-"ilsan"}
DEVICE=${2:-"1"}
TIME_RESOLUTION=${3:-1.0}

# DataLoader settings (optimized for parallel data loading)
NUM_WORKERS=4
DATALOADER_BATCH_SIZE=64    # Patients per DataLoader batch
GPU_BATCH_SIZE=256           # Max samples per GPU forward pass (controls memory)
PREFETCH_FACTOR=2

# Prediction horizons to process
PREDICTION_WINDOWS=(0 48 72)

echo "=============================================="
echo "Online Inference - Transformer Model"
echo "=============================================="
echo "Develop Set: $DEVELOP_SET"
echo "Device: $DEVICE"
echo "Time Resolution: $TIME_RESOLUTION hours"
echo "Prediction Windows: ${PREDICTION_WINDOWS[*]}"
echo ""
echo "DataLoader Settings:"
echo "  num_workers: $NUM_WORKERS"
echo "  dataloader_batch_size: $DATALOADER_BATCH_SIZE"
echo "  gpu_batch_size: $GPU_BATCH_SIZE"
echo "  prefetch_factor: $PREFETCH_FACTOR"
echo "=============================================="
echo ""

# Loop through all prediction windows
for PREDICTION_WINDOW_SIZE in "${PREDICTION_WINDOWS[@]}"; do
    echo ""
    echo "----------------------------------------------"
    echo "Processing prediction_window_${PREDICTION_WINDOW_SIZE}"
    echo "----------------------------------------------"
    
    python experiment_online_inference.py \
        --develop_set "$DEVELOP_SET" \
        --prediction_window_size "$PREDICTION_WINDOW_SIZE" \
        --device "$DEVICE" \
        --time_resolution "$TIME_RESOLUTION" \
        --min_observations 1 \
        --num_workers "$NUM_WORKERS" \
        --dataloader_batch_size "$DATALOADER_BATCH_SIZE" \
        --gpu_batch_size "$GPU_BATCH_SIZE" \
        --prefetch_factor "$PREFETCH_FACTOR"
    
    echo "Completed prediction_window_${PREDICTION_WINDOW_SIZE}"
done

echo ""
echo "=============================================="
echo "All prediction windows completed!"
echo "=============================================="
