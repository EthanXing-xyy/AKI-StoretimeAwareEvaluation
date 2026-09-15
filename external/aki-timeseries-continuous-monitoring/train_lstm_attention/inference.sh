#!/bin/bash
# Inference script for LSTM with Attention model
# This script runs inference on trained models

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Default values
DEVELOP_SET="ilsan"
DEVICE="0"
BATCH_SIZE=32
NUM_WORKERS=8
EXP_NM="lstm_attention"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --develop_set)
            DEVELOP_SET="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "Inference - LSTM with Attention Model"
echo "=============================================="
echo "Develop Set: $DEVELOP_SET"
echo "Device: $DEVICE"
echo "Batch Size: $BATCH_SIZE"
echo "=============================================="

# Run inference for each prediction window
for PW in 0 48 72; do
    echo ""
    echo ">>> Inference for prediction_window_${PW}..."
    echo ""
    
    python experiment_inference.py \
        --develop_set $DEVELOP_SET \
        --prediction_window_size $PW \
        --device $DEVICE \
        --batch_size $BATCH_SIZE \
        --num_workers $NUM_WORKERS \
        --exp_nm $EXP_NM
        
    echo ""
    echo ">>> Completed prediction_window_${PW}"
    echo ""
done

echo "=============================================="
echo "Inference completed for all prediction windows!"
echo "=============================================="

