#!/bin/bash
# Online Inference script for LSTM with Attention model
# This script runs online inference (predictions at each timestamp)

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Default values
DEVELOP_SET="ilsan"
DEVICE="0"
TIME_RESOLUTION=1.0
NUM_WORKERS=4
GPU_BATCH_SIZE=32
DATALOADER_BATCH_SIZE=16
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
        --time_resolution)
            TIME_RESOLUTION="$2"
            shift 2
            ;;
        --num_workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "Online Inference - LSTM with Attention Model"
echo "=============================================="
echo "Develop Set: $DEVELOP_SET"
echo "Device: $DEVICE"
echo "Time Resolution: $TIME_RESOLUTION hours"
echo "Num Workers: $NUM_WORKERS"
echo "=============================================="

# Run online inference for each prediction window
for PW in 0 48 72; do
    echo ""
    echo ">>> Online inference for prediction_window_${PW}..."
    echo ""
    
    python experiment_online_inference.py \
        --develop_set $DEVELOP_SET \
        --prediction_window_size $PW \
        --device $DEVICE \
        --time_resolution $TIME_RESOLUTION \
        --num_workers $NUM_WORKERS \
        --gpu_batch_size $GPU_BATCH_SIZE \
        --dataloader_batch_size $DATALOADER_BATCH_SIZE \
        --exp_nm $EXP_NM
        
    echo ""
    echo ">>> Completed prediction_window_${PW}"
    echo ""
done

echo "=============================================="
echo "Online inference completed for all prediction windows!"
echo "=============================================="

