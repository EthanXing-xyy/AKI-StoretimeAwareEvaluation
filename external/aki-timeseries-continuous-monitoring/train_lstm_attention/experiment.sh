#!/bin/bash
# Training script for LSTM with Attention model
# This script trains the model on different prediction windows

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Default values
DEVELOP_SET="ilsan"
DEVICE="0"
MAX_EPOCH=50
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
        --max_epoch)
            MAX_EPOCH="$2"
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
echo "Training LSTM with Attention Model"
echo "=============================================="
echo "Develop Set: $DEVELOP_SET"
echo "Device: $DEVICE"
echo "Max Epochs: $MAX_EPOCH"
echo "Batch Size: $BATCH_SIZE"
echo "=============================================="

# Train for each prediction window
for PW in 0 48 72; do
    echo ""
    echo ">>> Training prediction_window_${PW}..."
    echo ""
    
    python experiment_train.py \
        --develop_set $DEVELOP_SET \
        --prediction_window_size $PW \
        --device $DEVICE \
        --max_epoch $MAX_EPOCH \
        --batch_size $BATCH_SIZE \
        --num_workers $NUM_WORKERS \
        --exp_nm $EXP_NM \
        --d_model 64 \
        --max_seq_len 1024 \
        --gradient_checkpointing
        
    echo ""
    echo ">>> Completed prediction_window_${PW}"
    echo ""
done

echo "=============================================="
echo "Training completed for all prediction windows!"
echo "=============================================="

bash inference.sh
bash online_inference.sh