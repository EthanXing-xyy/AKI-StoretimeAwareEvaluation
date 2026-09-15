for develop_set in ilsan; do
    for prediction_window_size in 72; do
        for input_seq_len in 256; do
            for apply_prob in 0.25; do
                python experiment_holdout_setting.py \
                    --develop_set $develop_set \
                    --prediction_window_size $prediction_window_size \
                    --input_seq_len $input_seq_len \
                    --apply_prob $apply_prob \
                    --device 0,1 \
                    --max_epoch 50 \
                    --batch_size 1024\
                    --num_workers 8 \
                    --data_dir ../../../data/processed \
                    --save_dir_predictions ../../../result/predictions \
                    --exp_nm rms_norm_cnn
            done
        done
    done
done