for develop_set in ilsan; do
    for prediction_window_size in 0 48 72; do
        for input_seq_len in 256; do
            for apply_prob in 0.0; do
                python experiment_holdout_setting.py \
                    --develop_set $develop_set \
                    --prediction_window_size $prediction_window_size \
                    --input_seq_len $input_seq_len \
                    --apply_prob $apply_prob \
                    --device 0 \
                    --max_epoch 50 \
                    --batch_size 1024\
                    --num_workers 16 \
                    --data_dir ../../../data/processed \
                    --save_dir_predictions ../../../result/predictions \
                    --exp_nm mask_rms_cnn
            done
        done
    done
done

bash inference.sh
bash online_inference.sh