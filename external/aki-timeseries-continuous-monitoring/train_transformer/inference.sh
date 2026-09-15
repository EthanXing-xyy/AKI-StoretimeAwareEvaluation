for develop_set in ilsan cchlmc; do
    for prediction_window_size in 0 48 72; do
        python experiment_inference.py \
            --develop_set $develop_set \
            --prediction_window_size $prediction_window_size \
            --device 1 \
            --max_epoch 50 \
            --batch_size 128\
            --num_workers 16 \
            --data_dir ../../data/processed \
            --save_dir_predictions ../../result/predictions \
            --exp_nm ite_transformer
    done
done