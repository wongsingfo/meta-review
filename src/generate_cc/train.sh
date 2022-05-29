#!/bin/bash


python classification.py \
            --dataPath "./generate_data/" \
            --fileModelSave "../results/class_epoch1_f12_bsz32" \
            --numDevice 3 \
            --learning_rate 1e-5 \
            --epochs 1 \
            --train_batch_size 32 \
            --test_batch_size 2 \
            --warmup_steps 100 \
            --dropout_rate 0.1 \
            --weight_decay 0.2 
