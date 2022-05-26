#!/usr/bin/env bash

mkdir -p results

CUDA_VISIBLE_DEVICES=0 \
	python src/summarization.py \
	--model_name_or_path facebook/bart-large-cnn \
	--do_train --do_eval --do_predict \
	--train_file      data/filtered_controlled_data/train_rate_concat_sent-ctrl.csv \
	--validation_file data/filtered_controlled_data/val_rate_concat_sent-ctrl.csv \
	--test_file       data/filtered_controlled_data/test_rate_concat_sent-ctrl.csv \
	--output_dir      results/rate_concat_1024_sent-ctrl \
	--overwrite_output_dir \
	--per_device_train_batch_size=1 \
	--per_device_eval_batch_size=1 \
	--predict_with_generate \
	--seed 0 --save_total_limit 1 \
	--max_source_length 1024 --max_target_length 400 --gen_target_min 20 \
	2>&1 | tee summarization.log

