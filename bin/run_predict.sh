#!/usr/bin/env bash

mkdir -p results/predict/html

args="\
	--model_name_or_path results/rate_concat_1024_sent-ctrl-feng/checkpoint-16000/ \
	--do_predict \
	--test_file       data/predict/input.csv \
	--output_dir      results/predict \
	--attention_html  results/predict/html \
	--predict_with_generate \
	--max_source_length 1024 --max_target_length 400 --gen_target_min 20 \
"

echo "Arguments:" $args

if [ -n "$DEBUG" ]; then
	CUDA_VISIBLE_DEVICES=0 \
		python -m pdb src/summarization.py $args
else
	CUDA_VISIBLE_DEVICES=0 \
		python src/summarization.py $args 2>&1 | tee predict.log
fi
