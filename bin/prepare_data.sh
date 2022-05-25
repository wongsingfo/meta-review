#!/usr/bin/env bash

# set -x

data_dir="data"

function download() {
	file="$1"
	dir="filtered_controlled_data"
	mkdir -p "$data_dir/$dir"
	output="$data_dir/$dir/$file"
	url="https://raw.githubusercontent.com/Shen-Chenhui/MReD/master/summarization/abstractive/$dir/$file"
	[ -e "$output" ] || { echo "Downloading $url" && curl -L "$url" -o "$output" ; }
}

download train_rate_concat_sent-ctrl.csv
download val_rate_concat_sent-ctrl.csv
download test_rate_concat_sent-ctrl.csv

