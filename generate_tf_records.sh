#!/usr/bin/bash

# Dataset root dir
base_dir=$1

# Dataset chunk names
declare -a chunks=(
	"base"
	"camera"
	"chess"
	"flat"
	"gradient"
	"light" 
	"perlin")

for chunk in "${chunks[@]}"
do
	python generate_tfrecord.py \
		--data_path ${base_dir}/${chunk}/ \
		--images_path ${base_dir}/${chunk}/images_resize \
		--csv_path ${base_dir}/${chunk}/test.csv \
		--filename ${chunk}
done