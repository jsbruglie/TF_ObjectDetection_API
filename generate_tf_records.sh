#!/usr/bin/bash

# Dataset root dir
base_dir=$1

# Dataset chunk names
declare -a chunks=(
	"sim_no_camera_base"
	"sim_no_camera_chess"
	"sim_no_camera_flat"
	"sim_no_camera_gradient"
	"sim_no_camera_perlin")

for chunk in "${chunks[@]}"
do
	python generate_tfrecord.py \
		--data_path ${base_dir}/${chunk}/ \
		--images_path ${base_dir}/${chunk}/images_resize \
		--csv_path ${base_dir}/${chunk}/${chunk}.csv \
		--filename ${chunk}
done