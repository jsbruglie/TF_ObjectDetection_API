#!/usr/bin/bash

# Dataset root dir
base_dir=$1
# Output root dir
out_dir=$2
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
	python2 preproc.py \
		--img_ext jpg \
		--in_img_dir ${base_dir}/${chunk}/images \
		--in_xml_dir ${base_dir}/${chunk}/annotations \
		--out_csv ${out_dir}/${chunk}/${chunk}.csv \
		--out_img_dir ${out_dir}/${chunk}/images_resize
done
