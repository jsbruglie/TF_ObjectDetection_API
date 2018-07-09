#!/usr/bin/bash

# Workspace dir
ws=$1

# Setup environment variables
#wd=`pwd`"/models/research"
#export PYTHONPATH=$PYTHONPATH:${wd}:${wd}/slim
#protoc ${wd}/object_detection/protos/*.proto --python_out=${wd}

# Trials
declare -a trials=(
   "sim_no_camera_base"
   "sim_no_camera_chess"
   "sim_no_camera_flat"
   "sim_no_camera_gradient"
   "sim_no_camera_perlin")


# Train from ImageNet runs
for trial in "${trials[@]}"
do
    # Launch eval (on CPU) first and send to background
    (export CUDA_VISIBLE_DEVICES=3; \
    python models/research/object_detection/eval.py \
        --logtostderr \
        --eval_dir=${ws}/eval/${trial}/ \
        --pipeline_config_path=${ws}/config/${trial}.config \
        --checkpoint_dir=${ws}/train/${trial}/ &)

    # Launch train and wait for conclusion
    python models/research/object_detection/train.py \
        --logtostderr \
        --train_dir=${ws}/train/${trial}/ \
        --pipeline_config_path=${ws}/config/${trial}.config \
        --num_clones=2 \
        --ps_tasks=1

    # Kill eval process
    wait 5m
    pkill -f eval.py

done

# Export inference graph
for trial in "${trials[@]}"
do
    python models/research/object_detection/export_inference_graph.py \
        --input_type image_tensor \
        --pipeline_config_path ${ws}/config/${trial}.config \
        --trained_checkpoint_prefix ${ws}/train/${trial}/model.ckpt-100000 \
        --output_directory ${ws}/inference_graph/${trial}  
done

# Fine tune runs
for trial in "${trials[@]}"
do
    # Launch eval (on CPU) first and send to background
    (export CUDA_VISIBLE_DEVICES=3; \
    python models/research/object_detection/eval.py \
        --logtostderr \
        --eval_dir=${ws}/eval/fine_tune/${trial}/ \
        --pipeline_config_path=${ws}/config/fine_tune_${trial}.config \
        --checkpoint_dir=${ws}/train/fine_tune/${trial}/ &)

    # Launch train and wait for conclusion
    python models/research/object_detection/train.py \
        --logtostderr \
        --train_dir=${ws}/train/fine_tune/${trial}/ \
        --pipeline_config_path=${ws}/config/fine_tune_${trial}.config \
        --num_clones=2 \
        --ps_tasks=1

    # Kill eval process
    wait 5m
    pkill -f eval.py

done

# Export inference graph
for trial in "${trials[@]}"
do
    python models/research/object_detection/export_inference_graph.py \
        --input_type image_tensor \
        --pipeline_config_path ${ws}/config/fine_tune_${trial}.config \
        --trained_checkpoint_prefix ${ws}/train/fine_tune/${trial}/model.ckpt-25000 \
        --output_directory ${ws}/inference_graph/fine_tune/${trial}  
done

# Run metrics
for trial in "${trials[@]}"
do
    mkdir -p ${ws}/inference_results/fine_tune/${trial}

    python -m object_detection/inference/infer_detections \
        --input_tfrecord_paths=${ws}/data/test.record \
        --output_tfrecord_path=${ws}/inference_results/fine_tune/${trial}/detections.tfrecord \
        --inference_graph=${ws}/inference_graph/fine_tune/${trial}/frozen_inference_graph.pb \
        --discard_image_pixels

    python -m object_detection/metrics/offline_eval_map_corloc \
        --eval_dir=${ws}/inference_results/fine_tune/${trial}/ \
        --eval_config_path=test_eval_config.pbtxt \
        --input_config_path=${ws}/test_input_config/fine_tune_${trial}.pbtxt 

done
