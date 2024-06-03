#!/usr/bin/env bash

# Run SV3D-u inference using different tomesd token merging ratios
# fm_ratio=(0.1 0.2 0.3 0.4 0.5 0.6 0.62 0.65 0.75 0.8)
compress_ratio=(0.2)
tm=0.40
bm=0.40
output=/home/yifanyang/container_us/zyhe/muaz/sv3d_u/full_inference

for ratio in ${compress_ratio[@]}; do
    # Print a dashed line containing the current tomesd token merging ratio
    echo "--------------------------------------------------"
    echo "Running SV3D-u inference using tomesd for compression: ${ratio}"
    echo "--------------------------------------------------"
    CUDA_VISIBLE_DEVICES=1 accelerate launch --config_file accelerate_config.yaml \
        --num_processes 1 \
        scripts/sampling/benchmark_simple_video.py \
        --input_path ~/sv3d_test/0.png \
        --version sv3d_u \
        --output_folder ${output} \
        --fm_ratio ${ratio} \
        --tm_ratio ${tm} \
        --bm_ratio ${bm} \
        --bypass_saving True\
        --bypass_tomesd True \
        --logger_projectname sv3d_inference_with_compression \
        --exp_name fm_${ratio}_tm_${tm}_bm_${bm} \
        --verbose True 
done