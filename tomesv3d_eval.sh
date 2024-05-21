#!/usr/bin/env bash

# Run SV3D-u inference using different tomesd token merging ratios
# fm_ratio=(0.1 0.2 0.3 0.4 0.5 0.6 0.62 0.65 0.75 0.8)
compress_ratio=(0.20)

for ratio in ${compress_ratio[@]}; do
    # Print a dashed line containing the current tomesd token merging ratio
    echo "--------------------------------------------------"
    echo "Running SV3D-u inference using tomesd for compression: ${ratio}"
    echo "--------------------------------------------------"
    accelerate launch --config_file accelerate_config.yaml \
        scripts/sampling/simple_video_sample.py \
        --input_path ~/sv3d_test \
        --version sv3d_u \
        --output_folder /blob/zyhe/muaz/output/sv3d_u \
        --fm_ratio ${ratio} \
        --tm_ratio 0.6 \
        --bm_ratio 0.5 \
        --logger_projectname sv3d_inference_with_compression \
        --exp_name fm_compression_${ratio}_tm_0.6_bm_0.5 \
        --verbose True 2>&1 | tee /blob/zyhe/muaz/output/sv3d_u/fm_${ratio}_bm_0.5_tm_0.6.log
done