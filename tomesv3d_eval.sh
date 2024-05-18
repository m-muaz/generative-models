#!/usr/bin/env bash

# Run SV3D-u inference using different tomesd token merging ratios
tomesd_token_ratios=(0.1 0.2 0.3 0.4 0.5 0.6 0.75)

for ratio in ${tomesd_token_ratios[@]}; do
    # Print a dashed line containing the current tomesd token merging ratio
    echo "--------------------------------------------------"
    echo "Running SV3D-u inference using tomesd token merging ratio: ${ratio}"
    echo "--------------------------------------------------"
    python scripts/sampling/simple_video_sample.py \
        --input_path ~/sv3d_test \
        --version sv3d_u \
        --output_folder /blob/zyhe/muaz/output/sv3d_u \
        --tomesd_ratio ${ratio} \
        --verbose True 2>&1 | tee /blob/zyhe/muaz/output/sv3d_u/tomesd_ratio_${ratio}.log
done