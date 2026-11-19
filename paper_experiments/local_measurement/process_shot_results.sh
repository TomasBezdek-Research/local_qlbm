#!/bin/bash

# Exit immediately if any command exits with a non-zero status
set -e

# Parse command line arguments for log_scale
LOG_SCALE_FLAG=""
NAME_SUFFIX=""

for arg in "$@"; do
    if [ "$arg" == "--log-scale" ] || [ "$arg" == "log_scale=true" ]; then
        LOG_SCALE_FLAG="--log_scale True"
        NAME_SUFFIX="_log"
    fi
done

echo "=================================================="
echo "Starting all experiment processing scripts..."
if [ -n "$LOG_SCALE_FLAG" ]; then
    echo "Mode: Log scale enabled (suffix: _log)"
fi
echo "=================================================="

# --------------------------------------------------
# Part 1: Simple Sampling Scaling Cuts
# --------------------------------------------------
echo ""
echo "--- [1/4] Running Simple Sampling Experiments ---"
echo -n "[1/3] "
uv run ./process_results_shot_scaling.py ./scaling_output/shot_scaling_simplesampling_point.csv sampling_point${NAME_SUFFIX} --ground_true_value 0.111111 ${LOG_SCALE_FLAG}

echo -n "[2/3] "
uv run ./process_results_shot_scaling.py ./scaling_output/shot_scaling_simplesampling_cloud.csv sampling_cloud${NAME_SUFFIX} --ground_true_value 0.333333 ${LOG_SCALE_FLAG}

echo -n "[3/3] "
uv run ./process_results_shot_scaling.py ./scaling_output/shot_scaling_simplesampling_interval.csv sampling_interval${NAME_SUFFIX} --ground_true_value 0.444444 ${LOG_SCALE_FLAG}


# --------------------------------------------------
# Part 2: OBPA Scaling Cuts
# --------------------------------------------------
echo ""
echo "--- [2/4] Running OBPA Experiments ---"
echo -n "[1/3] "
uv run ./process_results_shot_scaling.py ./scaling_output/shot_scaling_obpa_point.csv obpa_point${NAME_SUFFIX} --ground_true_value 0.111111 ${LOG_SCALE_FLAG}

echo -n "[2/3] "
uv run ./process_results_shot_scaling.py ./scaling_output/shot_scaling_obpa_interval.csv obpa_interval${NAME_SUFFIX} --ground_true_value 0.444444 ${LOG_SCALE_FLAG}

echo -n "[3/3] "
uv run ./process_results_shot_scaling.py ./scaling_output/shot_scaling_obpa_cloud.csv obpa_cloud${NAME_SUFFIX} --ground_true_value 0.333333 ${LOG_SCALE_FLAG}


# --------------------------------------------------
# Part 3: IQAE Scaling Inverse
# --------------------------------------------------
echo ""
echo "--- [3/4] Running IQAE Inverse Experiments ---"
echo -n "[1/4] "
uv run ./process_results_iqae_scaling.py ./scaling_output/scaling_iqae_cloud.csv cloud${NAME_SUFFIX} ${LOG_SCALE_FLAG}

echo -n "[2/4] "
uv run ./process_results_iqae_scaling.py ./scaling_output/scaling_iqae_interval.csv interval${NAME_SUFFIX} ${LOG_SCALE_FLAG}

echo -n "[3/4] "
uv run ./process_results_iqae_scaling.py ./scaling_output/scaling_iqae_zero.csv zero${NAME_SUFFIX} ${LOG_SCALE_FLAG}

echo -n "[4/4] "
uv run ./process_results_iqae_scaling.py ./scaling_output/scaling_iqae_point.csv point${NAME_SUFFIX} ${LOG_SCALE_FLAG}


# --------------------------------------------------
# Part 4: IQAE Gate Scaling
# --------------------------------------------------
echo ""
echo "--- [4/4] Running IQAE Gate Scaling Experiments ---"
uv run ./process_results_iqae_gates.py ./scaling_output/scaling_iqae_gates.csv


echo ""
echo "=================================================="
echo "All experiments successfully processed and saved!"
echo "=================================================="
