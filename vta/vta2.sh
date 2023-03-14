#!/usr/bin/env bash

sbatch vta.sh train_mdl.py \
    --coding_len_coeff=0.1 \
    --use_abs_pos_kl=1.0 \
    --batch-size=512 \
    --name=color \
    --seed=1 \
    --dataset-path=./data/simple_colors.npy \
    --max-iters=40000

sbatch vta.sh train_mdl.py \
    --coding_len_coeff=0.1 \
    --use_abs_pos_kl=1.0 \
    --batch-size=512 \
    --name=conditional_color \
    --seed=1 \
    --dataset-path=./data/conditional_colors.npy \
    --max-iters=40000

# sbatch vta.sh train_rl.py \
#     --coding_len_coeff=0.001 \
#     --kl_coeff=0 \
#     --rec_coeff=1.0 \
#     --use_abs_pos_kl=1.0 \
#     --batch-size=64 \
#     --name=love+1 \
#     --seed=1  \
#     --dataset-path=compile  \
#     --max-iters=20000 \
#     --use_min_length_boundary_mask \

