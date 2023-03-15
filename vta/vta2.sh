#!/usr/bin/env bash

# sbatch vta.sh train_mdl.py \
#     --coding_len_coeff=0.1 \
#     --use_abs_pos_kl=1.0 \
#     --batch-size=512 \
#     --name=color \
#     --seed=1 \
#     --dataset-path=./data/simple_colors.npy \
#     --max-iters=40000

# sbatch vta.sh train_mdl.py \
#     --coding_len_coeff=0.1 \
#     --use_abs_pos_kl=1.0 \
#     --batch-size=512 \
#     --name=conditional_color \
#     --seed=1 \
#     --dataset-path=./data/conditional_colors.npy \
#     --max-iters=40000

sbatch vta.sh train_rl.py \
    --coding_len_coeff=0.001 \
    --kl_coeff=0 \
    --rec_coeff=1.0 \
    --use_abs_pos_kl=1.0 \
    --batch-size=64 \
    --name=compile2 \
    --seed=2  \
    --dataset-path=compile  \
    --max-iters=20000 \
    --use_min_length_boundary_mask \

sbatch vta.sh train_rl.py \
    --coding_len_coeff=0.001 \
    --kl_coeff=0 \
    --rec_coeff=1.0 \
    --use_abs_pos_kl=1.0 \
    --batch-size=64 \
    --name=compile3 \
    --seed=3  \
    --dataset-path=compile  \
    --max-iters=20000 \
    --use_min_length_boundary_mask \

sbatch vta.sh train_rl.py \
    --coding_len_coeff=0.001 \
    --kl_coeff=0 \
    --rec_coeff=1.0 \
    --use_abs_pos_kl=1.0 \
    --batch-size=64 \
    --name=compile4 \
    --seed=4  \
    --dataset-path=compile  \
    --max-iters=20000 \
    --use_min_length_boundary_mask \

sbatch vta.sh train_rl.py \
    --coding_len_coeff=0.0005 \
    --kl_coeff=0 \
    --rec_coeff=1.0 \
    --use_abs_pos_kl=1.0 \
    --batch-size=64 \
    --name=bw2 \
    --seed=2  \
    --dataset-path=boxworld  \
    --max-iters=20000 \
    --use_min_length_boundary_mask \

sbatch vta.sh train_rl.py \
    --coding_len_coeff=0.0005 \
    --kl_coeff=0 \
    --rec_coeff=1.0 \
    --use_abs_pos_kl=1.0 \
    --batch-size=64 \
    --name=bw3 \
    --seed=3  \
    --dataset-path=boxworld  \
    --max-iters=20000 \
    --use_min_length_boundary_mask \

sbatch vta.sh train_rl.py \
    --coding_len_coeff=0.0005 \
    --kl_coeff=0 \
    --rec_coeff=1.0 \
    --use_abs_pos_kl=1.0 \
    --batch-size=64 \
    --name=bw4 \
    --seed=4  \
    --dataset-path=boxworld  \
    --max-iters=20000 \
    --use_min_length_boundary_mask \

sbatch vta.sh train_rl.py \
    --coding_len_coeff=0.001 \
    --kl_coeff=0 \
    --rec_coeff=1.0 \
    --use_abs_pos_kl=1.0 \
    --batch-size=64 \
    --name=bw2_coding001 \
    --seed=2  \
    --dataset-path=boxworld  \
    --max-iters=20000 \
    --use_min_length_boundary_mask \

sbatch vta.sh train_rl.py \
    --coding_len_coeff=0.001 \
    --kl_coeff=0 \
    --rec_coeff=1.0 \
    --use_abs_pos_kl=1.0 \
    --batch-size=64 \
    --name=bw3_coding001 \
    --seed=3  \
    --dataset-path=boxworld  \
    --max-iters=20000 \
    --use_min_length_boundary_mask \

sbatch vta.sh train_rl.py \
    --coding_len_coeff=0.001 \
    --kl_coeff=0 \
    --rec_coeff=1.0 \
    --use_abs_pos_kl=1.0 \
    --batch-size=64 \
    --name=bw4_coding001 \
    --seed=4  \
    --dataset-path=boxworld  \
    --max-iters=20000 \
    --use_min_length_boundary_mask \

