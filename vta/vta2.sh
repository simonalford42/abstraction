#!/usr/bin/env bash

sbatch vta.sh train_rl.py \
    --coding_len_coeff=0.001 \
    --kl_coeff=0 \
    --rec_coeff=1.0 \
    --use_abs_pos_kl=1.0 \
    --batch-size=64 \
    --name=bw3-21 \
    --seed=1  \
    --dataset-path=boxworld  \
    --max-iters=20000 \
    --use_min_length_boundary_mask \

sbatch vta.sh train_rl.py \
    --coding_len_coeff=0.001 \
    --kl_coeff=0 \
    --rec_coeff=1.0 \
    --use_abs_pos_kl=1.0 \
    --batch-size=64 \
    --name=bw3-21 \
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
    --name=bw3-21 \
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
    --name=bw3-21 \
    --seed=1  \
    --dataset-path=boxworld  \
    --max-iters=20000 \
    --use_min_length_boundary_mask \
    --attention_enc \

sbatch vta.sh train_rl.py \
    --coding_len_coeff=0.001 \
    --kl_coeff=0 \
    --rec_coeff=1.0 \
    --use_abs_pos_kl=1.0 \
    --batch-size=64 \
    --name=bw3-21 \
    --seed=2  \
    --dataset-path=boxworld  \
    --max-iters=20000 \
    --use_min_length_boundary_mask \
    --attention_enc \

sbatch vta.sh train_rl.py \
    --coding_len_coeff=0.001 \
    --kl_coeff=0 \
    --rec_coeff=1.0 \
    --use_abs_pos_kl=1.0 \
    --batch-size=64 \
    --name=bw3-21 \
    --seed=3  \
    --dataset-path=boxworld  \
    --max-iters=20000 \
    --use_min_length_boundary_mask \
    --attention_enc \

sbatch vta.sh train_rl.py \
    --coding_len_coeff=0.001 \
    --kl_coeff=0 \
    --rec_coeff=1.0 \
    --use_abs_pos_kl=1.0 \
    --batch-size=64 \
    --name=bw3-21 \
    --seed=1  \
    --dataset-path=boxworld  \
    --max-iters=20000 \
    --use_min_length_boundary_mask \
    --attention_enc \
    --second_enc \

sbatch vta.sh train_rl.py \
    --coding_len_coeff=0.001 \
    --kl_coeff=0 \
    --rec_coeff=1.0 \
    --use_abs_pos_kl=1.0 \
    --batch-size=64 \
    --name=bw3-21 \
    --seed=2  \
    --dataset-path=boxworld  \
    --max-iters=20000 \
    --use_min_length_boundary_mask \
    --attention_enc \
    --second_enc \

sbatch vta.sh train_rl.py \
    --coding_len_coeff=0.001 \
    --kl_coeff=0 \
    --rec_coeff=1.0 \
    --use_abs_pos_kl=1.0 \
    --batch-size=64 \
    --name=bw3-21 \
    --seed=3  \
    --dataset-path=boxworld  \
    --max-iters=20000 \
    --use_min_length_boundary_mask \
    --attention_enc \
    --second_enc \
