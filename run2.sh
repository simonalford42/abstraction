#!/usr/bin/env bash

sbatch run.sh --neurosym --length 4 --dim 128 --num_attn_blocks 4 --num_heads 8 --traj_updates 5E8
sbatch run.sh --neurosym --length 4 --dim 128 --num_attn_blocks 8 --num_heads 8 --traj_updates 5E8
sbatch run.sh --neurosym --length 4 --dim 128 --num_attn_blocks 8 --num_heads 16 --traj_updates 5E8
sbatch run.sh --neurosym --length 4 --dim 128 --num_attn_blocks 4 --num_heads 16 --traj_updates 5E8
