#!/usr/bin/env bash

sbatch run.sh --sv_options --traj_updates 1E8 --model fc
sbatch run.sh --sv_options --traj_updates 1E8 --model fc
sbatch run.sh --sv_options --traj_updates 1E8 --model attn
sbatch run.sh --sv_options --traj_updates 1E8 --model attn
# sbatch run.sh --model hmm-homo --abstract_pen 1 --dim 64 --num_attn_blocks 2 --num_heads 4 --traj_updates 2E7
# sbatch run.sh --model hmm-homo --abstract_pen 1 --dim 64 --num_attn_blocks 2 --num_heads 4 --traj_updates 2E7
# sbatch run.sh --model hmm-homo --abstract_pen 1 --dim 64 --num_attn_blocks 4 --num_heads 4 --traj_updates 2E7
# sbatch run.sh --model hmm-homo --abstract_pen 1 --dim 128 --num_attn_blocks 4 --num_heads 4 --traj_updates 2E7
# sbatch run.sh --model hmm-homo --abstract_pen 1 --dim 64 --num_attn_blocks 4 --num_heads 8 --traj_updates 2E7
# sbatch run.sh --model hmm-homo --abstract_pen 1 --dim 128 --num_attn_blocks 4 --num_heads 8 --traj_updates 2E7
# sbatch run.sh --model hmm-homo --abstract_pen 1 --dim 64 --num_attn_blocks 8 --num_heads 8 --traj_updates 2E7
# sbatch run.sh --model hmm-homo --abstract_pen 1 --dim 128 --num_attn_blocks 8 --num_heads 8 --traj_updates 2E7
# sbatch run.sh --model hmm-homo --abstract_pen 1 --dim 64 --num_attn_blocks 8 --num_heads 16 --traj_updates 2E7
# sbatch run.sh --model hmm-homo --abstract_pen 1 --dim 128 --num_attn_blocks 8 --num_heads 16 --traj_updates 2E7
# sbatch run.sh --model hmm-homo --abstract_pen 1 --dim 64 --num_attn_blocks 4 --num_heads 16 --traj_updates 2E7
# sbatch run.sh --model hmm-homo --abstract_pen 1 --dim 128 --num_attn_blocks 4 --num_heads 16 --traj_updates 2E7

# sbatch run.sh --model hmm-homo --abstract_pen 1 --traj_updates 4E7 --lr 4E-4
# sbatch run.sh --model hmm-homo --abstract_pen 1 --traj_updates 4E7 --lr 4E-4
# sbatch run.sh --model hmm-homo --abstract_pen 1 --traj_updates 4E7 --lr 4E-4
