#!/usr/bin/env bash
sbatch --partition-ellis run.sh --sv_micro --sv_micro_data_type ground_truth
sbatch --partition-ellis run.sh --sv_micro --sv_micro_data_type ground_truth
sbatch --partition-ellis run.sh --sv_micro --sv_micro_data_type from_model
sbatch --partition-ellis run.sh --sv_micro --sv_micro_data_type full_traj

# sbatch run.sh --model hmm --abstract_pen 1 --abstract_dim 128
# sbatch run.sh --model hmm --abstract_pen 1 --abstract_dim 256
# sbatch run.sh --model hmm --abstract_pen 1 --abstract_dim 324
# sbatch run.sh --model hmm --abstract_pen 1 --abstract_dim 128
# sbatch run.sh --model hmm --abstract_pen 1 --abstract_dim 256
# sbatch run.sh --model hmm --abstract_pen 1 --abstract_dim 324
