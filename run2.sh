#!/usr/bin/env bash

# sbatch --partition ellis run.sh --model cc --rnn_macro --ellis
# sbatch --partition ellis run.sh --model cc --ellis
# sbatch run.sh --model cc --rnn_macro
# sbatch run.sh --model cc

# sbatch run.sh --model sv
sbatch run.sh --model hmm --abstract_pen 1.0 --lr 3E-4 --traj_updates 1E8
sbatch run.sh --model hmm --abstract_pen 1.0 --lr 1E-4 --traj_updates 1E8
sbatch run.sh --model hmm --abstract_pen 1.0 --lr 8E-4 --traj_updates 1E8
