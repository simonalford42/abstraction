#!/usr/bin/env bash
# sbatch --partition=ellis run.sh --fake_cc_neurosym --model hmm --abstract_pen 1 --ellis --save_every 60
sbatch --partition=ellis run.sh --fake_cc_neurosym --model hmm --abstract_pen 1 --ellis --relational_macro --save_every 60

# sbatch run.sh --model hmm --abstract_pen 1 --abstract_dim 128
# sbatch run.sh --model hmm --abstract_pen 1 --abstract_dim 256
# sbatch run.sh --model hmm --abstract_pen 1 --abstract_dim 324
# sbatch run.sh --model hmm --abstract_pen 1 --abstract_dim 128
# sbatch run.sh --model hmm --abstract_pen 1 --abstract_dim 256
# sbatch run.sh --model hmm --abstract_pen 1 --abstract_dim 324
