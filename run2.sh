#!/usr/bin/env bash

# sbatch run.sh --model cc --abstract_dim 32 --tau_noise 0.01 --abstract_pen 1 --ellis
# sbatch run.sh --model cc --abstract_dim 32 --tau_noise 0.02 --abstract_pen 1 --ellis
# sbatch run.sh --model cc --abstract_dim 32 --tau_noise 0.03 --abstract_pen 1

sbatch run.sh --model ccts --abstract_dim 32 --abstract_pen 1
# sbatch run.sh --model ccts --abstract_dim 32 --abstract_pen 1
# sbatch run.sh --model ccts --abstract_dim 32 --abstract_pen 0
# sbatch run.sh --model ccts --abstract_dim 32 --abstract_pen 0

# sbatch run.sh --model hmm-homo --abstract_pen 0
# sbatch run.sh --model hmm-homo --abstract_pen 0
