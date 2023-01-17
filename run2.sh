#!/usr/bin/env bash

sbatch run.sh --model hmm --traj_updates 2E7 --lr 1E-4 --abstract_pen 0.2
sbatch run.sh --model hmm --traj_updates 2E7 --lr 1E-4 --abstract_pen 0.1
sbatch run.sh --model hmm --traj_updates 2E7 --lr 1E-4 --abstract_pen 0.3

