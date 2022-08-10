#!/usr/bin/env bash

# sbatch run.sh --neurosym --traj_updates 1E8
# sbatch run.sh --neurosym --traj_updates 1E8


sbatch run.sh --model hmm-homo --abstract_pen 1 --traj_updates 2E7 --length '(1, )'
sbatch run.sh --model hmm-homo --abstract_pen 1 --traj_updates 2E7 --length '(1, )'
sbatch run.sh --model hmm-homo --abstract_pen 1 --traj_updates 2E7 --length '(2, )'
sbatch run.sh --model hmm-homo --abstract_pen 1 --traj_updates 2E7 --length '(2, )'
sbatch run.sh --model hmm-homo --abstract_pen 1 --traj_updates 2E7 --length '(3, )'
sbatch run.sh --model hmm-homo --abstract_pen 1 --traj_updates 2E7 --length '(3, )'
sbatch run.sh --model hmm-homo --abstract_pen 1 --traj_updates 2E7 --length '(4, )'
sbatch run.sh --model hmm-homo --abstract_pen 1 --traj_updates 2E7 --length '(4, )'

sbatch run.sh --model hmm-homo --abstract_pen 1 --traj_updates 2E7 --length '(3, 4, )'
sbatch run.sh --model hmm-homo --abstract_pen 1 --traj_updates 2E7 --length '(2, 3, 4, )'
sbatch run.sh --model hmm-homo --abstract_pen 1 --traj_updates 2E7 --length '(1, 2, 3, 4, )'
