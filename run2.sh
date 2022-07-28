#!/usr/bin/env bash

# sbatch --partition ellis run.sh --batch_size 128 --neurosym --n 20000
sbatch run.sh --batch_size 128 --neurosym --n 20000 --length 2

# sbatch --partition ellis run.sh --batch_size 128 --neurosym --n 20000

# sbatch run.sh --model hmm-homo --abstract_pen 1 --lr 8E-4 --seed 1
# sbatch run.sh --model hmm-homo --abstract_pen 1 --lr 8E-4 --seed 2
# sbatch run.sh --model hmm-homo --abstract_pen 1 --lr 8E-4 --seed 3
# sbatch run.sh --model hmm-homo --abstract_pen 1 --lr 8E-4 --seed 4
# sbatch run.sh --model hmm-homo --abstract_pen 1 --lr 8E-4 --seed 5

# sbatch run.sh --model hmm-homo --abstract_pen 1 --lr 1E-4 --seed 1
# sbatch run.sh --model hmm-homo --abstract_pen 1 --lr 1E-4 --seed 2
# sbatch run.sh --model hmm-homo --abstract_pen 1 --lr 1E-4 --seed 3
# sbatch run.sh --model hmm-homo --abstract_pen 1 --lr 1E-4 --seed 4
# sbatch run.sh --model hmm-homo --abstract_pen 1 --lr 1E-4 --seed 5
