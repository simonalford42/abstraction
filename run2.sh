#!/usr/bin/env bash

sbatch run.sh --model hmm --abstract_pen 1 --b 10
sbatch run.sh --model hmm --abstract_pen 1 --b 30
# sbatch run.sh --n 5000 --fine_tune --loss 'kl' --ellis
# sbatch run.sh --n 500 --fine_tune --loss 'kl' --ellis
