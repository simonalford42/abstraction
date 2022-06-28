#!/usr/bin/env bash

# sbatch run.sh --muzero --load --n 100 --model cc --length 4
# sbatch run.sh --muzero --load --n 500 --model cc --length 4
# sbatch run.sh --muzero --load --n 1000 --model cc --length 4
# sbatch run.sh --muzero --load --n 2000 --model cc --length 4
# sbatch run.sh --muzero --load --n 5000 --model cc --length 4
sbatch run.sh --muzero --load --n 10000 --model cc --length 4
# sbatch run.sh --muzero --load --n 20000 --model cc --length 4
# sbatch run.sh --model cc --abstract_dim 128 --model hmm --abstract_pen 1 
# sbatch run.sh --model cc --abstract_dim 128 --model cc --abstract_pen 1
# sbatch run.sh --model cc --abstract_dim 256 --model hmm --abstract_pen 1
# sbatch run.sh --model cc --abstract_dim 256 --model cc --abstract_pen 1
