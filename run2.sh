#!/usr/bin/env bash

# sbatch --partition ellis run.sh --load --model cc --fine_tune --n 1000 --ellis
# sbatch --partition ellis run.sh --load --model cc --fine_tune --n 2000 --ellis
# sbatch --partition ellis run.sh --load --model cc --fine_tune --n 5000 --ellis

sbatch run.sh --muzero --n 500 --test_every 60  --load  --length 4 --muzero_scratch
sbatch run.sh --muzero --n 500 --test_every 60  --load  --length 4
sbatch run.sh --muzero --n 500 --test_every 60  --load  --length 4 --freeze_tau
