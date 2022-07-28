#!/usr/bin/env bash

# sbatch --partition ellis run.sh --batch_size 128 --neurosym --n 20000
# sbatch --partition ellis run.sh --batch_size 128 --neurosym --n 20000


sbatch run.sh --model hmm-homo --abstract_pen 1 --lr 8E-4 --seed 1
sbatch run.sh --model hmm-homo --abstract_pen 1 --lr 8E-4 --seed 2
sbatch run.sh --model hmm-homo --abstract_pen 1 --lr 8E-4 --seed 3
sbatch run.sh --model hmm-homo --abstract_pen 1 --lr 8E-4 --seed 4
sbatch run.sh --model hmm-homo --abstract_pen 1 --lr 8E-4 --seed 5

sbatch run.sh --model hmm-homo --abstract_pen 1 --lr 1E-4 --seed 1
sbatch run.sh --model hmm-homo --abstract_pen 1 --lr 1E-4 --seed 2
sbatch run.sh --model hmm-homo --abstract_pen 1 --lr 1E-4 --seed 3
sbatch run.sh --model hmm-homo --abstract_pen 1 --lr 1E-4 --seed 4
sbatch run.sh --model hmm-homo --abstract_pen 1 --lr 1E-4 --seed 5

# sbatch --partition ellis run.sh --model hmm --ellis --b 21 --abstract_pen 1
# sbatch --partition ellis run.sh --model hmm --ellis --b 30 --abstract_pen 1

# sbatch --partition ellis run.sh --load --model cc --fine_tune --n 2000 --ellis
# sbatch --partition ellis run.sh --load --model cc --fine_tune --n 5000 --ellis

# sbatch run.sh --muzero --n 500 --test_every 60  --load  --length 4 --muzero_scratch
# sbatch run.sh --muzero --n 500 --test_every 60  --load  --length 4
# sbatch run.sh --muzero --n 500 --test_every 60  --load  --length 4 --freeze_tau
