#!/usr/bin/env bash

sbatch run.sh --n 10000 --load --lr 0.0008 --fine_tune
sbatch run.sh --n 10000 --load --lr 0.0008 --fine_tune --batch_norm
sbatch run.sh --n 10000 --load --lr 0.008 --fine_tune
sbatch run.sh --n 10000 --load --lr 0.00008 --fine_tune
sbatch run.sh --n 10000 --load --fine_tune --replace_trans_net

# sbatch run.sh --n 20000 --model hmm-homo --abstract_pen 0
# sbatch run.sh --n 20000 --model hmm-homo --abstract_pen 0
# # sbatch run.sh --n 20000 --model cc --abstract_pen 0
# sbatch run.sh --n 20000 --model hmm-homo --abstract_pen 1 --b 10 --variable_abstract_pen
# sbatch run.sh --n 20000 --model hmm-homo --abstract_pen 1 --b 30
# sbatch run.sh --n 20000 --model hmm-homo --abstract_pen 2 --b 10 --variable_abstract_pen
