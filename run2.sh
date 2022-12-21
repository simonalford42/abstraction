#!/usr/bin/env bash

# sbatch --partition ellis run.sh --model cc --rnn_macro --ellis
# sbatch --partition ellis run.sh --model cc --ellis
# sbatch run.sh --model cc --rnn_macro
# sbatch run.sh --model cc

# sbatch run.sh --model sv
sbatch run.sh --model hmm --solution_length 1
sbatch run.sh --model hmm --solution_length 2
sbatch run.sh --model hmm --solution_length 3
sbatch run.sh --model hmm --solution_length 4
sbatch run.sh --model hmm --solution_length 5
sbatch run.sh --model hmm --solution_length (2, 3)
sbatch run.sh --model hmm --solution_length (2, 4)
sbatch run.sh --model hmm --solution_length (1, 3, 5)
sbatch run.sh --model hmm --b 50
sbatch run.sh --model hmm --b 50
sbatch run.sh --model hmm --b 20
sbatch run.sh --model hmm --b 20
