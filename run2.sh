#!/usr/bin/env bash
# sbatch run.sh --model cc --abstract_pen 0.0
# sbatch run.sh --model cc --abstract_pen 1.0

# sbatch run.sh --model cc --abstract_pen 2.0
# sbatch run.sh --model hmm-homo --abstract_pen 0.0
# sbatch run.sh --model hmm-homo --abstract_pen 1.0
# sbatch run.sh --model hmm-homo --abstract_pen 2.0
# sbatch run.sh --model hmm --abstract_pen 0.0
# sbatch run.sh --model hmm --abstract_pen 1.0
# sbatch run.sh --model hmm --abstract_pen 2.0

# sbatch run.sh --model cc --abstract_pen 0.0 --seed 2
# sbatch run.sh --model cc --abstract_pen 1.0 --seed 2
# sbatch run.sh --model cc --abstract_pen 2.0 --seed 2
# sbatch run.sh --model hmm-homo --abstract_pen 0.0 --seed 2
# sbatch run.sh --model hmm-homo --abstract_pen 1.0 --seed 2
# sbatch run.sh --model hmm-homo --abstract_pen 2.0 --seed 2
# sbatch run.sh --model hmm --abstract_pen 0.0 --seed 2
# sbatch run.sh --model hmm --abstract_pen 1.0 --seed 2
# sbatch run.sh --model hmm --abstract_pen 2.0 --seed 2

# sbatch run.sh --model hmm-homo --n 10000
# sbatch run.sh --model hmm-homo --n 20000
# sbatch run.sh --model hmm-homo --n 10000 --seed 2
# sbatch run.sh --model hmm-homo --n 20000 --seed 2

----

# sbatch run.sh --model cc --abstract_pen 1.0
# sbatch run.sh --model cc --abstract_pen 1.0
# sbatch run.sh --model cc --abstract_pen 1.0

# sbatch run.sh --model cc --abstract_pen 2.0
# sbatch run.sh --model cc --abstract_pen 2.0
# sbatch run.sh --model cc --abstract_pen 2.0

sbatch run.sh --model cc --abstract_pen 1.0 --noise 0.0001
sbatch run.sh --model cc --abstract_pen 1.0 --noise 0.005
sbatch run.sh --model cc --abstract_pen 1.0 --noise 0.001
sbatch run.sh --model cc --abstract_pen 1.0 --noise 0.005
sbatch run.sh --model cc --abstract_pen 1.0 --noise 0.01
sbatch run.sh --model cc --abstract_pen 1.0 --noise 0.05
sbatch run.sh --model cc --abstract_pen 1.0 --noise 0.1
