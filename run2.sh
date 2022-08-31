#!/usr/bin/env bash

sbatch --partition=ellis run.sh --neurosym --traj_updates 1E8
sbatch --partition=ellis run.sh --neurosym --traj_updates 1E8

# sbatch --partition=ellis run.sh --model cc --cc_neurosym --abstract_pen 1 --traj_updates 4E7
# sbatch --partition=ellis run.sh --model cc --cc_neurosym --abstract_pen 1 --traj_updates 4E7

sbatch run.sh --neurosym --traj_updates 1E8
sbatch run.sh --neurosym --traj_updates 1E8

# sbatch run.sh --model cc --cc_neurosym --abstract_pen 1 --traj_updates 4E7
# sbatch run.sh --model cc --cc_neurosym --abstract_pen 1 --traj_updates 4E7


# sbatch run.sh --model hmm-homo --abstract_pen 1.5 --traj_updates 2E7
# sbatch run.sh --model hmm-homo --abstract_pen 2.0 --traj_updates 2E7
# sbatch run.sh --model hmm-homo --abstract_pen 5.0 --traj_updates 2E7

# sbatch run.sh --model hmm --abstract_pen 0.2 --traj_updates 4E7
# sbatch run.sh --model hmm --abstract_pen 0.2 --traj_updates 4E7
# sbatch run.sh --model hmm --abstract_pen 0.3 --traj_updates 4E7
# sbatch run.sh --model hmm --abstract_pen 0.3 --traj_updates 4E7
# sbatch run.sh --model hmm --abstract_pen 0.5 --traj_updates 4E7
# sbatch run.sh --model hmm --abstract_pen 0.5 --traj_updates 4E7
# sbatch run.sh --model hmm --abstract_pen 0.7 --traj_updates 4E7
# sbatch run.sh --model hmm --abstract_pen 0.7 --traj_updates 4E7

# sbatch run.sh --model hmm --abstract_pen 1 --traj_updates 4E7 --length '(3, 4, )'
# sbatch run.sh --model hmm --abstract_pen 1 --traj_updates 4E7 --length '(2, 3, 4, )'

# sbatch run.sh --model hmm --abstract_pen 0 --traj_updates 4E7
# sbatch run.sh --model hmm --abstract_pen 0 --traj_updates 4E7
# sbatch run.sh --model hmm --abstract_pen 0 --traj_updates 4E7
