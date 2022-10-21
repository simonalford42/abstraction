#!/usr/bin/env bash

# sbatch run.sh --sv_micro --sv_micro_data_type ground_truth --traj_updates 1E9 --relational_micro --lr 1E-4
# sbatch run.sh --sv_micro --sv_micro_data_type from_model --traj_updates 1E9 --relational_micro --lr 1E-4
# sbatch run.sh --sv_micro --sv_micro_data_type full_traj --traj_updates 1E9 --relational_micro --lr 1E-4

# sbatch run.sh --cc_neurosym --model cc --abstract_pen 1 --relational_micro
# sbatch run.sh --cc_neurosym --model cc --relational_micro

# sbatch --partition ellis run.sh --model hmm --ellis
# sbatch --partition ellis run.sh --model hmm --ellis
# sbatch run.sh --model hmm
# sbatch run.sh --model hmm

sbatch --partition ellis run.sh --neurosym
# sbatch --partition ellis run.sh --neurosym

# sbatch --partition ellis run.sh --fake_cc_neurosym --model hmm --relational_micro --ellis
# sbatch --partition ellis run.sh --fake_cc_neurosym --model hmm --relational_micro --abstract_pen 1 --ellis
# sbatch --partition ellis run.sh --fake_cc_neurosym --model hmm --abstract_pen 1 --ellis
# sbatch --partition ellis run.sh --fake_cc_neurosym --model hmm --ellis

# sbatch run.sh --sv_micro --sv_
