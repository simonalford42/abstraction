#!/usr/bin/env bash

# sbatch --partition=ellis run.sh --neurosym --traj_updates 1E9
# sbatch --partition=ellis run.sh --neurosym --traj_updates 1E8

# sbatch run.sh --model cc --cc_neurosym --abstract_pen 1 --cc_weight 0
# sbatch run.sh --model cc --cc_neurosym --abstract_pen 1 --cc_weight 0.1
# sbatch run.sh --model cc --cc_neurosym --abstract_pen 1 --cc_weight 0.5
# sbatch run.sh --neurosym --state_loss_weight 0.0
# sbatch run.sh --neurosym --state_loss_weight 0.1
# sbatch run.sh --neurosym --state_loss_weight 0.5
# sbatch run.sh --neurosym --state_loss_weight 2

sbatch --partition=ellis run.sh --ellis --model cc --cc_neurosym --abstract_pen 1 --cc_weight 0
sbatch --partition=ellis run.sh --ellis --model cc --cc_neurosym --abstract_pen 1 --cc_weight 0.1
# sbatch --partition=ellis run.sh --ellis --model cc --cc_neurosym --abstract_pen 1 --cc_weight 0.5
sbatch --partition=ellis run.sh --ellis --neurosym --state_loss_weight 0.0
sbatch --partition=ellis run.sh --ellis --neurosym --state_loss_weight 0.1
# sbatch --partition=ellis run.sh --ellis --neurosym --state_loss_weight 0.5
# sbatch --partition=ellis run.sh --ellis --neurosym --state_loss_weight 2


# sbatch run.sh --neurosym --traj_updates 4E8
# sbatch run.sh --neurosym --traj_updates 4E8

# sbatch run.sh --neurosym --traj_updates 4E8 --supervised_symbolic
