#!/usr/bin/env bash

sbatch run.sh --neurosym --symbolic_sv --micro_net2 --traj_updates 5E8
sbatch run.sh --neurosym --symbolic_sv --dim 128 --traj_updates 5E8
