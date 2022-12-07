#!/usr/bin/env bash

sbatch run.sh --fine_tune --n 20000 --lr 0.0003 --traj_updates 5E8
sbatch run.sh --fine_tune --n 20000 --lr 0.0001 --traj_updates 5E8
