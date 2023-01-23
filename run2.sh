#!/usr/bin/env bash

sbatch run.sh --model sv --random_goal --seed 3
sbatch run.sh --model sv --random_goal --seed 2
sbatch run.sh --model cc --random_goal --seed 2
