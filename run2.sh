#!/usr/bin/env bash

sbatch run.sh --model sv
sbatch run.sh --model sv --random_goal
sbatch run.sh --model cc --random_goal
