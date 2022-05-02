#!/usr/bin/env bash

# sbatch run.sh --model cc --abstract_pen 1 --tau_noise 0.03 --freeze 0
sbatch run.sh --n 100 --fine_tune --depth 3 --weighting 'uniform'
sbatch run.sh --n 100 --fine_tune --depth 3 --weighting 'logp'
