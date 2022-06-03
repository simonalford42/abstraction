#!/usr/bin/env bash

sbatch runs.h --model hmm --abstract_pen 1
sbatch runs.h --model hmm --abstract_pen 1
sbatch run.sh --model hmm --abstract_pen 1 --gumbel --g_start_temp 1 --g_stop_temp 1
sbatch run.sh --model hmm --abstract_pen 1 --gumbel --g_start_temp 1 --g_stop_temp 1
sbatch run.sh --model hmm --abstract_pen 1 --gumbel --g_start_temp 0 --g_stop_temp 0
sbatch run.sh --model hmm --abstract_pen 1 --gumbel --g_start_temp 0 --g_stop_temp 0
sbatch run.sh --model hmm --abstract_pen 1 --gumbel --g_start_temp 1 --g_stop_temp 0
sbatch run.sh --model hmm --abstract_pen 1 --gumbel --g_start_temp 1 --g_stop_temp 0
sbatch run.sh --model hmm --abstract_pen 1 --gumbel --g_start_temp 1 --g_stop_temp 0.5
sbatch run.sh --model hmm --abstract_pen 1 --gumbel --g_start_temp 1 --g_stop_temp 0.5

