#!/usr/bin/env bash

sbatch run.sh --model hmm --abstract_pen 1 --gumbel --g_start_temp 10 --g_stop_temp 10
sbatch run.sh --model hmm --abstract_pen 1 --gumbel --g_start_temp 10 --g_stop_temp 10
sbatch run.sh --model hmm --abstract_pen 1 --gumbel --g_start_temp 5 --g_stop_temp 5
sbatch run.sh --model hmm --abstract_pen 1 --gumbel --g_start_temp 5 --g_stop_temp 5
sbatch run.sh --model hmm --abstract_pen 1 --gumbel --g_start_temp 10 --g_stop_temp 5
sbatch run.sh --model hmm --abstract_pen 1 --gumbel --g_start_temp 10 --g_stop_temp 5

