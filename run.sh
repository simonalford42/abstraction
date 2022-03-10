#!/usr/bin/env bash


 # job name
#SBATCH -J sv
 # output file (%j expands to jobID)
#SBATCH -o out/bw7_%A.out
 # total nodes
#SBATCH -N 1
 # total cores
#SBATCH -n 1
 # total limit (hh:mm:ss)
#SBATCH -t 48:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=ellis
#xSBATCH --partition=gpu
python -u main.py "$@"
