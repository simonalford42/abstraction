#!/usr/bin/env bash


 # job name
#SBATCH -J sv
 # output file (%j expands to jobID)
#SBATCH -o out/bw11_%A.out
 # total nodes
#SBATCH -N 1
 # total cores
#SBATCH -n 1
#SBATCH --cpus-per-task=8
 # total limit (hh:mm:ss)
#SBATCH -t 72:00:00
#xSBATCH --mem=65G
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#xSBATCH --partition=ellis
python -u main.py "$@"
# python -u main_old.py
echo repeating
python -u main.py "$@"
