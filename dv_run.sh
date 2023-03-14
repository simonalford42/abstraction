#!/usr/bin/env bash


 # job name
#SBATCH -J dv2
 # output file (%j expands to jobID)
#SBATCH -o dv_out/%A.out
 # total nodes
#SBATCH -N 1
 # total cores
#SBATCH -n 1
#SBATCH --requeue
#SBATCH --cpus-per-task=8
 # total limit (hh:mm:ss)
#SBATCH -t 12-00
#SBATCH --mem=100G
#SBATCH --gres=gpu:a6000:1
#SBATCH --partition=gpu

python dv2.py
