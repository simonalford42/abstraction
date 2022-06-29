#!/usr/bin/env bash

sbatch --partition ellis run.sh --load --model cc --fine_tune --n 1000 --ellis
sbatch --partition ellis run.sh --load --model cc --fine_tune --n 2000 --ellis
sbatch --partition ellis run.sh --load --model cc --fine_tune --n 5000 --ellis
