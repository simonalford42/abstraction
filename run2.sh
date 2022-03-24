#!/usr/bin/env bash
# sbatch run.sh --hmm --abstract_pen 0.0
# sbatch run.sh --hmm --abstract_pen 1.0
# sbatch run.sh --cc 1.0 --abstract_pen 0.0
# sbatch run.sh --cc 1.0 --abstract_pen 1.0
sbatch run.sh --cc 1.0 --abstract_pen 2.0
sbatch run.sh --hmm --abstract_pen 2.0
sbatch run.sh --hmm --abstract_pen 0.0 --homo
sbatch run.sh --hmm --abstract_pen 1.0 --homo
