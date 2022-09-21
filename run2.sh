#!/usr/bin/env bash
# sbatch --partition=ellis run.sh --fake_cc_neurosym --model hmm --abstract_pen 1 --ellis
sbatch --partition=ellis run.sh --fake_cc_neurosym --model hmm --abstract_pen 1 --ellis --relational_macro_net
