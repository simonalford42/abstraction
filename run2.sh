#!/usr/bin/env bash

# sbatch run.sh --model hmm --bigger_micro --seed 4 --b 20 --batch_size 16
# sbatch run.sh --model hmm --random_goal

# sbatch run.sh --model hmm --options_fine_tune --model_load_path models/8110c8302c1946a5a6838cd2430b705f.pt --note "random goal pretrained" --abstract_pen 0
# sbatch run.sh --model hmm --options_fine_tune --model_load_path models/94a19359dad64917988276fcc6108f82-epoch-192.pt --note "no random goal pretraining" --abstract_pen 0
# sbatch run.sh --model hmm --options_fine_tune --model_load_path models/8110c8302c1946a5a6838cd2430b705f.pt --note "random goal pretrained"
# sbatch run.sh --model hmm --options_fine_tune --model_load_path models/94a19359dad64917988276fcc6108f82-epoch-192.pt --note "no random goal pretraining"

# sbatch --partition ellis run.sh --model hmm --abstract_pen 0 --seed 1 --ellis
# sbatch run.sh --model hmm --abstract_pen 0 --seed 2
sbatch run.sh --model hmm --abstract_pen 0 --seed 3

sbatch run.sh --model hmm --abstract_pen 0.3 --seed 1
sbatch run.sh --model hmm --abstract_pen 0.3 --seed 2
# sbatch run.sh --model hmm --abstract_pen 0.3 --seed 3

sbatch run.sh --model hmm --abstract_pen 0.7 --seed 1
sbatch run.sh --model hmm --abstract_pen 0.7 --seed 2
# sbatch run.sh --model hmm --abstract_pen 0.4 --seed 3

