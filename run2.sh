#!/usr/bin/env bash

# sbatch run.sh --traj_updates 1E9 --sv_micro --sv_micro_data_type ground_truth
# sbatch run.sh --traj_updates 1E9 --sv_micro --sv_micro_data_type from_model
# sbatch run.sh --traj_updates 1E9 --sv_micro --sv_micro_data_type full_traj
# sbatch run.sh --traj_updates 1E9 --sv_micro --sv_micro_data_type ground_truth --relational_micro --lr 0.0001
# sbatch run.sh --traj_updates 1E9 --sv_micro --sv_micro_data_type from_model --relational_micro --lr 0.0001
# sbatch run.sh --traj_updates 1E9 --sv_micro --sv_micro_data_type full_traj --relational_micro --lr 0.0001
# sbatch run.sh --traj_updates 1E9 --sv_micro --sv_micro_data_type ground_truth --lr 0.0001
# sbatch run.sh --traj_updates 1E9 --sv_micro --sv_micro_data_type from_model --lr 0.0001
# sbatch run.sh --traj_updates 1E9 --sv_micro --sv_micro_data_type full_traj --lr 0.0001

# sbatch run.sh --b 5 --model hmm --batch_size 16 --seed 1
# sbatch run.sh --b 5 --model hmm --batch_size 16 --seed 2
# sbatch run.sh --b 5 --model hmm --batch_size 16 --seed 3
# sbatch run.sh --b 5 --model hmm --batch_size 16 --seed 4
# sbatch run.sh --b 5 --model hmm --batch_size 16 --seed 5
# sbatch run.sh --b 10 --model hmm --batch_size 16 --seed 1
# sbatch run.sh --b 10 --model hmm --batch_size 16 --seed 2
# sbatch run.sh --b 10 --model hmm --batch_size 16 --seed 3
# sbatch run.sh --b 10 --model hmm --batch_size 16 --seed 4
# sbatch run.sh --b 10 --model hmm --batch_size 16 --seed 5
# sbatch run.sh --b 20 --model hmm --batch_size 16 --seed 1
# sbatch run.sh --b 20 --model hmm --batch_size 16 --seed 2
# sbatch run.sh --b 20 --model hmm --batch_size 16 --seed 3
# sbatch run.sh --b 20 --model hmm --batch_size 16 --seed 4
# sbatch run.sh --b 20 --model hmm --batch_size 16 --seed 5

# sbatch run.sh --model hmm --lr 1E-4 --traj_updates 1E9
# sbatch run.sh --model hmm --lr 2E-4 --traj_updates 1E9
# sbatch run.sh --model hmm --lr 4E-4 --traj_updates 1E9

# sbatch run.sh --model hmm --bigger_micro --seed 1 --b 20 --batch_size 16
# sbatch run.sh --model hmm --bigger_micro --seed 2 --b 20 --batch_size 16
# sbatch run.sh --model hmm --bigger_micro --seed 3 --b 20 --batch_size 16
# sbatch run.sh --model hmm --bigger_micro --seed 4 --b 20 --batch_size 16


# sbatch run.sh --model hmm --random_goal
# sbatch run.sh --model hmm --random_goal --seed 2

sbatch run.sh --model hmm --options_fine_tune --model_load_path models/8110c8302c1946a5a6838cd2430b705f.pt --note "random goal pretrained" --abstract_pen 0
sbatch run.sh --model hmm --options_fine_tune --model_load_path models/94a19359dad64917988276fcc6108f82-epoch-192.pt --note "no random goal pretraining" --abstract_pen 0
sbatch run.sh --model hmm --options_fine_tune --model_load_path models/8110c8302c1946a5a6838cd2430b705f.pt --note "random goal pretrained"
sbatch run.sh --model hmm --options_fine_tune --model_load_path models/94a19359dad64917988276fcc6108f82-epoch-192.pt --note "no random goal pretraining"
