# Installation
This code required Python3. We recommend installing dependencies inside of a
`virtualenv`.

```
python3 -m virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
tar -xvf actual_walls.tar.gz
```

# Training
The following commands reproduce the experiments reported in the paper.

For the Simple Colors and Conditional Colors datasets:

```
PYOPENGL_PLATFORM=egl python train_mdl.py \
    --coding_len_coeff=0.1 \
    --use_abs_pos_kl=1.0 \
    --batch-size=512 \
    --name=color \
    --seed=1 \
    --dataset-path=./data/simple_colors.npy \
    --max-iters=40000

PYOPENGL_PLATFORM=egl python train_mdl.py \
    --coding_len_coeff=0.1 \
    --use_abs_pos_kl=1.0 \
    --batch-size=512 \
    --name=conditional_color \
    --seed=1 \
    --dataset-path=./data/conditional_colors.npy \
    --max-iters=40000
```

For the multi-task grid world from ComPILE, we provide a pretrained checkpoint.
First, set the `PYTHONPATH` variable to point at the current directory:

```
export PYTHONPATH=/path/to/current/directory
```

Then, run:

```
PYOPENGL_PLATFORM=egl python3 dqn/main.py \
    love+1 \
    -b agent.policy.epsilon_schedule.total_steps=500000 \
    -b checkpoint=\"pretrained.ckpt\" \
    -b threshold=0 \
    -b sparse_reward=True \
    -b visit_length=3 \
    -b bc=False \
    -b oracle=False \
    --seed 1
```

Or, the following command can be used to run the LOVE skill learning phase to
generate a new checkpoint:

```
PYOPENGL_PLATFORM=egl python train_rl.py \
    --coding_len_coeff=0.001 \
    --kl_coeff=0 \
    --rec_coeff=1.0 \
    --use_abs_pos_kl=1.0 \
    --batch-size=64 \
    --name=love+1 \
    --seed=1  \
    --dataset-path=compile  \
    --max-iters=20000 \
    --use_min_length_boundary_mask
```
