import gym
from lib.dreamerv2.dreamerv2 import api as dv2
import box_world as bw

config = dv2.defaults.update({
    'logdir': '~/logdir/test',
    # 'log_every': 1e3,
    # 'train_every': 10,
    # 'prefill': 1e5,
    # 'actor_ent': 3e-3,
    # 'loss_scales.kl': 1.0,
    # 'discount': 0.99,
    'replay.minlen': 5,
}).parse_flags()

env = bw.GymWrapper(bw.BoxWorldEnv(solution_length=(1,), num_forward=(0,), num_backward=(0,)))
dv2.train(env, config)
