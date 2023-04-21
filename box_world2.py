import abstraction.pycolab_env as pycolab_env
from pycolab.examples.research.box_world import box_world as bw
from gym import spaces


class BoxWorldEnv(pycolab_env.PyColabEnv):
    def __init__(self,
                 max_num_steps=160,
                 default_reward=-1.,
                 **kwargs):
        self.bw_args = kwargs
        super(BoxWorldEnv, self).__init__(
            max_iterations=max_num_steps,
            default_reward=default_reward,
            action_space=spaces.Discrete(4),
            resize_scale=8)

    def make_game(self):
        return bw.make_game(**self.bw_args)
