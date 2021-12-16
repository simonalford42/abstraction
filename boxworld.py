import einops
import gym
import numpy as np
from gym import envs
import gym_boxworld
import torch
from torch.utils.data import Dataset  # , DataLoader
from utils import assertEqual
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from torch.distributions import Categorical


def make_env():
    env_name = 'BoxRandWorld'
    env_id = env_name + 'NoFrameskip-v4'
    env = gym.make(env_id, level='easy')
    return env


# for reference, never used in code
COLOR_NAMES = {'M': 'Magenta',
               'Y': 'Yellow',
               'C': 'Cyan',
               'W': 'White',
               'G': 'Green',
               'R': 'Red',
               'S': 'Salmon',
               'P': 'Purple',
               'B': 'Black',
               'DG': 'Dark Grey',
               'GR': 'Grey'
               }


PATH = [('M',),
        ('M', 'Y'),
        ('Y', 'G'),
        ('G', 'R'),
        ('R', 'W')]

COLORS = {'M': (255., 0., 255.),
          'Y': (255., 255., 0.),
          'C': (0., 255., 255.,),
          'W': (255., 255., 255.),
          'G': (0., 255., 0.),
          'R': (255., 0., 0.),
          'S': (255. , 127.5, 127.5),
          'P': (127.5,   0. , 255.),
          'B': (0., 0., 0.),
          'DG': (105., 105., 105.),
          'GR': (169., 169., 169.),
          }

RGB_TO_COLOR = {v: k for k, v in COLORS.items()}


def to_color_obs(obs):
    return np.array([[RGB_TO_COLOR[tuple(obs[y, x])] for x in range(len(obs[y]))] for y in range(len(obs))])


def get_domino_position(obs, p):
    for y in range(len(obs)):
        for x in range(len(obs)):
            if domino_matches(p, obs, y, x):
                return y, x
    return None


def domino_matches(p, obs, y, x):
    # matches at the last pixel of p, since that's where the key we want to pick up is.
    try:
        return (obs[y, x-len(p)] in ['B', 'GR', 'DG']
                and obs[y, x+1] in ['B', 'GR', 'DG']
                and np.array_equal(obs[y, x-len(p)+1:x+1], list(p[::-1])))
    except IndexError:
        return False


def shortest_path(color_obs, goal):
    # goal is coordinates
    # start is assumed to be where grey dot is
    start = tuple(np.argwhere(color_obs == 'DG')[0])
    assert color_obs[start] == 'DG'
    nodes, adj_matrix = make_graph(color_obs, goal)
    return dijkstra(nodes, adj_matrix, start, goal)


def make_graph(obs, goal):
    # nodes that can be walked through:
    # - "self" color (dark grey)
    # - background color (grey)
    # - target key color
    nodes = {(y, x) for y in range(len(obs))
                    for x in range(len(obs[y]))
                    if (obs[y, x] in ['DG', 'GR']
                        or (y, x) == goal)}
    adj_matrix = {(y, x): {n2: 1 for n2 in [(y+1, x), (y-1, x), (y, x-1), (y, x+1)]
                                 if n2 in nodes}
                  for (y, x) in nodes}

    return nodes, adj_matrix


class NoPathError(Exception):
    pass

def dijkstra(nodes, adj_matrix, start, goal):
    distances = {n: float('inf') for n in nodes}
    predecessors = {n: None for n in nodes}
    distances[start] = 0
    visited = set()

    def get_min_unvisited():
        unvisited_dists = [(n, d) for (n, d) in distances.items() if n not in visited]
        return min(unvisited_dists, key=lambda t: t[1])

    def get_path_to(node):
        path = [node]
        while path[-1] != start:
            if node is None:
                raise NoPathError
            node = predecessors[node]
            path.append(node)
        return path[::-1]

    while goal not in visited:
        current, current_dist = get_min_unvisited()

        for neighbor, weight_from_current_to_neighbor in adj_matrix[current].items():
            neighbor_dist = distances[neighbor]
            alt_neighbor_dist = current_dist + weight_from_current_to_neighbor
            if alt_neighbor_dist < neighbor_dist:
                assert neighbor not in visited
                distances[neighbor] = alt_neighbor_dist
                predecessors[neighbor] = current

        visited.add(current)

    return get_path_to(goal)


def path_to_moves(path):
    """
    path is a sequence of points.
    converts to a sequence of up/down/left/right actions.
    """
    def tuple_diff(a, b):
        return a[0] - b[0], a[1] - b[1]

    diffs = [tuple_diff(a, b) for a, b in zip(path[1:], path[:-1])]

    def move(diff):
        if diff == (0, 1):
            return 3  # right
        if diff == (0, -1):
            return 2  # left
        if diff == (1, 0):
            return 1  # down
        if diff == (-1, 0):
            return 0  # up
        raise ValueError(f'unknown diff {diff}')

    return [move(d) for d in diffs]


def exec(moves, env):
    for m in moves:
        obs, _, done, _ = env.step(m)
    return obs, done


def eval_model(net, env, n=100):
    def obs_to_net(obs):
        assertEqual(obs.shape, (14, 14, 3))
        obs = torch.tensor(obs)
        obs = einops.rearrange(obs, 'h w c -> 1 c h w')
        return obs

    print(f'Evaluating model on {n} episodes')
    solved = 0
    for i in range(n):
        states, moves = generate_traj(env)
        obs = states[0]
        done = False
        for t in range(10 * len(moves)):
            obs = obs_to_net(obs)
            out = net(obs)[0]
            a = torch.distributions.Categorical(logits=out).sample()
            obs, rew, done, info = env.step(a)
            if done:
                break
        if done:
            solved += 1
    print(f'Solved {solved}/{n} episodes')
    return solved
            

def test_solving():
    env = make_env()
    obs = env.reset()
    color_obs = to_color_obs(obs)
    goals = [get_domino_position(color_obs, p) for p in PATH]
    for goal in goals:
        color_obs = to_color_obs(obs)
        try:
            option = shortest_path(color_obs, goal)
        except NoPathError:
            # unsolvable, so who cares
            return
        # print(option)
        obs, done = exec(path_to_moves(option), env)
    assert done, 'uh oh'


def generate_traj(env=None):
    if env is None:
        env = make_env()
    # [states], [moves]
    obs = env.reset()
    color_obs = to_color_obs(obs)
    goals = [get_domino_position(color_obs, p) for p in PATH]

    states = [obs]
    moves = []

    for goal in goals:
        color_obs = to_color_obs(obs)
        try:
            option = shortest_path(color_obs, goal)
        except NoPathError:
            # try over again
            return generate_traj(env)
        # env.render()
        # print(option)

        for a in path_to_moves(option):
            obs, _, done, _ = env.step(a)
            states.append(obs)
            moves.append(a)
    assert done, 'uh oh'

    return states, moves


def generate_boxworld_data(n, env=None):
    if env is None:
        env = make_env()
    return [generate_traj(env) for i in range(n)]


class BoxWorldDataset(Dataset):
    def __init__(self, data):
        """
        data: list of (states, moves) tuples
        """
        self.data = data
        self.state_shape = data[0][0][0].shape  # should be (14, 14, 3)
        assertEqual(self.state_shape, (14, 14, 3))

        # ignore last state
        self.states = [torch.tensor(s).reshape(3, 14, 14) for states, _ in self.data for s in states[:-1]]
        self.moves = [torch.tensor(m) for _, moves in self.data for m in moves]

    def __len__(self):
        return len(self.states)

    def __getitem__(self, i):
        return self.states[i], self.moves[i]


class BoxWorldData():
    def __init__(self, data):
        """
        data: list of (states, moves) tuples
        """
        self.data = data
        self.state_shape = data[0][0][0].shape  # should be (14, 14, 3)
        assertEqual(self.state_shape, (14, 14, 3))

        # channels need to be second dim
        self.states = [torch.tensor(np.array(states)).reshape(-1, 3, 14, 14)
                       for (states, moves) in data]
        self.moves = [torch.tensor(moves)
                      for (states, moves) in data]
        self.max_steps = max(len(m) for m in self.moves)

    def tensorize_obs(obs):
        # batch size 1
        return torch.tensor(obs).reshape(1, 3, 14, 14)

    def action_to_move(action):
        return 'UDLR'[action]


def norm_rgb(obs):
    return [[obs[i][j]/255. for j in range(len(obs[i]))]
            for i in range(len(obs))]


def render_sequence(states, fps=3):
    states = [norm_rgb(s) for s in states]

    fig = plt.figure()
    im = plt.imshow(states[0])

    def init():
        im.set_data(states[0])
        return [im]

    def animate(i):
        im.set_data(states[i])
        return [im]

    anim = animation.FuncAnimation(fig, animate, frames=len(states),
            interval=1000/fps, repeat=False)
    plt.show()


def sample_trajectories(net, n, env, max_steps, full_abstract=False, render=True):
    """
    To sample with options:
    1. sample option from start state.
    2. choose actions according to option policy until stop.
    3. after stopping, sample new option.
    4. repeat until done.

    if full_abstract=True, then we will execute alpha(t_0, b) to get new t_i to
    sample new option from. Otherwise, will get t_i from tau(s_i).
    """
    print(f"max_steps: {max_steps}")
    for i in range(n):
        obs = env.reset()
        moves_taken = ''
        options = []
        current_option_path = ''
        start_t_of_current_option = None
        option = None
        obss = [obs]

        for j in range(max_steps):
            state_batch = BoxWorldData.tensorize_obs(obs)
            # only use action_logps, stop_logps, and start_logps
            t_i, action_logps, stop_logps, start_logps, causal_penalty = net.abstract_policy_net(state_batch)
            if option is None:
                option = Categorical(logits=start_logps).sample()
                start_t_of_current_option = t_i[0]
            else:
                # possibly stop previous option!
                stop = Categorical(logits=stop_logps[0, option, :]).sample()
                if stop == net.abstract_policy_net.stop_net_stop_ix:
                    if full_abstract:
                        new_t_i = net.abstract_policy_net.alpha_transition(start_t_of_current_option, option)
                        logits = net.abstract_policy_net.new_option_logps(new_t_i)
                    else:
                        logits = start_logps[0]

                    option = Categorical(logits=logits).sample()
                    options.append(current_option_path)
                    current_option_path = ''
                    start_t_of_current_option = new_t_i
                    # make obs flash yellow in the border
                    obs2 = []
                    for y in range(len(obs)):
                        obs2_row = []
                        for x in range(len(obs[y])):
                            if np.array_equal(obs[y, x], [0., 0., 0.,]):
                                obs2_row.append([255., 255., 0.])
                            else:
                                obs2_row.append(obs[y, x])
                        obs2.append(obs2_row)
                    obss[-1] = np.array(obs2)


            current_option_path += str(option.item())
            action = Categorical(logits=action_logps[0, option, :]).sample()
            obs, _, done, _ = env.step(action)
            move = BoxWorldData.action_to_move(action)
            moves_taken += move
            obss.append(obs)
            if done:
                break

        options.append(current_option_path)
        print(f"moves: {''.join(moves_taken)}")
        print(f"options: {'/'.join(options)}")
        print(f"reached end: {done}")
        print('-'*10)
        if render:
            render_sequence(obss, fps=10)


if __name__ == '__main__':
    env = make_env()
    trajs = generate_boxworld_data(n=1000)
    data = BoxWorldData(trajs)

    states, moves = generate_traj(env)
    render_sequence(states)
