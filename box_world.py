from collections import Counter
from typing import Any, Optional
import gym
import argparse
import numpy as np
import pycolab
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import einops
from utils import assertEqual, POS, DEVICE

from pycolab.examples.research.box_world import box_world as bw


NUM_ASCII = len('# *.abcdefghijklmnopqrst')


class BoxWorldEnv(gym.Env):
    """
    OpenAI gym interface for the BoxWorld env implemented by DeepMind as part of their pycolab game engine.
    """

    def __init__(
        self,
        grid_size=12,
        solution_length=(1, 2, 3, 4),
        num_forward=(0, 1, 2, 3, 4),
        num_backward=(0,),
        branch_length=1,
        max_num_steps=120,
        seed=0,
    ):
        self.grid_size = grid_size
        self.solution_length = solution_length
        self.num_forward = num_forward
        self.num_backward = num_backward
        self.branch_length = branch_length
        self.max_num_steps = max_num_steps
        self.random_state = np.random.RandomState(seed)
        self.obs = self.reset()

    def reset(self):
        self.game = bw.make_game(
            grid_size=self.grid_size,
            solution_length=self.solution_length,
            num_forward=self.num_forward,
            num_backward=self.num_backward,
            branch_length=self.branch_length,
            max_num_steps=self.max_num_steps,
            random_state=self.random_state,
        )
        # from line 267 of human_ui.py
        obs, reward, _ = self.game.its_showtime()
        assert reward is None
        obs = self.process_obs(obs)
        return obs

    def process_obs(self, obs) -> np.ndarray:
        return np.array([list(row.tobytes().decode('ascii')) for row in obs.board])

    def step(
        self, action: int
    ) -> tuple[Any, float, bool, dict]:
        """
        Applies action to the boxworld environment.

        Returns:
            observation (object): agent's observation of the current environment, a numpy array of string characters.
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        obs, reward, _ = self.game.play(action)
        obs = self.process_obs(obs)
        self.obs = obs
        done = self.game.game_over
        return obs, reward, done, dict()


def hex_to_rgb(hex: str) -> tuple[int]:
    # https://stackoverflow.com/a/29643643/4383594
    hex = hex.lstrip('#')
    return tuple(int(hex[i:i+2], 16) for i in (0, 2, 4))


def color_name_to_rgb(name: str) -> tuple[int]:
    # https://matplotlib.org/stable/gallery/color/named_colors.html
    hex = matplotlib.colors.CSS4_COLORS[name]
    return hex_to_rgb(hex)


def ascii_to_color(ascii: str):
    ascii = ascii.lower()
    if (ascii not in {bw.BORDER, bw.BACKGROUND, bw.GEM, bw.PLAYER}
            and ascii not in '# *.abcdefghijklmnopqrst'):
        raise ValueError('invalid ascii provided: ' + ascii)
    if ascii == bw.BORDER:
        return color_name_to_rgb('black')
    elif ascii == bw.BACKGROUND:
        return color_name_to_rgb('lightgray')
    elif ascii == bw.GEM:
        return color_name_to_rgb('white')
    elif ascii == bw.PLAYER:
        return color_name_to_rgb('dimgrey')
    else:
        s = 'abcdefghijklmnopqrst'
        i = s.index(ascii)
        colors = ['brown', 'maroon', 'red', 'orangered', 'orange',
                  'yellow', 'gold', 'olive', 'greenyellow', 'limegreen', 'green', 'darkgreen', 'cyan',
                  'dodgerblue', 'blue', 'darkblue', 'indigo', 'purple', 'magenta', 'violet']
        assertEqual(len(colors), len(s))
        return color_name_to_rgb(colors[i])


def ascii_to_int(ascii: str):
    # lower and uppercase render the same, just like the colors
    return '# *.abcdefghijklmnopqrst'.index(ascii.lower())



def to_color_obs(obs):
    return np.array([[ascii_to_color(a) for a in row] for row in obs])


def render_obs(obs, color=True, pause=0.0001):
    """
    With color=True, makes a plot.
    With color=False, prints ascii to command line.
    if color=True, then pause tells how long to wait between frames.
    """
    if color:
        color_array = to_color_obs(obs)

        fig = plt.figure(0)
        plt.clf()
        plt.imshow(color_array / 255)
        fig.canvas.draw()
        plt.pause(pause)
    else:
        for row in obs:
            print(''.join(row))


def run_deepmind_ui(**args):
    if args['seed'] is not None:
        raise RuntimeError("Deepmind ui doesn't work with nondefault seed")

    del args['seed']

    game = bw.make_game(**args)

    ui = pycolab.human_ui.CursesUi(
        keys_to_actions={
            'w': bw.ACTION_NORTH,
            's': bw.ACTION_SOUTH,
            'a': bw.ACTION_WEST,
            'd': bw.ACTION_EAST,
            -1: bw.ACTION_DELAY,
        },
        delay=50,
        colour_fg=bw.OBJECT_COLORS)
    ui.play(game)


def play_game(env):
    obs = env.reset()
    done = False
    print('Enter wasd to move, q to quit')
    while not done:
        render_obs(obs, color=False)
        key = input()
        if key.lower() == 'q':
            return False

        if len(key) != 1 or key not in 'wasd':
            continue

        if key == 'w':
            action = bw.ACTION_NORTH
        elif key == 'a':
            action = bw.ACTION_WEST
        elif key == 's':
            action = bw.ACTION_SOUTH
        else:
            action = bw.ACTION_EAST

        obs, rew, done, info = env.step(action)


def get_dominoes(obs) -> dict[str, POS]:
    """
     {'aT': (3, 3), etc.}
        The location value is where the agent wants to go.
        So it's the right side of the domino.
    """

    dominoes = {}
    for y in range(len(obs)):
        for x in range(len(obs[y])):
            # top left is held key, not a domino!
            if (y, x) == (0, 0):
                continue

            s = obs[y, x]
            if s == bw.BACKGROUND or s == bw.BORDER:
                continue
            if s == bw.PLAYER:
                dominoes[s] = (y, x)

            left_of = obs[y, x - 1]
            right_of = obs[y, x + 1]

            if s == bw.GEM:
                # goal is always on left.
                if right_of not in {bw.PLAYER, bw.BACKGROUND}:
                    dominoes[s + right_of] = (y, x + 1)
                else:
                    # goal is already unlocked
                    # agent should be right next to it, but maybe he walked away stupidly
                    dominoes[s] = (y, x)
            elif (left_of in {bw.PLAYER, bw.BACKGROUND, bw.BORDER}
                    and right_of in {bw.PLAYER, bw.BACKGROUND, bw.BORDER}):
                # lone key!
                dominoes[s] = (y, x)
            elif s.islower() and right_of.isupper():
                dominoes[s + right_of] = (y, x + 1)

    return dominoes


def get_goal_domino(dominoes: set[str]) -> str:
    for d in dominoes:
        if d[0] == '*':
            return d

    raise ValueError('No goal domino in set provided')


def get_held_key(obs) -> Optional[str]:
    return obs[0, 0] if obs[0, 0] != bw.BORDER else None


def get_tree(domino_pos_map: dict[str, POS],
             held_key: Optional[str]) -> tuple[set[str], dict[str, dict[str, int]]]:
    """
    returns nodes, adj_matrix where nodes are one or two character ascii strings
    for the dominoes.
    """
    nodes = set(domino_pos_map.keys())

    currently_accessible_nodes = {n: 1 for n in nodes if n != bw.PLAYER
                                  and (len(n) == 1 or held_key == n[1].lower())}

    adj_matrix = {bw.PLAYER: currently_accessible_nodes}
    for n in nodes:
        if n == bw.PLAYER:
            continue
        key = n[0]  # works for domino or lone key, even for the GEM
        adj_matrix[n] = {n2: 1 for n2 in nodes
                            if len(n2) == 2 and n2[1].lower() == key}

    return nodes, adj_matrix


def is_valid_solution_path(path: list[str], dominoes: set[str], held_key: str) -> bool:
    if path[0] != bw.PLAYER:
        return False
    if any([p not in dominoes for p in path]):
        return False
    if len(path[1]) == 2 and path[1][1].lower() != held_key:
        return False
    for i in range(2, len(path)-1):
        if path[i][0] != path[i+1][1].lower():
            return False
    if path[-1][0] != bw.GEM:
        return False
    return True


class NoPathError(Exception):
    pass


def dijkstra(nodes: set, adj_matrix: dict[Any, dict[Any, int]], start, goal) -> list:
    """
    nodes: set of items
    adj_matrix: map from item to map of adjacent_node: weight_to_node pairs (may be directed graph)
    start: a node
    goal: a node

    returns a list of nodes to the solution.
    """
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


def make_move_graph(obs, goal_pos) -> tuple[set[POS], dict[POS, dict[POS, int]]]:
    # nodes that can be walked through:
    # - "self"
    # - background
    # - target key
    nodes = {(y, x) for y in range(len(obs))
                    for x in range(len(obs[y]))
                    if (obs[y, x] in [bw.BACKGROUND, bw.PLAYER]
                        or (y, x) == goal_pos)}
    adj_matrix = {(y, x): {n2: 1 for n2 in [(y+1, x), (y-1, x), (y, x-1), (y, x+1)]
                           if n2 in nodes}
                  for (y, x) in nodes}

    return nodes, adj_matrix


def shortest_path(obs, goal_pos: POS) -> list[POS]:
    start_pos = tuple(np.argwhere(obs == bw.PLAYER)[0])
    assert obs[start_pos] == bw.PLAYER

    nodes, adj_matrix = make_move_graph(obs, goal_pos)
    return dijkstra(nodes, adj_matrix, start_pos, goal_pos)


def path_to_moves(path: list[POS]) -> list[int]:
    """
    path is a sequence of (y, x) positions.
    converts to a sequence of up/down/left/right actions.
    """
    def tuple_diff(a, b):
        return a[0] - b[0], a[1] - b[1]

    diffs = [tuple_diff(a, b) for a, b in zip(path[1:], path[:-1])]

    dir_to_action_map = {d: a for (a, d) in bw.ACTION_MAP.items()}
    return [dir_to_action_map[d] for d in diffs]


def generate_traj(env: BoxWorldEnv) -> tuple[list, list]:
    obs = env.reset()

    domino_pos_map = get_dominoes(obs)
    held_key = get_held_key(obs)
    goal_domino = get_goal_domino(domino_pos_map.keys())
    nodes, adj_matrix = get_tree(domino_pos_map, held_key)
    path = dijkstra(nodes, adj_matrix, start=bw.PLAYER, goal=goal_domino)
    assert is_valid_solution_path(path, domino_pos_map.keys(), held_key)

    states = [obs]
    moves = []

    for i, domino in enumerate(path):
        subgoal_pos: POS = domino_pos_map[domino]
        option: list[POS] = shortest_path(obs, subgoal_pos)

        for a in path_to_moves(option):
            obs, _, done, _ = env.step(a)
            # render_obs(obs, pause=0.01)
            states.append(obs)
            moves.append(a)
        if len(domino) > 1:
            # move left to pick up new key, or final gem
            obs, _, done, _ = env.step(bw.ACTION_WEST)
            # render_obs(obs, pause=0.03)
            states.append(obs)
            moves.append(a)

    assert done, 'uh oh, our path solver didnt actually solve'
    return states, moves


def eval_model(net, env, n=100, T=100, render=False):

    print(f'Evaluating model on {n} episodes')
    solved = 0
    num_found_keys = []
    found_keys = set()
    for i in range(n):
        obs = env.reset()
        done = False
        for t in range(T):
            if render:
                render_obs(obs)
            if obs[0, 0].isalpha():
                found_keys.add(obs[0, 0])
            obs = obs_to_tensor(obs)
            obs = einops.rearrange(obs, 'c h w -> 1 c h w')
            obs = obs.to(DEVICE)
            out = net(obs)[0]
            a = torch.distributions.Categorical(logits=out).sample().item()
            obs, rew, done, info = env.step(a)
            if done:
                break
        if done:
            solved += 1
        num_found_keys.append(len(found_keys))
    print(f'Solved {solved}/{n} episodes; {Counter(num_found_keys)} are path dists')
    return solved


def generate_boxworld_data(n, env) -> list[tuple[list, list]]:
    return [generate_traj(env) for i in range(n)]


def obs_to_tensor(obs) -> torch.Tensor:
    obs = torch.tensor([[ascii_to_int(a) for a in row]
                        for row in obs])
    obs = F.one_hot(obs, num_classes=NUM_ASCII).to(torch.float)
    obs = einops.rearrange(obs, 'h w c -> c h w')
    return obs


class BoxWorldDataset(Dataset):
    def __init__(self, data: list[tuple[list, list]]):
        """
        data: list of (states, moves) tuples
        """
        self.data = data
        self.state_shape = data[0][0][0].shape

        # ignore last state
        self.states = [obs_to_tensor(s)
                       for states, _ in self.data for s in states[:-1]]
        self.moves = [torch.tensor(m) for _, moves in self.data for m in moves]

    def __len__(self):
        return len(self.states)

    def __getitem__(self, i):
        return self.states[i], self.moves[i]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Play Box-World.')
    parser.add_argument(
        '--grid_size', type=int, default=12, help='height and width of the grid.')
    parser.add_argument(
        '--solution_length',
        nargs='+',
        type=int,
        default=(1, 2, 3, 4),
        help='number of boxes in the path to the goal.')
    parser.add_argument(
        '--num_forward',
        nargs='+',
        type=int,
        default=(0, 1, 2, 3, 4),
        help='possible values for num of forward distractors.')
    parser.add_argument(
        '--num_backward',
        nargs='+',
        type=int,
        default=(0,),
        help='possible values for num of backward distractors.')
    parser.add_argument(
        '--branch_length',
        type=int,
        default=1,
        help='length of forward distractor branches.')
    parser.add_argument(
        '--max_num_steps',
        type=int,
        default=120,
        help='number of steps before the episode is halted.')
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='random seed')
    FLAGS = parser.parse_args()
    # while True:
    #     run_deepmind_ui(**vars(FLAGS))

    env = BoxWorldEnv(**vars(FLAGS))
    while True:
        generate_traj(env)
