import queue
from typing import Any, Optional, List, Tuple, Callable
import gym
import argparse
import numpy as np
import pycolab
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from einops import rearrange
from utils import assert_equal, POS, DEVICE
from profiler import profile
from torch.distributions import Categorical
from abstract import STOP_NET_STOP_IX
from pycolab.examples.research.box_world import box_world as bw


NUM_ASCII = len('# *.abcdefghijklmnopqrst')


class BoxWorldEnv(gym.Env):
    """
    OpenAI gym interface for the BoxWorld env implemented by DeepMind as part of their pycolab game engine.
    """

    def __init__(
        self,
        grid_size=12,  # note grid shape is really (grid_size+2, grid_size+2) bc of border
        solution_length=(1, 2, 3, 4),
        num_forward=(0, 1, 2, 3, 4),
        num_backward=(0,),
        branch_length=1,
        max_num_steps=120,
        seed: int = 0,
    ):
        self.grid_size = grid_size
        # extra 2 because of border
        self.shape = (grid_size + 2, grid_size + 2)
        self.solution_length = solution_length
        self.num_forward = num_forward
        self.num_backward = num_backward
        self.branch_length = branch_length
        self.max_num_steps = max_num_steps
        self.random_state = np.random.RandomState(seed)

        # self.action_space = spaces.Discrete(4)
        # self.observation_space = spaces.Box(low=0, high=100, shape=np.zeros(self.shape))

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
    ) -> Tuple[Any, float, bool, dict]:
        """
        Applies action to the box world environment.

        Returns:
            observation (object): agent's observation of the current environment, a numpy array of string characters.
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        if action not in [-1, 0, 1, 2, 3]:
            raise ValueError(f'Invalid action provided: {action}')

        obs, reward, _ = self.game.play(action)
        obs = self.process_obs(obs)
        self.obs = obs
        done = self.game.game_over
        return obs, reward, done, dict()


def hex_to_rgb(hex: str) -> Tuple[int]:
    # https://stackoverflow.com/a/29643643/4383594
    hex = hex.lstrip('#')
    return tuple(int(hex[i:i+2], 16) for i in (0, 2, 4))


def color_name_to_rgb(name: str) -> Tuple[int]:
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
        assert_equal(len(colors), len(s))
        return color_name_to_rgb(colors[i])


def ascii_to_int(ascii: str):
    # lower and uppercase render the same, just like the colors
    return '# *.abcdefghijklmnopqrst'.index(ascii.lower())


def to_color_obs(obs):
    return np.array([[ascii_to_color(a) for a in row] for row in obs])


def render_obs(obs, option=None, color=True, pause=0.0001):
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
        if option is not None:
            plt.title(f'option={option}')
        fig.canvas.draw()
        plt.pause(pause)
    else:
        print(f'option={option}')
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
             held_key: Optional[str]) -> Tuple[set[str], dict[str, dict[str, int]]]:
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


def is_valid_solution_path(path: List[str], dominoes: set[str], held_key: str) -> bool:
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


def dijkstra(nodes: set, adj_matrix: dict[Any, dict[Any, int]], start, goal) -> List:
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

    unvisited_queue = queue.PriorityQueue()
    visited = set()

    for n in nodes:
        unvisited_queue.put((distances[n], n))

    def get_min_unvisited():
        return unvisited_queue.get()

    def get_path_to(node):
        path = [node]
        while path[-1] != start:
            if node is None:
                raise NoPathError
            node = predecessors[node]
            path.append(node)
        return path[::-1]

    goal_visited = False
    while not goal_visited:
        while True:
            current_dist, current = get_min_unvisited()
            if current not in visited:
                break

        for neighbor, weight_from_current_to_neighbor in adj_matrix[current].items():
            neighbor_dist = distances[neighbor]
            alt_neighbor_dist = current_dist + weight_from_current_to_neighbor
            if alt_neighbor_dist < neighbor_dist:
                distances[neighbor] = alt_neighbor_dist
                predecessors[neighbor] = current
                unvisited_queue.put((alt_neighbor_dist, neighbor))

        goal_visited = current == goal
        visited.add(current)

    return get_path_to(goal)


def make_move_graph(obs, goal_pos) -> Tuple[set[POS], dict[POS, dict[POS, int]]]:
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


def shortest_path(obs, goal_pos: POS) -> List[POS]:
    start_pos = tuple(np.argwhere(obs == bw.PLAYER)[0])
    assert obs[start_pos] == bw.PLAYER

    nodes, adj_matrix = make_move_graph(obs, goal_pos)
    return dijkstra(nodes, adj_matrix, start_pos, goal_pos)


def path_to_moves(path: List[POS]) -> List[int]:
    """
    path is a sequence of (y, x) positions.
    converts to a sequence of up/down/left/right actions.
    """
    def tuple_diff(a, b):
        return a[0] - b[0], a[1] - b[1]

    diffs = [tuple_diff(a, b) for a, b in zip(path[1:], path[:-1])]

    dir_to_action_map = {d: a for (a, d) in bw.ACTION_MAP.items()}
    return [dir_to_action_map[d] for d in diffs]


def generate_traj(env: BoxWorldEnv) -> Tuple[List, List]:
    obs = env.reset()
    # render_obs(obs, pause=1)

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
        option: List[POS] = shortest_path(obs, subgoal_pos)

        for a in path_to_moves(option):
            obs, _, done, _ = env.step(a)
            # render_obs(obs, pause=0.01)
            states.append(obs)
            moves.append(a)
        if len(domino) > 1:
            # move left to pick up new key, or final gem
            obs, _, done, _ = env.step(bw.ACTION_WEST)
            # render_obs(obs, pause=1)
            states.append(obs)
            moves.append(bw.ACTION_WEST)

    assert done, 'uh oh, our path solver didnt actually solve'
    return states, moves


def eval_options_model(control_net, env, n=100, renderer: Callable = None):
    """
    control_net takes in a single observation, and outputs tuple of:
        (b, a) action logps
        (b, 2) stop logps
        (b, ) start logps
    renderer is a callable that takes in obs.
    """
    print(f'Evaluating model on {n} episodes')
    control_net.eval()
    num_solved = 0

    for i in range(n):
        obs = env.reset()
        done, solved = False, False
        t = 0
        options = []

        current_option = None

        while not (done or solved):
            t += 1
            if renderer is not None:
                renderer(obs, current_option)
            obs = obs_to_tensor(obs)
            obs = obs.to(DEVICE)
            # (b, a), (b, 2), (b, )
            action_logps, stop_logps, start_logps = control_net.eval_obs(obs)

            if current_option is not None:
                stop = Categorical(logits=stop_logps[current_option]).sample().item()
            if current_option is None or stop == STOP_NET_STOP_IX:
                current_option = Categorical(logits=start_logps).sample().item()
            options.append(current_option)

            a = Categorical(logits=action_logps[current_option]).sample().item()
            obs, rew, done, info = env.step(a)
            solved = rew == bw.REWARD_GOAL

        if i < 3:
            print(f"options for eval {i}: {options}")
        if solved:
            num_solved += 1

    print(f'Solved {num_solved}/{n} episodes')
    control_net.train()
    return solved


def eval_model(net, env, n=100, renderer: Callable = None):
    """
    renderer is a callable that takes in obs.
    """
    print(f'Evaluating model on {n} episodes')
    net.eval()
    num_solved = 0

    for i in range(n):
        obs = env.reset()
        done, solved = False, False
        t = 0

        while not (done or solved):
            t += 1
            if renderer is not None:
                renderer(obs)
            obs = obs_to_tensor(obs)
            obs = obs.to(DEVICE)
            action_logps = net.eval_obs(obs)
            a = Categorical(logits=action_logps).sample().item()
            obs, rew, done, info = env.step(a)
            solved = rew == bw.REWARD_GOAL

        if solved:
            num_solved += 1

    print(f'Solved {num_solved}/{n} episodes')
    net.train()
    return solved


def obs_to_tensor(obs) -> torch.Tensor:
    obs = torch.tensor([[ascii_to_int(a) for a in row]
                       for row in obs])
    obs = F.one_hot(obs, num_classes=NUM_ASCII).to(torch.float)
    assert_equal(obs.shape[-1], NUM_ASCII)
    obs = rearrange(obs, 'h w c -> c h w')
    return obs


def traj_collate(batch: list[tuple[torch.Tensor, torch.Tensor, int]]):
    """
    batch is a list of (states, moves, length) tuples.
    """
    max_T = max([length for _, _, length in batch])
    states_batch = []
    moves_batch = []
    lengths = []
    for states, moves, length in batch:
        _, *s = states.shape
        T = moves.shape[0]
        to_add = max_T - T
        states2 = torch.cat((states, torch.zeros((to_add, *s))))
        moves2 = torch.cat((moves, torch.zeros(to_add, dtype=int)))
        assert_equal(states2.shape, (max_T + 1, *s))
        assert_equal(moves2.shape, (max_T, ))
        states_batch.append(states2)
        moves_batch.append(moves2)
        lengths.append(length)

    return torch.stack(states_batch), torch.stack(moves_batch), torch.tensor(lengths)


def box_world_dataloader(env: BoxWorldEnv, n: int, traj: bool = True, batch_size: int = 256):
    data = BoxWorldDataset(env, n, traj)
    return DataLoader(data, batch_size=batch_size, shuffle=not traj, collate_fn=traj_collate)


class BoxWorldDataset(Dataset):
    def __init__(self, env: BoxWorldEnv, n: int, traj: bool = False):
        """
        If traj is true, spits out a trajectory and its actions.
        Otherwise, spits out a single state and its action.
        """
        # all in memory
        # list of (states, moves) tuple
        self.data: List[Tuple[List, List]] = [generate_traj(env) for i in range(n)]
        # states, moves = self.data[0]
        # self.data = [(states[0:2], moves[0:1])]
        self.traj = traj

        # ignore last state
        self.states = [obs_to_tensor(s)
                       for states, _ in self.data for s in states[:-1]]
        self.moves = [torch.tensor(m) for _, moves in self.data for m in moves]
        assert_equal(len(self.states), len(self.moves))

        self.traj_states = [torch.stack([obs_to_tensor(s) for s in states]) for states, _ in self.data]
        self.traj_moves = [torch.stack([torch.tensor(m) for m in moves]) for _, moves in self.data]

        self.traj_states, self.traj_moves = zip(*sorted(zip(
            self.traj_states, self.traj_moves), key=lambda t: t[0].shape[0]))
        assert_equal([m.shape[0] + 1 for m in self.traj_moves], [ts.shape[0] for ts in self.traj_states])

    def __len__(self):
        if self.traj:
            return len(self.traj_states)
        else:
            return len(self.states)

    def __getitem__(self, i):
        if self.traj:
            return self.traj_states[i], self.traj_moves[i], len(self.traj_moves[i])
        else:
            return self.states[i], self.moves[i]


@profile(sort_by='cumulative', lines_to_print=20, strip_dirs=True)
def profile_traj_generation2(env):
    for i in range(50):
        generate_traj(env)


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
    profile_traj_generation2(env)
