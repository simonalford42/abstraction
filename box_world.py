import queue
import copy

from typing import Any, Optional, List, Tuple
import gym
import argparse
import numpy as np
import pycolab
import matplotlib
import matplotlib.pyplot as plt
from utils import assert_equal
from pycolab.examples.research.box_world import box_world as bw

POS = Tuple[int, int]

# colors for keys
COLORS = '*abcdefghijklmnopqrst'[:bw.NUM_COLORS + 1]
# all colors
ASCII = '# .' + COLORS

GOAL_COLOR = '*'
NUM_COLORS = len(COLORS)  # if all colors used, should be 21
assert_equal(NUM_COLORS, bw.NUM_COLORS + 1)  # this is because we count the goal color as a color, while pycolab boxworld does not)
# colors like 'abcdefghijklmnopqrst*' and also the player, background, and border
NUM_ASCII = len(ASCII)
# NUM_ASCII = 64
# assert_equal(NUM_ASCII, 24)


DEFAULT_GRID_SIZE = (14, 14)


class BoxWorldEnv(gym.Env):
    """
    OpenAI gym interface for the BoxWorld env implemented by DeepMind as part of their pycolab game engine.
    """

    def __init__(
        self,
        grid_size=12,  # note grid shape is really (grid_size+2, grid_size+2) bc of border
        solution_length: tuple=(1, 2, 3, 4),
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
        if max(solution_length) + 1 >= bw.NUM_COLORS:
            print(f'WARNING: current # colors limits solution length to be at most {bw.NUM_COLORS - 2}')
            self.solution_length = tuple([x for x in solution_length if x < bw.NUM_COLORS - 1])
        self.num_forward = num_forward
        self.num_backward = num_backward
        self.branch_length = branch_length
        self.max_num_steps = max_num_steps
        self.seed = seed
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
        self.done = False
        self.solved = False
        return obs

    def copy(self):
        return copy.deepcopy(self)

    def process_obs(self, obs) -> np.ndarray:
        obs = np.array([list(row.tobytes().decode('ascii')) for row in obs.board])
        self.obs = obs
        return obs

    def step(self, action: int) -> Tuple[Any, float, bool, dict]:
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
        self.done = done
        self.solved = reward == bw.REWARD_GOAL
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
            and ascii not in ASCII):
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
        i = COLORS.index(ascii)
        colors = ['blue', 'red', 'orange', 'green', 'cyan', 'purple', 'yellow', 'green',
                  'brown', 'maroon', 'gold', 'olive', 'limegreen', 'dodgerblue', 'indigo', 'violet',
                  'orangered', 'greenyellow', 'darkgreen', 'darkblue', 'magenta'][:len(COLORS)]
        return color_name_to_rgb(colors[i])


def ascii_to_int(ascii: str):
    # lower and uppercase render the same, just like the colors
    if ascii.lower() not in ASCII:
        print(f"ascii: {ascii} not found in {ASCII=}")
    return ASCII.index(ascii.lower())


def int_to_ascii(ix: int) -> str:
    return ASCII[ix]


def to_color_obs(obs):
    return np.array([[ascii_to_color(a) for a in row] for row in obs])


def print_obs(obs):
    for row in obs:
        print(''.join(row))


def obs_figure(obs):
    color_array = to_color_obs(obs)
    fig = plt.figure(0)
    plt.clf()
    plt.imshow(color_array / 255)
    return fig


def render_obs(obs, title=None, pause=0.0001):
    """
    Pause tells how long to wait between frames.
    """
    color_array = to_color_obs(obs)

    fig = plt.figure(0)
    plt.clf()
    plt.imshow(color_array / 255)
    if title:
        plt.title(title)
    fig.canvas.draw()
    plt.xticks([])
    plt.yticks([])
    plt.pause(pause)


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
    print(NUM_COLORS)
    print('Enter wasd to move, q to quit')
    while not done:
        # render_obs(obs)
        print_obs(obs)
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

    # render_obs(obs, pause=5)


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
            elif s.islower() and (right_of.isupper() or right_of.islower()):
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


def player_pos(obs) -> tuple[int, int]:
    return tuple(np.argwhere(obs == bw.PLAYER)[0])


def shortest_path(obs, goal_pos: POS) -> List[POS]:
    start_pos = player_pos(obs)
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


def generate_abstract_traj(env: BoxWorldEnv) -> Tuple[List, List]:
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
        options: List[POS] = shortest_path(obs, subgoal_pos)
        done = False

        for a in path_to_moves(options):
            obs, _, done, _ = env.step(a)
            # render_obs(obs, pause=0.01)
            # states.append(obs)
            # moves.append(a)

        if len(domino) > 1:
            # move left to pick up new key, or final gem
            obs, _, done, _ = env.step(bw.ACTION_WEST)
            # render_obs(obs, pause=1)
            # states.append(obs)
            # moves.append(bw.ACTION_WEST)

        if done:
            # add the goal key to the top left, to make program learning for
            # transition function consistent
            obs[0, 0] = bw.GEM

        if len(domino) > 1:
            states.append(obs)
            moves.append(domino)

    # for state in states:
        # render_obs(state)
    # print(moves)
    assert done, 'uh oh, our path solver didnt actually solve'
    return states, moves


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
        play_game(env)
    # profile_traj_generation2(env)
