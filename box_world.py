import queue
import os
import random
import copy

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
import torch.nn as nn
from pycolab.examples.research.box_world import box_world as bw


NUM_ASCII = len('# *.abcdefghijklmnopqrst')  # 24

STOP_IX = 0
CONTINUE_IX = 1 - STOP_IX
UNSOLVED_IX, SOLVED_IX = 0, 1

DEFAULT_GRID_SIZE = (14, 14)


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
    print('Enter wasd to move, q to quit')
    while not done:
        render_obs(obs)
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

    render_obs(obs, pause=5)


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


def eval_options_model_interactive(control_net, env, n=100, option='silent', run=None, epoch=None):
    control_net.eval()
    num_solved = 0
    check_cc = hasattr(control_net, 'tau_net')
    check_solved = hasattr(control_net, 'solved_net')
    cc_losses = []
    correct_solved_preds = 0

    for i in range(n):
        env2 = env.copy()
        eval_options_model(control_net, env2, n=1, option='verbose')
        obs = env.reset()
        options_trace = obs
        option_map = {i: [] for i in range(control_net.b)}
        done, solved = False, False
        correct_solved_pred = True
        t = -1
        options = []
        moves_without_moving = 0
        prev_pos = (-1, -1)

        current_option = None
        tau_goal = None

        while not (done or solved):
            t += 1
            render_obs(obs, pause=0.1)
            obs = obs_to_tensor(obs)
            obs = obs.to(DEVICE)
            # (b, a), (b, 2), (b, ), (2, )
            action_logps, stop_logps, start_logps, solved_logits = control_net.eval_obs(obs, option_start_s=obs)

            if check_solved:
                is_solved_pred = torch.argmax(solved_logits) == SOLVED_IX
                # if verbose:
                #     print(f'solved prob: {torch.exp(solved_logits[SOLVED_IX])}')
                #     print(f'is_solved_pred: {is_solved_pred}')
                if is_solved_pred:
                    correct_solved_pred = False

            if current_option is not None:
                stop = Categorical(logits=stop_logps[current_option]).sample().item()
            new_option = current_option is None or stop == STOP_IX
            if new_option:
                if check_cc:
                    tau = control_net.tau_embed(obs)
                if current_option is not None:
                    if check_cc:
                        cc_loss = ((tau_goal - tau)**2).sum()
                        print(f'cc_loss: {cc_loss}')
                        cc_losses.append(cc_loss.item())
                    options_trace[prev_pos] = 'e'
                print(f'Choose new option b; start probs = {torch.exp(start_logps)}')
                b = int(input('b='))
                # current_option = Categorical(logits=start_logps).sample().item()
                current_option = b
                option_start_s = obs
                if check_cc:
                    tau_goal = control_net.macro_transition(tau, current_option)
            else:
                # dont overwrite red dot
                if options_trace[prev_pos] != 'e':
                    options_trace[prev_pos] = 'm'

            options.append(current_option)

            a = Categorical(logits=action_logps[current_option]).sample().item()
            option_map[current_option].append(a)

            obs, rew, done, info = env.step(a)
            solved = rew == bw.REWARD_GOAL

            pos = player_pos(obs)
            if prev_pos == pos:
                moves_without_moving += 1
            else:
                moves_without_moving = 0
                prev_pos = pos
            if moves_without_moving >= 5:
                done = True

        if solved:
            obs = obs_to_tensor(obs)
            obs = obs.to(DEVICE)

            if check_solved:
                # check that we predicted that we solved
                _, _, _, solved_logits = control_net.eval_obs(obs, option_start_s)
                is_solved_pred = torch.argmax(solved_logits) == SOLVED_IX

                if not is_solved_pred:
                    correct_solved_pred = False
                else:
                    if correct_solved_pred:
                        correct_solved_preds += 1

            # add cc loss from last action.
            if check_cc:
                tau = control_net.tau_embed(obs)
                cc_loss = ((tau_goal - tau)**2).sum()
                cc_losses.append(cc_loss.item())
            num_solved += 1

    if check_cc:
        cc_loss_avg = sum(cc_losses) / len(cc_losses)
        if run:
            run[f'test/cc loss avg'].log(cc_loss_avg)
    if check_solved:
        solved_acc = 0 if not num_solved else correct_solved_preds / num_solved
        # print(f'Correct solved pred: {solved_acc:.2f}')
        if run:
            run[f'test/solved pred acc'].log(solved_acc)

    control_net.train()
    if check_cc:
        print(f'Solved {num_solved}/{n} episodes, CC loss avg = {cc_loss_avg}')
    else:
        print(f'Solved {num_solved}/{n} episodes')
    return num_solved / n


def eval_options_model(control_net, env, n=100, option='silent', run=None, epoch=None):
    """
    control_net needs to have fn eval_obs that takes in a single observation,
    and outputs tuple of:
        (b, a) action logps
j       (b, 2) stop logps
        (b, ) start logps

    option:
        - 'silent': evals without any printing
        -
    """
    control_net.eval()
    num_solved = 0
    check_cc = hasattr(control_net, 'tau_net')
    check_solved = hasattr(control_net, 'solved_net')
    verbose = option != 'silent'
    if verbose:
        print(f'Evaluating model on {n} episodes')
    cc_losses = []
    correct_solved_preds = 0

    for i in range(n):
        obs = env.reset()
        if run and i < 10:
            run[f'test/epoch {epoch}/obs'].log(obs_figure(obs), name='obs')
        options_trace = obs
        option_map = {i: [] for i in range(control_net.b)}
        done, solved = False, False
        correct_solved_pred = True
        t = -1
        options = []
        moves_without_moving = 0
        prev_pos = (-1, -1)

        current_option = None
        tau_goal = None

        while not (done or solved):
            t += 1
            obs = obs_to_tensor(obs)
            obs = obs.to(DEVICE)
            # (b, a), (b, 2), (b, ), (2, )
            action_logps, stop_logps, start_logps, solved_logits = control_net.eval_obs(obs, option_start_s=obs)
            if current_option is not None:
                stop_prob = torch.exp(stop_logps[current_option, STOP_IX])
                print(f'stop_prob: {stop_prob}')
                # if stop_prob.item() < 0.5:
                    # input()

            if check_solved:
                is_solved_pred = torch.argmax(solved_logits) == SOLVED_IX
                # if verbose:
                #     print(f'solved prob: {torch.exp(solved_logits[SOLVED_IX])}')
                #     print(f'is_solved_pred: {is_solved_pred}')
                if is_solved_pred:
                    correct_solved_pred = False

            if current_option is not None:
                stop = Categorical(logits=stop_logps[current_option]).sample().item()
            new_option = current_option is None or stop == STOP_IX
            if new_option:
                if check_cc:
                    tau = control_net.tau_embed(obs)
                if current_option is not None:
                    if check_cc:
                        cc_loss = ((tau_goal - tau)**2).sum()
                        print(f'cc_loss: {cc_loss}')
                        cc_losses.append(cc_loss.item())
                    options_trace[prev_pos] = 'e'
                current_option = Categorical(logits=start_logps).sample().item()
                option_start_s = obs
                if check_cc:
                    tau_goal = control_net.macro_transition(tau, current_option)
            else:
                # dont overwrite red dot
                if options_trace[prev_pos] != 'e':
                    options_trace[prev_pos] = 'm'

            options.append(current_option)

            a = Categorical(logits=action_logps[current_option]).sample().item()
            option_map[current_option].append(a)

            obs, rew, done, info = env.step(a)
            solved = rew == bw.REWARD_GOAL

            pos = player_pos(obs)
            if prev_pos == pos:
                moves_without_moving += 1
            else:
                moves_without_moving = 0
                prev_pos = pos
            if moves_without_moving >= 5:
                if verbose:
                    print('Quitting due to 5 repeated moves')
                done = True

            if verbose:
                title = f'option={current_option}'
                pause = 0.1 if new_option else 0.05
                if new_option:
                    title += ' (new)'
                option_map[current_option].append((obs, title, pause))
                render_obs(obs, title=title, pause=pause)

        if solved:
            obs = obs_to_tensor(obs)
            obs = obs.to(DEVICE)

            if check_solved:
                # check that we predicted that we solved
                _, _, _, solved_logits = control_net.eval_obs(obs, option_start_s)
                is_solved_pred = torch.argmax(solved_logits) == SOLVED_IX
                if verbose:
                    print(f'END is_solved_pred: {is_solved_pred}')
                    print(f'solved_logits: {torch.exp(solved_logits)}')

                if not is_solved_pred:
                    correct_solved_pred = False
                else:
                    if correct_solved_pred:
                        correct_solved_preds += 1

            # add cc loss from last action.
            if check_cc:
                tau = control_net.tau_embed(obs)
                cc_loss = ((tau_goal - tau)**2).sum()
                cc_losses.append(cc_loss.item())
            num_solved += 1

        if verbose:
            render_obs(options_trace, title=f'{solved=}', pause=1)
        if run and i < 10:
            run[f'test/epoch {epoch}/obs'].log(obs_figure(options_trace),
                                               name='orange=new option')
        print(f"options: {options}")

    if check_cc and len(cc_losses) > 0:
        cc_loss_avg = sum(cc_losses) / len(cc_losses)
        if run:
            run[f'test/cc loss avg'].log(cc_loss_avg)
    if check_solved:
        solved_acc = 0 if not num_solved else correct_solved_preds / num_solved
        # print(f'Correct solved pred: {solved_acc:.2f}')
        if run:
            run[f'test/solved pred acc'].log(solved_acc)

    print(f'options: {options}')
    control_net.train()
    if check_cc and len(cc_losses) > 0:
        print(f'Solved {num_solved}/{n} episodes, CC loss avg = {cc_loss_avg}')
    else:
        print(f'Solved {num_solved}/{n} episodes')
    return num_solved / n


def eval_model(net, env, n=100, renderer: Callable = None):
    """
    renderer is a callable that takes in obs.
    """
    # print(f'Evaluating model on {n} episodes')
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
            action_logps, _, _ = net.eval_obs(obs)
            action_logps = action_logps[0]
            a = Categorical(logits=action_logps).sample().item()
            obs, rew, done, info = env.step(a)
            solved = rew == bw.REWARD_GOAL

        if solved:
            num_solved += 1

    print(f'Solved {num_solved}/{n} episodes')
    net.train()
    return num_solved/n


def obs_to_tensor(obs) -> torch.Tensor:
    obs = torch.tensor([[ascii_to_int(a) for a in row]
                       for row in obs])
    obs = F.one_hot(obs, num_classes=NUM_ASCII).to(torch.float)
    assert_equal(obs.shape[-1], NUM_ASCII)
    obs = rearrange(obs, 'h w c -> c h w')
    return obs


def traj_collate(batch: list[tuple[torch.Tensor, torch.Tensor, int]]):
    """
    batch is a list of (states, moves, length, masks) tuples.
    """
    max_T = max([length for _, _, length in batch])
    # max_T = MAX_LEN
    states_batch = []
    moves_batch = []
    lengths = []
    masks = []
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
        mask = torch.zeros(max_T, dtype=int)
        mask[:T] = 1
        masks.append(mask)

    return torch.stack(states_batch), torch.stack(moves_batch), torch.tensor(lengths), torch.stack(masks)


def box_world_dataloader(env: BoxWorldEnv, n: int, traj: bool = True, batch_size: int = 256):
    data = BoxWorldDataset(env, n, traj)
    if traj:
        return DataLoader(data, batch_size=batch_size, shuffle=not traj, collate_fn=traj_collate)
    else:
        return DataLoader(data, batch_size=batch_size, shuffle=not traj)


class BoxWorldDataset(Dataset):
    def __init__(self, env: BoxWorldEnv, n: int, traj: bool = True):
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

        self.traj_states, self.traj_moves = zip(*sorted(zip(self.traj_states, self.traj_moves),
                                                key=lambda t: t[0].shape[0]))
        self.traj_states, self.traj_moves = list(self.traj_states), list(self.traj_moves)
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

    def shuffle(self, batch_size):
        """
        Shuffle trajs which share the same length, while still keeping the overall order the same.
        Then shuffles among the batches.
        """
        ixs = list(range(len(self.traj_states)))
        random.shuffle(ixs)
        self.traj_states[:] = [self.traj_states[i] for i in ixs]
        self.traj_moves[:] = [self.traj_moves[i] for i in ixs]
        self.traj_states[:], self.traj_moves[:] = zip(*sorted(zip(self.traj_states, self.traj_moves),
                                                      key=lambda t: t[0].shape[0]))
        self.traj_states, self.traj_moves = list(self.traj_states), list(self.traj_moves)

        # keep the last batch at the end
        n = len(self.traj_states)
        n = n - (n % batch_size)
        ixs = list(range(n))
        blocks = [ixs[i:i + batch_size] for i in range(0, n, batch_size)]
        random.shuffle(blocks)
        ixs = [b for bs in blocks for b in bs]
        assert_equal(len(ixs), n)
        self.traj_states[:n] = [self.traj_states[i] for i in ixs]
        self.traj_moves[:n] = [self.traj_moves[i] for i in ixs]


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
    play_game(env)
    # profile_traj_generation2(env)
