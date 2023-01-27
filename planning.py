import time
import argparse
import random
import itertools
import numpy as np
from matplotlib import pyplot as plt
from queue import PriorityQueue

import box_world
import data
import torch
from abstract import HeteroController
from utils import DEVICE, assert_shape
import utils
import math
from collections import namedtuple, Counter
from data import SOLVED_IX
from box_world import BoxWorldEnv
import box_world as bw
from dataclasses import dataclass, field
from typing import Any, Generator, Optional, Tuple, List
import wandb
from torch.distributions import Categorical
from data import STOP_IX
from torch.utils.data import DataLoader
import torch.nn.functional as F
from einops import rearrange
import hmm
import abstract


@dataclass(order=True)
class PrioritizedItem:
    priority: float
    item: Any = field(compare=False)


def hlc_bfs(s0, control_net, timeout=float('inf'), depth: int=-1
            ) -> Generator[Optional[Tuple[List[int], float, float]], None, None]:
    """
    Best first search with the high level controller abstract policy.
    solved threshold: probability that we solved the task needed to return something
    Returns:
        (actions, logp, solved_logp)
    """
    temp = 1
    start_time = time.time()
    Node = namedtuple("Node", "t prev b logp solved_logp depth")
    solved_logp_threshold = math.log(0.5)
    start_tau = control_net.tau_embed(s0)
    start_solved_logp = control_net.solved_logps(start_tau)[SOLVED_IX]
    start = Node(t=control_net.tau_embed(s0), prev=None, b=None, logp=0.0, solved_logp=start_solved_logp, depth=0)
    num_actions = control_net.b
    expand_queue: PriorityQueue[Node] = PriorityQueue()
    expand_queue.put(PrioritizedItem(-start.logp, start))

    def expand(node: Node):
        # for each b action, gives new output
        action_logps, new_taus, solved_logps = control_net.eval_abstract_policy(node.t)

        solved_logps = solved_logps[:, SOLVED_IX]

        logps = action_logps + node.logp
        logps = logps * temp
        if depth > 0 and node.depth >= depth:
            # set all logps to negative inf
            logps = torch.full_like(logps, -float('inf'))

        assert_shape(logps, (num_actions, ))
        nodes = [Node(t=tau, prev=node, b=b, logp=logp, solved_logp=solved_logp, depth=node.depth + 1)
                 for (tau, b, logp, solved_logp)
                 in zip(new_taus, range(num_actions), logps, solved_logps)]

        for node in nodes:
            expand_queue.put(PrioritizedItem(-node.logp, node))

    def get_path(node: Node):
        states, actions = [], []
        while node is not None:
            states.append(node.t)
            actions.append(node.b)
            node = node.prev
        return states[::-1], actions[:-1][::-1]

    while True:
        if (time.time() - start_time) > timeout:
            print('timed out')
            return

        node = expand_queue.get().item
        # print(f'checking node ({node.b=}, {node.logp=}, {node.solved_logp=})')

        if node.solved_logp >= solved_logp_threshold:
            states, actions = get_path(node)
            # print(f'HL plan proposes path {actions=} with prob {node.solved_logp.exp()}')
            yield actions, node.logp, node.solved_logp

        expand(node)


def llc_sampler(s: torch.Tensor, b, control_net: HeteroController, env, render=False):
    """
    Sample actions from the low level controller until we finish or hit a loop.
    Returns actions: list, final state, done: bool, looped: bool.
    """
    actions = []
    done = env.done
    first_step = True
    pos_visits = {}

    while not done:
        action_logps, stop_logps = control_net.micro_policy(s, b)
        a = torch.argmax(action_logps).item()
        stop = not first_step and torch.argmax(stop_logps) == data.STOP_IX
        if stop:
            break

        actions.append(a)
        obs, rew, done, info = env.step(a)

        pause = 0.2 if first_step else 0.01
        if render:
            box_world.render_obs(obs, pause=pause)

        pos = box_world.player_pos(obs)
        s = bw.obs_to_tensor(obs).to(DEVICE)
        if pos in pos_visits:
            pos_visits[pos] += 1
            if pos_visits[pos] > 5:
                return actions, s, done, True
        else:
            pos_visits[pos] = 1

        first_step = False

    return actions, s, done, False


def llc_plan(options, control_net, env, render=False) -> Tuple[List[List], List, bool]:
    s = bw.obs_to_tensor(env.obs).to(DEVICE)
    all_actions = []
    states_between_options = [s]  # s0 option s1 option s2 ... s_n.
    # print(f'LLC for plan: {abstract_actions}')
    for b in options:
        out = llc_sampler(s, b, control_net, env, render=render)
        actions, s, done, looped = out
        if looped:
            break
        # print(f'LL plan for b={b}: {actions}')
        all_actions.append(actions)
        states_between_options.append(s)
        if done:
            break

    return all_actions, states_between_options, env.solved


def multiple_plan(env, control_net, n, depth, timeout):
    num_solved = 0
    num_tried = 0

    for i in range(n):
        env.reset()
        print(f'Planning for new task {i}')

        # 1. greedy solve.
        # out_dict = data.greedy_solve(env.copy(), control_net, render=False)
        # options = out_dict['options']
        # if out_dict['solved']:
            # print(f'greedy solved with options: {options}; skipping planning')
            # continue
        # else:
            # print(f'greedy FAILED with options: {options}')

        # if depth >= 4:
            # print('Skipping checking all plans for depth >= 4')
        # else:
            # rankings = check_all_plans(env.copy(), control_net, depth=depth)
            # print(f"{rankings=}")
            # if rankings is not None and len(rankings) == 0:
                # print('No solving plans found; skipping planning')
                # continue

        solved, time = plan(env.copy(), control_net, depth=depth, timeout=timeout)
        num_tried += 1
        num_solved += solved
        print(f'Solved {num_solved}/{num_tried}')


def multiple_random_shooting(env, control_net, n, depth):
    for i in range(n):
        env.reset()
        print(f'Random shooting for new task {i}')

        # 1. greedy solve.
        out_dict = data.greedy_solve(env.copy(), control_net, render=False)
        options = out_dict['options']
        if out_dict['solved']:
            print(f'greedy solved with options: {options}; skipping planning')
            continue
        else:
            print(f'greedy FAILED with options: {options}')

        rankings = check_all_plans(env.copy(), control_net, depth=3)
        print(f"{rankings=}")
        if rankings is not None and len(rankings) == 0:
            print('No solving depth 3 plans found; skipping planning')
            continue

        start = time.time()
        while (time.time() - start) < 100:
            options, solved = random_shooting(env.copy(), control_net, depth=depth)
            if solved:
                print(f'Solved with options {options}')
                break
            else:
                print(f'Failed with options {options}')


def check_planning_possible(env, control_net, n):
    possible = []
    greedy_solved = 0

    for i in range(n):
        print(i)
        env.reset()
        # 1. greedy solve.
        out_dict = data.greedy_solve(env.copy(), control_net, render=False)
        if out_dict['solved']:
            greedy_solved += 1
            continue

        rankings = check_all_plans(env.copy(), control_net, depth=3)
        if rankings:
            possible.append((out_dict['options'], rankings))

    print(f'Number possible to solve = {len(possible)}/{n}')
    print(f'Greedy solved {greedy_solved}/{n}')
    print(possible)


def fake_hl_planner(s, control_net):
    plans = itertools.product(range(control_net.b), repeat=4)
    # plans = random.shuffle(list(plans))
    for plan in plans:
        yield (plan[::-1], 0, 0)


L2_SOLVED = 0
L2_ATTEMPTED = 0
BETTER = 0


def eval_sampling(control_net, env, n, macro=False, argmax=False, render=False):
    lengths = []

    total_solved = 0
    for i in range(n):
        env.reset()
        out_dict = data.greedy_solve(env.copy(), control_net, macro=macro, argmax=argmax, render=render)
        solved = out_dict['solved']
        options = out_dict['options']

        if solved:
            total_solved += 1
            lengths.append(len(options))

    print(f'Full sample solved {len(lengths)}/{n}')
    print(Counter(lengths))

    return total_solved


def eval_planner(control_net, env, n):
    control_net.eval()

    # solved_with_model = eval_sampling(control_net, env.copy(), n, macro=True, argmax=False)
    # solved_with_sim = eval_sampling(control_net, env.copy(), n, macro=False, argmax=False)
    # print(f'For sampling, solved {solved_with_model}/{n} with abstract model, {solved_with_sim}/{n} with simulator')
    # wandb.log({'test/model_acc': solved_with_model/n,
    #           'test/simulator_acc': solved_with_sim/n})

    num_solved = 0
    lengths = []
    correct_with_length = {i: 0 for i in range(env.max_num_steps)}
    for i in range(n):
        env.reset()
        obs = bw.obs_to_tensor(env.obs).to(DEVICE)

        out_dict = data.greedy_solve(env, control_net, render=False, argmax=True)
        solved = out_dict['solved']
        options = out_dict['options']

        if solved:
            num_solved += 1
            lengths.append(len(options))

            t = control_net.tau_embed(obs)
            matches = True
            for i in range(1, len(options)):
                t = control_net.macro_transition(t, options[i-1])
                start_logps = control_net.macro_policy_net(t.unsqueeze(0))[0]
                b = torch.argmax(start_logps)
                if b != options[i]:
                    matches = False
            if matches:
                correct_with_length[len(options)] += 1

    control_net.train()

    print(f'Solved {num_solved}/{n}.')
    # print(f"lengths: {lengths}")
    # wandb.log({'test/acc': num_solved/n})
    lengths = Counter(lengths)
    if num_solved > 0:
        for i in range(max(lengths) + 1):
            if lengths[i] > 0:
                length_acc = correct_with_length[i] / lengths[i]
                print(f'\t{i}: {correct_with_length[i]}/{lengths[i]}={length_acc:.2f}')
                # wandb.log({f'test/length_{i}_acc': length_acc})


def plan(env, control_net, depth, timeout):
    s = bw.obs_to_tensor(env.obs).to(DEVICE)
    hl_plan_gen = hlc_bfs(s, control_net, timeout=timeout, depth=depth)
    # hl_plan_gen = fake_hl_planner(s, control_net)
    tried = 0

    while True:
        try:
            (options, logp, solved_logp) = next(hl_plan_gen)
            tried += 1
            actions, _, solved = llc_plan(options, control_net, env.copy())
            if solved:
                print(f'Solved with plan {options[:len(actions)]}')
                return True, time.time()
        except StopIteration:
            print(f'Failed to solve after trying {tried} plans')
            return False, 0


def plan_logp(options, s0, control_net: HeteroController, skip_first=False):
    logp = 0.0
    t0 = control_net.tau_embed(s0)
    t = t0
    first = True
    for option in options:
        action_logps = control_net.macro_policy_net(t.unsqueeze(0))[0]

        if first and skip_first:
            pass
        else:
            logp += action_logps[option]

        first = False

        t = control_net.macro_transition(t, option)

    return logp


def test_llc_stochasticity():
    env = box_world.BoxWorldEnv()

    control_net = utils.load_model('models/b363dbf36c974020a70e6d2876207dad-epoch-25000_control.pt')  # n = 100 full fine tune

    for i in range(5):
        env.reset()
        solved, options, _ = data.greedy_solve(env.copy(), control_net, render=True)
        print(f'solved: {solved} options: {options}')
        if solved:
            for i in range(3):
                actions, _, solved = llc_plan(options, control_net, env.copy(), render=True)


RANKINGS = []


def check_all_plans(env, control_net, depth):
    obs = bw.obs_to_tensor(env.obs).to(DEVICE)

    plans = itertools.product(range(control_net.b), repeat=depth)
    rankings = []
    n_solved = 0
    n_plans = 0
    for options in plans:
        n_plans += 1
        actions, _, solved = llc_plan(options, control_net, env.copy())
        if solved:
            print(f'{options=}')
        if solved and len(actions) < len(options):
            # the task is solved in fewer than depth options, so this wasn't a good example
            print(f'Solved early with options={options[:len(actions)]}')
            return

        logp = plan_logp(options, obs, control_net, skip_first=True)
        rankings.append((options, logp, solved))
        if solved:
            n_solved += 1

    rankings = sorted(rankings, key=lambda t: -t[1])
    rankings = [(*r, i) for i, r in enumerate(rankings)]
    rankings = [r for r in rankings if r[2]]
    return rankings


def check_plan_rankings(env, control_net, depth, n):
    for i in range(n):
        print(f"{i=}")
        env.reset()
        rankings = check_all_plans(env, control_net)
        print(f"{rankings=}")


def random_shooting(env, control_net, depth, sample_size=100):
    def random_trajectory(init_state, control_net, max_steps=depth):
        trajectory = []
        logp = 0.0
        t = control_net.tau_embed(init_state)
        for i in range(max_steps):
            action_logps = control_net.macro_policy_net(t.unsqueeze(0))[0]
            action = Categorical(logits=action_logps).sample().item()
            trajectory.append(action)
            logp = logp * action_logps[action]
            t = control_net.macro_transition(t, action)
        return trajectory, logp

    action_traj: List[List[int]] = []
    option_traj = []

    """
    1. sample sample_size random trajectories
    2. for each trajectory, compute the logp
    3. choose the trajectory with the highest overall logp
    4. take the first action in the trajectory
    5. repeat from new state
    6. repeat until solved or finished (or visit an already seen state).
    """
    current_state = bw.obs_to_tensor(env.obs).to(DEVICE)
    done = False
    while (len(option_traj) <= depth) and (not done):
        box_world.render_obs(env.obs, pause=1, title='current state in random shooting')
        trajectories = [random_trajectory(current_state, control_net) for _ in range(sample_size)]
        print('Random shoots:', Counter([str(t[0]) for t in trajectories]))
        best_trajectory = max(trajectories, key=lambda t: t[1])
        option = best_trajectory[0][0]
        option_traj.append(option)
        actions, current_state, done, _ = llc_sampler(current_state, option, control_net, env, render=False)
        action_traj.append(actions)

    return option_traj, env.solved


if __name__ == '__main__':
    random.seed(0)
    torch.manual_seed(0)

    rnn_model_id = None
    # model_id = '9128babca5684c9caa0c40dc2a09bd97-epoch-175'; control_net = False
    # model_id = 'e36c3e2385d8418a8b1109d78587da68-epoch-1000'; control_net = False

    # model_id = '62f87e8a7da34f5fa84cd7408e84ca54-epoch-21826_control'; control_net = True
    # rnn_model_id = '62f87e8a7da34f5fa84cd7408e84ca54-epoch-21826_rnn'
    model_id = '51a6cc693bc8477ea05d2f5843569098'; control_net = False

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--random_shooting', action='store_true')
    parser.add_argument('-d', '--depth', default=3, type=int)
    parser.add_argument('-sd', '--search_depth', default=0, type=int)
    parser.add_argument('-n', '--n', default=100, type=int)
    parser.add_argument('-e', '--eval', action='store_true')

    args = parser.parse_args()

    if args.search_depth == 0:
        args.search_depth = args.depth

    net = utils.load_model(f'models/{model_id}.pt')
    if control_net:
        control_net = net
    else:
        control_net = net.control_net

    control_net.tau_noise_std = 0
    control_net.eval()

    if rnn_model_id:
        rnn = utils.load_model(f'models/{rnn_model_id}.pt')
        control_net.add_rnn(rnn)

    # control_net = hmm.SVNet(abstract.boxworld_homocontroller(b=1)).control_net
    # env = box_world.BoxWorldEnv(seed=3, solution_length=(args.depth, ))
    env = box_world.BoxWorldEnv(seed=3, random_goal=True)

    with torch.no_grad():
        # check_macro(env, control_net)
        # acc = eval_planner(control_net, env, n=n)
        # test_consistency(env, control_net, n=n)
        # eval_sampling(control_net, env, n=n, render=False, macro=True)
        # check_planning_possible(env, control_net, n=n)

        if args.eval:
            data.eval_options_model(control_net, env, n=args.n, render=False, new_option_pause=.1)
        elif args.random_shooting:
            solve_times = multiple_random_shooting(env, control_net, n=args.n, depth=args.search_depth)
        else:
            solve_times = multiple_plan(env, control_net, n=args.n, depth=args.search_depth, timeout=30)
