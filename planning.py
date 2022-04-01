import time
import torch.nn.functional as F
import random
from pycolab.examples.research.box_world import box_world as bw
import copy
from matplotlib import pyplot as plt
from queue import PriorityQueue

import box_world
import torch
from abstract import HeteroController
import abstract
from utils import DEVICE, assert_equal, assert_shape
import utils
import math
from collections import namedtuple
from box_world import SOLVED_IX, STOP_IX
from torch.distributions import Categorical

from dataclasses import dataclass, field
from typing import Any, Union, Generator, Optional


@dataclass(order=True)
class PrioritizedItem:
    priority: float
    item: Any = field(compare=False)


def hlc_bfs(s0, control_net, solved_threshold=0.001, timeout=float('inf')
            ) -> Generator[Optional[tuple[list, list[int], float, float]], None, None]:
    """
    Best first search with the abstract policy.
    solved threshold: probability that we solved the task needed to return something
    Returns:
        (states, actions, logp, solved_logp)
    """
    start_time = time.time()
    Node = namedtuple("Node", "t prev b logp solved_logp")
    solved_logp_threshold = math.log(solved_threshold)
    # print(f'solved_logp_threshold: {solved_logp_threshold}')
    start_tau = control_net.tau_embed(s0)
    start_solved_logp = control_net.solved_logps(start_tau)[SOLVED_IX]
    start = Node(t=control_net.tau_embed(s0), prev=None, b=None, logp=0.0, solved_logp=start_solved_logp)
    num_actions = control_net.b
    expand_queue: PriorityQueue[Node] = PriorityQueue()
    expand_queue.put(PrioritizedItem(-start.logp, start))

    def expand(node: Node) -> Union[bool, tuple[list, list[int], float, float]]:
        # for each b action, gives new output
        action_logps, new_taus, solved_logps = control_net.eval_abstract_policy(node.t)
        solved_logps = solved_logps[:, SOLVED_IX]
        print(f'solved_logps: {solved_logps}')
        print(f'action_logps: {action_logps}')

        logps = action_logps + node.logp

        assert_shape(logps, (num_actions, ))
        nodes = [Node(t=tau, prev=node, b=b, logp=logp, solved_logp=solved_logp)
                 for (tau, b, logp, solved_logp)
                 in zip(new_taus, range(num_actions), logps, solved_logps)]

        for node in nodes:
            # print(f'\tnode Node({node.t=}, {node.logp=}, {node.solved_logp=})')
            if True or node.solved_logp >= solved_logp_threshold:
                states, actions = get_path(node)
                return states, actions, node.logp, node.solved_logp
            expand_queue.put(PrioritizedItem(-node.logp, node))
        return False

    def get_path(node: Node):
        states, actions = [], []
        while node is not None:
            states.append(node.t)
            actions.append(node.b)
            node = node.prev

        return states[::-1], actions[:-1][::-1]

    while True:
        if (time.time() - start_time) > timeout:
            yield

        node = expand_queue.get().item
        # print(f'expanding node Node({node.t=}, {node.logp=}, {node.solved_logp=})')
        out = expand(node)
        if out is not False:
            yield out


def hlc_sampler(s0, control_net) -> tuple[list, list[int], list[float], float]:
    """
    Just samples a high level trajectory. yields so low level plan can fetch as needed until solved.
    """
    pass


def llc_sampler(s: torch.Tensor, b, control_net: HeteroController, env) -> tuple[list[int], torch.Tensor]:
    actions = []
    # box_world.render_obs(env.obs, pause=1)
    done = False

    while not done:
        action_logps, stop_logps = control_net.micro_policy(s, b)
        a = torch.argmax(action_logps).item()
        stop = torch.argmax(stop_logps) == box_world.STOP_IX
        if stop:
            break
        actions.append(a)
        s, rew, done, info = env.step(a)
        solved = rew == bw.REWARD_GOAL
        s = box_world.obs_to_tensor(s).to(DEVICE)

    return actions, s, solved


def llc_plan(s: torch.Tensor, abstract_actions, control_net, env) -> tuple[list, bool]:
    all_actions = []
    for b in abstract_actions:
        actions, s, done = llc_sampler(s, b, control_net, env)
        print(f'LL plan for b={b}: {actions}')
        all_actions.append(actions)
        if done:
            break

    return all_actions, env.solved


def multiple_plan(env, control_net, timeout, n):
    num_solved = 0
    solve_times = []
    for i in range(n):
        solved, time = plan(env, control_net, timeout)
        if solved:
            num_solved += 1
            solve_times.append(time)

    solve_times = sorted(solve_times)
    return solve_times


def plot_times(solve_times, n):
    plt.plot(solve_times, [i/n for i in range(len(solve_times))])
    plt.xlabel('Time (s)')
    plt.ylabel(f'Percent of tasks solved, out of {n}')
    plt.ylim(top=1.0)
    plt.show()


def fake_hl_planner(s, control_net):
    for i in range(control_net.b):
        yield (None, [i], 0, 0)


def plan(env, control_net, timeout):
    obs = env.reset()

    for _ in range(1):
        env2 = copy.deepcopy(env)
        box_world.render_obs(env2.obs, pause=1)
        solved, options = full_sample_solve(env2, env2.obs, control_net, render=False)  # what
        print(f'solved: {solved}, options: {options}')

    s = box_world.obs_to_tensor(obs).to(DEVICE)
    hl_plan_gen = hlc_bfs(s, control_net, timeout=timeout)
    # hl_plan_gen = fake_hl_planner(s, control_net)

    while True:
        # so we can simulate multuple times from same start point
        env2 = copy.deepcopy(env)
        box_world.render_obs(env2.obs, pause=1)
        assert not env2.done
        out = next(hl_plan_gen)
        if out is None:
            return False, timeout
        (states, actions, logp, solved_logp) = next(hl_plan_gen)
        print(f'HL proposal: actions: {actions}, solved logp: {solved_logp}, logp: {logp}')
        assert not env2.done
        actions, solved = llc_plan(s, actions, control_net, env2)
        if solved:
            print('Solved!')
            return True, time.time()


def full_sample_solve(env, obs, control_net, render=False):
    options_trace = obs
    option_map = {i: [] for i in range(control_net.b)}
    done, solved = False, False
    t = 0
    options = []
    options2 = []
    moves_without_moving = 0
    prev_pos = (-1, -1)

    current_option = None

    while not (done or solved):
        t += 1
        obs = box_world.obs_to_tensor(obs)
        obs = obs.to(DEVICE)
        # (b, a), (b, 2), (b, ), (2, )
        action_logps, stop_logps, start_logps, solved_logits = control_net.eval_obs(obs)

        if current_option is not None:
            stop = Categorical(logits=stop_logps[current_option]).sample().item()
        new_option = current_option is None or stop == STOP_IX
        if new_option:
            if current_option is not None:
                options_trace[prev_pos] = 'e'
            else:
                print(f'new option start_logps: {start_logps}')
            current_option = Categorical(logits=start_logps).sample().item()
            options2.append(current_option)
        else:
            # dont overwrite red dot
            if options_trace[prev_pos] != 'e':
                options_trace[prev_pos] = 'm'

        options.append(current_option)

        a = Categorical(logits=action_logps[current_option]).sample().item()
        option_map[current_option].append(a)

        obs, rew, done, _ = env.step(a)
        if render:
            title = f'option={current_option}'
            pause = 0.2 if new_option else 0.1
            if new_option:
                title += ' (new)'
            option_map[current_option].append((obs, title, pause))
            box_world.render_obs(obs, title=title, pause=pause)
        solved = rew == bw.REWARD_GOAL

        pos = box_world.player_pos(obs)
        if prev_pos == pos:
            moves_without_moving += 1
        else:
            moves_without_moving = 0
            prev_pos = pos
        if moves_without_moving >= 5:
            done = True

    if render:
        box_world.render_obs(options_trace, title=f'{solved=}', pause=3)
    return solved, options2


if __name__ == '__main__':
    random.seed(3)
    torch.manual_seed(3)

    env = box_world.BoxWorldEnv(seed=1, solution_length=(1, ))
    control_net = utils.load_model('models/9a8017cb17e24b5db97f959aaedea0d9.pt').control_net

    n = 1
    solve_times = multiple_plan(env, control_net, timeout=30, n=n)
    plot_times(solve_times, n=n)
