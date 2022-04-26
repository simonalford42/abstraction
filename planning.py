import time
import numpy as np
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
            ) -> Generator[Optional[tuple[list[int], float, float]], None, None]:
    """
    Best first search with the high level controller abstract policy.
    solved threshold: probability that we solved the task needed to return something
    Returns:
        (actions, logp, solved_logp)
    """
    temp = 1
    start_time = time.time()
    Node = namedtuple("Node", "t prev b logp solved_logp")
    solved_logp_threshold = math.log(solved_threshold)
    # print(f'solved_logp_threshold: {solved_logp_threshold}')
    start_tau = control_net.tau_embed(s0)
    start_solved_logp = control_net.solved_logps(start_tau)[SOLVED_IX]
    # print(f'start_solved_prob: {torch.exp(start_solved_logp)}')
    start = Node(t=control_net.tau_embed(s0), prev=None, b=None, logp=0.0, solved_logp=start_solved_logp)
    num_actions = control_net.b
    expand_queue: PriorityQueue[Node] = PriorityQueue()
    expand_queue.put(PrioritizedItem(-start.logp, start))

    def expand(node: Node):
        # for each b action, gives new output
        action_logps, new_taus, solved_logps = control_net.eval_abstract_policy(node.t)
        solved_logps = solved_logps[:, SOLVED_IX]
        # print(f'(start) action_logps: {action_logps}')
        # print(f'solved probs: {torch.exp(solved_logps)}')

        logps = action_logps + node.logp
        logps = logps * temp
        # print(f'logps: {logps}')
        print(f'probs: {torch.exp(logps)}')

        assert_shape(logps, (num_actions, ))
        nodes = [Node(t=tau, prev=node, b=b, logp=logp, solved_logp=solved_logp)
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
            yield

        node = expand_queue.get().item

        if node.solved_logp >= solved_logp_threshold:
            states, actions = get_path(node)
            print(f'HLC solved with path {actions=}')
            yield actions, node.logp, node.solved_logp

        # print(f'expanding node Node({node.b=}, {node.logp=}, {node.solved_logp=})')
        expand(node)


def hlc_sampler(s0, control_net) -> tuple[list, list[int], list[float], float]:
    """
    Just samples a high level trajectory. yields so low level plan can fetch as needed until solved.
    """
    pass


def llc_sampler(s: torch.Tensor, b, control_net: HeteroController, env):
    actions = []
    done = env.done
    first_step = True
    pos_visits = {}

    while not done:
        action_logps, stop_logps = control_net.micro_policy(s, b)
        a = torch.argmax(action_logps).item()
        stop = not first_step and torch.argmax(stop_logps) == box_world.STOP_IX
        if stop:
            break

        actions.append(a)
        obs, rew, done, info = env.step(a)
        # pause = 0.2 if first_step else 0.01
        # box_world.render_obs(obs, pause=pause)
        pos = box_world.player_pos(obs)
        if pos in pos_visits:
            pos_visits[pos] += 1
            if pos_visits[pos] > 5:
                return
        else:
            pos_visits[pos] = 1

        s = box_world.obs_to_tensor(obs).to(DEVICE)
        first_step = False

    return actions, s, done


def llc_plan(s: torch.Tensor, abstract_actions, control_net, env) -> tuple[list, bool]:
    all_actions = []
    # print(f'LLC for plan: {abstract_actions}')
    for b in abstract_actions:
        out = llc_sampler(s, b, control_net, env)
        if out is None:
            break
        actions, s, done = out
        # print(f'LL plan for b={b}: {actions}')
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
            global L2_ATTEMPTED, L2_SOLVED
            print(f'Solved number {i})')
            if L2_ATTEMPTED > 0:
                print(f'acc={L2_SOLVED}/{L2_ATTEMPTED}={L2_SOLVED/L2_ATTEMPTED:.2f}')
            num_solved += 1
            solve_times.append(time)

    solve_times = sorted(solve_times)
    return solve_times


def fake_hl_planner(s, control_net):
    b = control_net.b
    for i in range(b):
        for j in range(b):
            yield ([i, j], 0.0, 0.0)


L2_SOLVED = 0
L2_ATTEMPTED = 0


def eval_planner(control_net, env, n):
    return test_consistency(env, control_net, n)


def test_consistency(env, control_net, n):
    num_sample_solved = 0
    num_plan_solved = 0
    num_solved = 0
    for i in range(n):
        obs = env.reset()
        obs = box_world.obs_to_tensor(obs).to(DEVICE)
        env2 = env.copy()
        solved, options = full_sample_solve(env2, env2.obs, control_net, render=False)
        if solved:
            num_solved += 1
            if len(options) > 1:
                num_sample_solved += 1

                t0 = control_net.tau_embed(obs)
                b = options[0]
                t1 = control_net.macro_transition(t0, b)
                start_logps, _, _ = control_net.eval_abstract_policy(t1)
                b1 = torch.argmax(start_logps)

                if b1 == options[1]:
                    num_plan_solved += 1

    # print(f'Tried {n}, solved {num_solved}. Of these, {num_sample_solved} had 2+ options, matched with {num_plan_solved}/{num_sample_solved}={num_plan_solved/num_sample_solved}')
    return num_plan_solved/num_sample_solved


def plan(env, control_net, timeout):
    obs = env.reset()

    for _ in range(1):
        # box_world.render_obs(env2.obs, pause=1)
        env2 = copy.deepcopy(env)
        solved, options = full_sample_solve(env2, env2.obs, control_net, render=False)
        print(f'full sample solved: {solved}, options: {options}')
        if solved and len(options) > 1:
            global L2_ATTEMPTED
            L2_ATTEMPTED += 1

    s = box_world.obs_to_tensor(obs).to(DEVICE)
    hl_plan_gen = hlc_bfs(s, control_net, timeout=timeout)
    # hl_plan_gen = fake_hl_planner(s, control_net)

    while True:
        # so we can simulate multuple times from same start point
        env2 = copy.deepcopy(env)
        # box_world.render_obs(env2.obs, pause=1)
        assert not env2.done
        out = next(hl_plan_gen)
        if out is None:
            return False, timeout
        (actions, logp, solved_logp) = out
        print(f'HL proposal: actions: {actions}, solved logp: {solved_logp}, logp: {logp}')
        assert not env2.done
        actions, solved = llc_plan(s, actions, control_net, env2)
        if solved:
            if len(actions) > 1:
                global L2_SOLVED
                L2_SOLVED += 1
            return True, time.time()
        else:
            return False, 0


def test_tau_solved(tau, tau2, control_net):
    tau, tau2 = tau.detach().numpy(), tau2.detach().numpy()
    n = 15
    taus = [(1 - i/(n-1)) * tau + i/(n-1) * tau2 for i in range(n)]
    solved_probs = [torch.exp(control_net.solved_logps(torch.tensor(t_i))[SOLVED_IX]).item() for t_i in taus]
    ccs = [((t - tau)**2).sum() for t in taus]
    taus = [np.concatenate((t, np.array([p, cc]))) for t, p, cc in zip(taus, solved_probs, ccs)]
    taus = np.array(taus)
    fig, ax = plt.subplots()
    ax.imshow(taus)

    # Loop over data dimensions and create text annotations.
    for j in range(n):
        for i in range(len(tau)):
            ax.text(i, j, f'{taus[j][i].item():.2f}', ha="center", va="center", color="w", fontsize=6)
        ax.text(len(tau), j, f'{np.log(solved_probs[j]):.2f}', ha="center", va="center", color="w", fontsize=6)
        ax.text(len(tau)+1, j, f'{ccs[j]:.4f}', ha="center", va="center", color="w", fontsize=6)

    plt.show()
    input()


def full_sample_solve(env, obs, control_net, render=False, macro=False, argmax=True):
    """
    macro: use macro transition model to base next option from previous trnasition prediction, to teset abstract transition model.
    """
    options_trace = obs
    option_map = {i: [] for i in range(control_net.b)}
    done, solved = False, False
    t = 0
    options = []
    options2 = []
    moves_without_moving = 0
    prev_pos = (-1, -1)
    op_new_tau = None
    op_new_tau_solved_prob = None
    moves = []

    current_option = None

    while not (done or solved):
        t += 1
        obs = box_world.obs_to_tensor(obs).to(DEVICE)
        # (b, a), (b, 2), (b, ), (2, )
        action_logps, stop_logps, start_logps, solved_logits = control_net.eval_obs(obs)

        if current_option is not None:
            if argmax:
                stop = torch.argmax(stop_logps[current_option]).item()
            else:
                stop = Categorical(logits=stop_logps[current_option]).sample().item()
        new_option = current_option is None or stop == STOP_IX
        if new_option:
            if current_option is not None and macro:
                start_logps = control_net.macro_policy_net(op_new_tau)

            tau = control_net.tau_embed(obs)
            if macro:
                tau = op_new_tau

            if current_option is not None:
                causal_consistency = ((tau - op_new_tau)**2).sum()
                # print(f'causal_consistency: {causal_consistency}')
                # print(f'tau: {tau}')
                # print(f'op_new_tau: {op_new_tau}')
                # test_tau_solved(tau, op_new_tau, control_net)
                # print(f'op_new_tau_solved_prob from before: {op_new_tau_solved_prob}')

                options_trace[prev_pos] = 'e'
            # print(f'start probs: {torch.exp(start_logps)}')
            if argmax:
                current_option = torch.argmax(start_logps).item()
            else:
                current_option = Categorical(logits=start_logps).sample().item()

            op_start_logps, op_new_taus, op_solved_logps = control_net.eval_abstract_policy(tau)
            op_new_tau = op_new_taus[current_option]
            # op_new_tau_solved_prob = torch.exp(op_solved_logps[current_option, box_world.SOLVED_IX])
            # print(f'solved prob from option: {op_new_tau_solved_prob}')
            options2.append(current_option)
        else:
            # dont overwrite red dot
            if options_trace[prev_pos] != 'e':
                options_trace[prev_pos] = 'm'

        options.append(current_option)

        a = Categorical(logits=action_logps[current_option]).sample().item()
        option_map[current_option].append(a)
        moves.append(a)

        obs, rew, done, _ = env.step(a)
        if render:
            title = f'option={current_option}'
            pause = 0.01 if new_option else 0.01
            if new_option:
                title += ' (new)'
            option_map[current_option].append((obs, title, pause))
            # box_world.render_obs(obs, title=title, pause=pause)
        solved = rew == bw.REWARD_GOAL

        pos = box_world.player_pos(obs)
        if prev_pos == pos:
            moves_without_moving += 1
        else:
            moves_without_moving = 0
            prev_pos = pos
        if moves_without_moving >= 5:
            done = True

    if solved:
        obs = box_world.obs_to_tensor(obs)
        obs = obs.to(DEVICE)

        # check that we predicted that we solved
        _, _, _, solved_logits = control_net.eval_obs(obs)
        # print(f'END solved prob: {torch.exp(solved_logits[SOLVED_IX])}')

    if render:
        box_world.render_obs(options_trace, title=f'{solved=}', pause=1)

    # print(f'moves:  {moves}')
    # print(f'options:{options}')
    return solved, options2


def plot_times(solve_times, n):
    plt.plot(solve_times, [i/n for i in range(len(solve_times))])
    plt.xlabel('Time (s)')
    plt.ylabel(f'Percent of tasks solved, out of {n}')
    plt.ylim(top=1.0)
    plt.show()


if __name__ == '__main__':
    random.seed(0)
    torch.manual_seed(0)

    env = box_world.BoxWorldEnv(seed=1)
    control_net = utils.load_model('models/e14b78d01cc548239ffd57286e59e819.pt').control_net
    # control_net = utils.load_model('models/b2261e15edc14ffdb4c28c0f96528006.pt').control_net
    # control_net = utils.load_model('models/918fba1553ee47eb8d29ae4b8095f413.pt').control_net
    # control_net = utils.load_model('models/9a3edab864e84db594a3cc2726018bc6-epoch-2500.pt').control_net
    control_net.tau_noise_std = 0

    # box_world.eval_options_model(control_net, env.copy(), n=2, option='verbose')

    n = 50
    test_consistency(env, control_net, n=n)
    # solve_times = multiple_plan(env, control_net, timeout=600, n=n)
