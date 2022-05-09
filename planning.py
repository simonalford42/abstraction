import time
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
from dataclasses import dataclass, field
from typing import Any, Generator, Optional


@dataclass(order=True)
class PrioritizedItem:
    priority: float
    item: Any = field(compare=False)


def eval_abstract_policy_fake(t, control_net: HeteroController, s, env: BoxWorldEnv):
    new_taus = []
    new_solved_logps = []
    new_states = []
    new_envs = []

    action_logps = control_net.macro_policy_net(t.unsqueeze(0)).reshape((control_net.b, ))
    for b in range(control_net.b):
        env2 = env.copy()
        out = llc_sampler(s, b, control_net, env2)
        if out is None:
            action_logps[b] = float('-inf')
            new_states.append(s)
            new_taus.append(t.detach())
            new_envs.append(env2)
            solved_logps = torch.zeros(2)
            solved_logps[data.SOLVED_IX] = float('-inf')
            new_solved_logps.append(solved_logps)
        else:
            actions, s, solved = out
            new_states.append(s)
            t = control_net.tau_embed(s)
            new_taus.append(t)
            new_envs.append(env2)
            new_solved_logps.append(control_net.solved_logps(t))

    return action_logps, torch.stack(new_taus), torch.stack(new_solved_logps), new_states, new_envs


def hlc_bfs_fake(s0, control_net, env: BoxWorldEnv, solved_threshold=0.001, timeout=float('inf'),) -> Generator[Optional[tuple[list[int], float, float]], None, None]:
    """
    uses low level controller to calc prob of high level actions, so "best case" for planning.
    unfortunately is too memory intensive.
    """
    temp = 1
    start_time = time.time()
    Node = namedtuple("Node", "t prev b logp solved_logp s env")
    solved_logp_threshold = math.log(solved_threshold)
    # print(f'solved_logp_threshold: {solved_logp_threshold}')
    start_tau = control_net.tau_embed(s0)
    start_solved_logp = control_net.solved_logps(start_tau)[SOLVED_IX]
    # print(f'start_solved_prob: {torch.exp(start_solved_logp)}')
    start = Node(t=control_net.tau_embed(s0), prev=None, b=None, logp=0.0, solved_logp=start_solved_logp, env=env.copy(), s=s0)
    num_actions = control_net.b
    expand_queue: PriorityQueue[Node] = PriorityQueue()
    expand_queue.put(PrioritizedItem(-start.logp, start))

    def expand(node: Node):
        # for each b action, gives new output
        action_logps, new_taus, solved_logps, new_states, new_envs = eval_abstract_policy_fake(node.t, control_net, node.s, node.env)

        solved_logps = solved_logps[:, SOLVED_IX]
        # print(f'(start) action_logps: {action_logps}')
        # print(f'solved probs: {torch.exp(solved_logps)}')

        logps = action_logps + node.logp
        logps = logps * temp
        # print(f'logps: {logps}')
        # print(f'probs: {torch.exp(logps)}')

        assert_shape(logps, (num_actions, ))
        nodes = [Node(t=tau, prev=node, b=b, logp=logp, solved_logp=solved_logp, s=s, env=env3)
                 for (tau, b, logp, solved_logp, s, env3)
                 in zip(new_taus, range(num_actions), logps, solved_logps, new_states, new_envs)]

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
            # print(f'HLC solved with path {actions=}')
            yield actions, node.logp, node.solved_logp

        # print(f'expanding node Node({node.b=}, {node.logp=}, {node.solved_logp=})')
        expand(node)


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
        # print(f'probs: {torch.exp(logps)}')

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
            return

        node = expand_queue.get().item

        if node.solved_logp >= solved_logp_threshold:
            states, actions = get_path(node)
            print(f'HLC solved with path {actions=}')
            yield actions, node.logp, node.solved_logp

        print(f'expanding node Node({node.b=}, {node.logp=}, {node.solved_logp=})')
        expand(node)


def llc_sampler(s: torch.Tensor, b, control_net: HeteroController, env, render=False):
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
        if pos in pos_visits:
            pos_visits[pos] += 1
            if pos_visits[pos] > 5:
                return
        else:
            pos_visits[pos] = 1

        s = data.obs_to_tensor(obs).to(DEVICE)
        first_step = False

    return actions, s, done


def llc_plan(options, control_net, env, render=False) -> tuple[list[list], list, bool]:
    s = data.obs_to_tensor(env.obs).to(DEVICE)
    all_actions = []
    states_between_options = [s]  # s0 option s1 option s2 ... s_n.
    # print(f'LLC for plan: {abstract_actions}')
    for b in options:
        out = llc_sampler(s, b, control_net, env, render=render)
        if out is None:
            break
        actions, s, done = out
        # print(f'LL plan for b={b}: {actions}')
        all_actions.append(actions)
        states_between_options.append(s)
        if done:
            break

    return all_actions, states_between_options, env.solved


def multiple_plan(env, control_net, timeout, n):
    num_solved = 0
    solve_times = []
    for i in range(n):
        solved, time = plan(env, control_net)
        if solved:
            global L2_ATTEMPTED, L2_SOLVED
            # if L2_ATTEMPTED > 0:
            #     print(f'acc={L2_SOLVED}/{L2_ATTEMPTED}={L2_SOLVED/L2_ATTEMPTED:.2f}')
            num_solved += 1
            solve_times.append(time)

    global BETTER
    print(f'Solved {num_solved}/{n}, including {BETTER} that full sample did not')
    solve_times = sorted(solve_times)
    return solve_times


def fake_hl_planner(s, control_net):
    plans = itertools.product(range(control_net.b), repeat=4)
    # plans = random.shuffle(list(plans))
    for plan in plans:
        yield (plan[::-1], 0, 0)


L2_SOLVED = 0
L2_ATTEMPTED = 0
BETTER = 0


def eval_sampling(control_net, env, n, macro=False, argmax=False):
    lengths = []

    total_solved = 0
    for i in range(n):
        env.reset()
        solved, options, _ = data.full_sample_solve(env.copy(), control_net, macro=macro, argmax=argmax)
        if solved:
            total_solved += 1
            lengths.append(len(options))

    print(f'Full sample solved {len(lengths)}/{n}')
    print(Counter(lengths))

    return total_solved


def eval_planner(control_net, env, n):
    solved_with_model = eval_sampling(control_net, env.copy(), n, macro=True, argmax=False)
    solved_with_sim = eval_sampling(control_net, env.copy(), n, macro=False, argmax=False)
    print(f'For sampling, solved {solved_with_model}/{n} with abstract model, {solved_with_sim}/{n} with simulator')

    num_solved = 0
    all_options = []
    lengths = []
    correct_with_length = {i: 0 for i in range(env.max_num_steps)}
    for i in range(n):
        env.reset()
        obs = data.obs_to_tensor(env.obs).to(DEVICE)
        solved, options, _ = data.full_sample_solve(env, control_net, render=False, argmax=True)
        all_options.append(options[0])
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

    print(f'Solved {num_solved}/{n}.')
    lengths = Counter(lengths)
    for i in range(max(lengths)):
        if lengths[i] > 0:
            print(f'\t{i}: {correct_with_length[i]}/{lengths[i]}={correct_with_length[i]/lengths[i]:.2f}')

    return sum(correct_with_length.values())/num_solved


def plan(env, control_net, max_hl_plans=-1):
    env.reset()

    full_sample_solved, options, _ = data.full_sample_solve(env.copy(), control_net, render=False)
    print(f'full sample solved: {full_sample_solved}, options: {options}')
    if full_sample_solved and len(options) > 1:
        global L2_ATTEMPTED
        L2_ATTEMPTED += 1

    s = data.obs_to_tensor(env.obs).to(DEVICE)
    # hl_plan_gen = hlc_bfs(s, control_net, timeout=100)
    # hl_plan_gen = hlc_bfs_fake(s, control_net, env, timeout=100)
    hl_plan_gen = fake_hl_planner(s, control_net)
    plan_num = 0
    try:
        while plan_num != max_hl_plans:
            plan_num += 1
            (options, logp, solved_logp) = next(hl_plan_gen)
            # print(f'HL proposal: options: {options}, solved logp: {solved_logp}, logp: {logp}')
            actions, _, solved = llc_plan(options, control_net, env.copy())
            if solved:
                print(f'Solved with plan {options[:len(actions)]}')
                if len(actions) > 1:
                    global L2_SOLVED
                    L2_SOLVED += 1
                if not full_sample_solved:
                    # print('BETTER')
                    global BETTER
                    BETTER += 1
                return True, time.time()
    except StopIteration:
        pass

    print('Failed to solve')
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


def plot_times(solve_times, n):
    plt.plot(solve_times, [i/n for i in range(len(solve_times))])
    plt.xlabel('Time (s)')
    plt.ylabel(f'Percent of tasks solved, out of {n}')
    plt.ylim(top=1.0)
    plt.show()


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
        solved, options, _ = data.full_sample_solve(env.copy(), control_net, render=True)
        print(f'solved: {solved} options: {options}')
        if solved:
            for i in range(3):
                actions, _, solved = llc_plan(options, control_net, env.copy(), render=True)


RANKINGS = []


def check_plan_rankings(env, control_net, depth, n):
    for i in range(n):
        env.reset()
        obs = data.obs_to_tensor(env.obs).to(DEVICE)

        for d in range(depth+1):
            plans = itertools.product(range(control_net.b), repeat=d)
            rankings = []
            n_solved = 0
            n_plans = 0
            for options in plans:
                n_plans += 1
                actions, _, solved = llc_plan(options, control_net, env.copy())
                if solved and len(actions) < len(options):
                    break

                logp = plan_logp(options, obs, control_net, skip_first=True)
                rankings.append((options, logp, solved))
                if solved:
                    n_solved += 1

            if d < 3 and n_solved > 0:
                break
            elif n_solved > 0:
                print(f'For depth={d}, solved w/ {n_solved}/{n_plans} plans')

                rankings = sorted(rankings, key=lambda t: -t[1])
                solved_rankings = []
                for rank, (options, logp, solved) in enumerate(rankings):
                    if solved:
                        solved_rankings.append((rank, options, logp))
                        global RANKINGS
                        RANKINGS.append(rank)
                        print(f'SOLVED {rank=}, {options=}, {logp=}')
                        break
                    # else:
                        # print(f'{rank=}, {options=}, {logp=}')
                break


if __name__ == '__main__':
    random.seed(0)
    torch.manual_seed(0)

    depth = 3
    env = box_world.BoxWorldEnv(seed=1)  # , solution_length=(depth, ))
    # control_net = utils.load_model('models/e14b78d01cc548239ffd57286e59e819.pt').control_net
    # control_net = utils.load_model('models/b2261e15edc14ffdb4c28c0f96528006.pt').control_net
    # control_net = utils.load_model('models/918fba1553ee47eb8d29ae4b8095f413.pt').control_net
    # control_net = utils.load_model('models/9a3edab864e84db594a3cc2726018bc6-epoch-2500.pt').control_net
    # control_net = utils.load_model('models/353d4695ec784f8e87d635eaaeae0270-epoch-1224_control.pt')
    control_net = utils.load_model('models/b363dbf36c974020a70e6d2876207dad-epoch-25000_control.pt')  # n = 100 full fine tune
    # control_net = utils.load_model('models/fcfc0586dcf04d838d20efa7fb5cfcf9-epoch-2500_control.pt')  # n = 20k full fine tune
    control_net.tau_noise_std = 0

    acc = data.eval_options_model(control_net, env.copy(), n=1, option='verbose')
    # print(f'acc: {acc}')

    # test_consistency(env, control_net, n=n)
    # eval_sampling(control_net, env, n=100)
    # solve_times = multiple_plan(env, control_net, timeout=600, n=n)
    # check_plan_rankings(env, control_net, depth=depth, n=100)

    # RANKINGS = sorted(RANKINGS)
    # plt.plot(RANKINGS)
    # plt.show()
