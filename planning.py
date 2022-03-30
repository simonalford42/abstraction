import time
from matplotlib import pyplot as plt
from queue import PriorityQueue

import torch
from abstract import HeteroController
from utils import DEVICE, assert_equal, assert_shape
import math
from collections import namedtuple
import box_world

from dataclasses import dataclass, field
from typing import Any, Union


@dataclass(order=True)
class PrioritizedItem:
    priority: float
    item: Any = field(compare=False)


def hlc_bfs(s0, controller, solved_threshold=0.5, timeout=float('inf')) -> tuple[list, list[int], float, float]:
    """
    Best first search with the abstract policy.
    solved threshold: probability that we solved the task needed to return something
    Returns:
        (states, actions, logp, solved_logp)
    """
    start_time = time.time()
    Node = namedtuple("Node", "t prev b logp solved_logp")
    solved_logp_threshold = math.log(solved_threshold)
    print(f'solved_logp_threshold: {solved_logp_threshold}')
    start_tau = controller.tau_embed(s0)
    start_solved_logp = controller.solved_logp(start_tau)
    start = Node(t=controller.tau_embed(s0), prev=None, b=None, logp=0.0, solved_logp=start_solved_logp)
    num_actions = controller.b
    expand_queue: PriorityQueue[Node] = PriorityQueue()
    expand_queue.put(PrioritizedItem(-start.logp, start))

    def expand(node: Node) -> Union[bool, tuple[list, list[int], float, float]]:
        action_logps, new_taus, solved_logps = controller.eval_abstract_policy(node.t)
        logps = action_logps + node.logp
        assert_shape(logps, (num_actions, ))
        nodes = [Node(t=tau, prev=node, b=b, logp=logp, solved_logp=solved_logp)
                 for (tau, b, logp, solved_logp)
                 in zip(new_taus, range(num_actions), logps, solved_logps)]

        for node in nodes:
            # print(f'\tnode Node({node.t=}, {node.logp=}, {node.solved_logp=})')
            if node.solved_logp >= solved_logp_threshold:
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
            return None

        node = expand_queue.get().item
        # print(f'expanding node Node({node.t=}, {node.logp=}, {node.solved_logp=})')
        out = expand(node)
        if out is not False:
            return out


def hlc_sampler(s0, controller) -> tuple[list, list[int], list[float], float]:
    """
    Just samples a high level trajectory. yields so low level plan can fetch as needed until solved.
    """
    pass


def llc_sampler(s: torch.Tensor, b, controller: HeteroController, env) -> tuple[list[int], torch.Tensor]:
    actions = []

    while True:
        action_logps, stop_logps = controller.micro_policy(s)
        a = torch.distributions.Categorical(logits=action_logps).sample()
        stop = torch.distributions.Categorical(logits=stop_logps).sample() == box_world.STOP_IX
        if stop:
            break
        actions.append(a)
        s, rew, done, info = env.step(a)
        s = box_world.obs_to_tensor(s).to(DEVICE)

    return actions, s


def llc_plan(s: torch.Tensor, abstract_actions, controller, env):
    all_actions = []
    for b in abstract_actions:
        actions, s = llc_sampler(s, b, controller, env)
        all_actions.append(actions)

    return env.done


def multiple_plan(env, controller, timeout, n):
    num_solved = 0
    solve_times = []
    for i in range(n):
        solved, time = plan(env, controller, timeout)
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


def plan(env, controller, timeout):
    start = time.time()
    while (time.time() - start) < timeout:
        s = box_world.obs_to_tensor(env.reset()).to(DEVICE)
        (states, actions, logp, solved_logp) = hlc_bfs(s, controller, timeout=10)
        done = llc_plan(s, actions, controller, env)
        if done:
            return True, time.time()

    return False, timeout


if __name__ == '__main__':
    env = box_world.BoxWorldEnv()
    n = 50
    solve_times = multiple_plan(env, None, timeout=60, n=n)
    plot_times(solve_times, n=n)
