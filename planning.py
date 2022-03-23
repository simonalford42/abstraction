import time
from queue import PriorityQueue
from utils import assert_equal, assert_shape
import math
from collections import namedtuple

from dataclasses import dataclass, field
from typing import Any, Union


@dataclass(order=True)
class PrioritizedItem:
    priority: float
    item: Any = field(compare=False)


def hlc_bfs(s0, controller, solved_threshold=0.5, timeout=None) -> tuple[list, list[int], float, float]:
    """
    Best first search with the abstract policy.
    solved threshold: probability that we solved the task needed to return something
    Returns:
        (states, actions, logp, solved_logp)
    """
    start = time.time()
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
        if (time.time() - start) > timeout:

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


def llc_plan(abstract_states, abstract_actions):
    """
    for each (state, action) -> new state in the seq, try to find
    """
    env = BoxWorldEnv()





