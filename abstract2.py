import torch.nn as nn
from einops import rearrange
import einops
from utils import assertEqual
import torch
from abstract import STOP_NET_STOP_IX, STOP_NET_CONTINUE_IX
import math


def calc_trans_fn(stop_logps, start_logps, i):
    """
    stop_logps: (T+1, b, 2) output from abstract_policy_net
    start_logps: (T+1, b) output from abstract_policy_net
    i: current time step

    output: (b, b) matrix whose i,j'th entry is the probabiolity of transitioning from option i to option j.
    """
    b = start_logps.shape[1]
    # for each b' -> b
    beta = stop_logps[i, :, STOP_NET_STOP_IX]  # (b,)
    one_minus_beta = stop_logps[i, :, STOP_NET_CONTINUE_IX]  # (b,)

    continue_trans_fn = torch.full((b, b), float('-inf'))
    continue_trans_fn[torch.arange(b), torch.arange(b)] = one_minus_beta
    trans_fn = torch.logaddexp(rearrange(beta, 'b -> b 1') + rearrange(start_logps[i], 'b -> 1 b'),
                                continue_trans_fn)
    assert torch.allclose(torch.logsumexp(trans_fn, axis=1), torch.zeros(b), atol=1E-6)
    return trans_fn


class ControlNet(nn.Module):
    """
    Like HMMNet, but no abstract model or anything.
    """
    def __init__(self, abstract_policy_net, abstract_penalty=0.5,
                 consistency_ratio=1.):
        super().__init__()
        self.abstract_policy_net = abstract_policy_net
        self.b = abstract_policy_net.b
        self.t = abstract_policy_net.t

        # logp penalty for longer sequences
        self.abstract_penalty = abstract_penalty
        self.consistency_ratio = consistency_ratio

    def forward(self, s_i, actions):
        """
        s_i: (T+1, s) tensor
        actions: (T,) tensor of ints

        outputs: negative logp of sequence
        """
        T = actions.shape[0]
        assertEqual(T+1, s_i.shape[0])

        # (T+1, t), (T+1, b, n), (T+1, b, 2), (T+1, b), (T+1, T+1, b)
        t_i, action_logps, stop_logps, start_logps, consistency_penalties = self.abstract_policy_net(s_i)
        total = 0
        for i, action in enumerate(actions):
            logp = action_logps[i, 0, action]
            total += logp

        return -total


class HMMNet(nn.Module):
    """
    Class for doing the HMM calculations for learning options.
    """
    def __init__(self, abstract_policy_net, abstract_penalty=0.5,
                 consistency_ratio=1.):
        super().__init__()
        self.abstract_policy_net = abstract_policy_net
        self.a = abstract_policy_net.a
        self.b = abstract_policy_net.b
        self.t = abstract_policy_net.t

        # logp penalty for longer sequences
        self.abstract_penalty = abstract_penalty
        self.consistency_ratio = consistency_ratio

    def forward(self, s_i, actions):
        """
        s_i: (T+1, s) tensor
        actions: (T,) tensor of ints

        outputs: logp of sequence

        HMM calculation, building off Smith et al. 2018.
        """
        T = len(actions)
        assertEqual(s_i.shape[0], T + 1)
        # (T+1, t), (T+1, b, n), (T+1, b, 2), (T+1, b), (T+1, T+1, b)
        _, action_logps, stop_logps, start_logps, _ = self.abstract_policy_net(s_i)

        f_0 = start_logps[0] + action_logps[0, :, actions[0]]
        f_prev = f_0
        for i in range(1, T):
            action = actions[i]
            trans_fn = calc_trans_fn(stop_logps, start_logps, i)

            f_unsummed = (rearrange(f_prev, 'b -> b 1')
                          + trans_fn
                          + rearrange(action_logps[i, :, action], 'b -> 1 b'))
            f = torch.logsumexp(f_unsummed, axis=0)
            assertEqual(f.shape, (self.b, ))
            f_prev = f

        assertEqual(T+1, stop_logps.shape[0])
        total_logp = torch.logsumexp(f + stop_logps[T, :, STOP_NET_STOP_IX], axis=0)
        return -total_logp


def forward_test():
    b = 2
    s = 2

    s_i = torch.tensor([0, 0, 1, 1])
    actions = torch.tensor([0, 0, 0])
    action_probs = torch.tensor([[0.5, 0.5], [0, 1]])
    action_probs = einops.repeat(action_probs, 'b a -> 3 b a')

    # b, s
    stop_probs = torch.tensor([[0, 0.2], [0.5, 0.5]])
    one_minus_stop_probs = 1 - stop_probs
    # stop goes first when concatenating
    assert STOP_NET_STOP_IX == 0
    stop_probs = torch.stack((stop_probs, one_minus_stop_probs))
    assertEqual(stop_probs.shape, (2, s, b))
    stop_probs = rearrange(stop_probs, 't b s -> s b t')
    stop_probs = torch.stack([stop_probs[s] for s in s_i])
    assertEqual(stop_probs.shape, (4, b, 2))

    start_probs = torch.tensor([[0.5, 0.4], [0.5, 0.6]])
    start_probs = rearrange(start_probs, 'b s -> s b')
    start_probs = torch.stack([start_probs[s] for s in s_i])
    assertEqual(start_probs.shape, (4, b))

    class APN():
        def __init__(self):
            self.a = 2
            self.b = 2
            self.t = 10

        def __call__(self, s_i):
            return None, torch.log(action_probs), torch.log(stop_probs), torch.log(start_probs), None

    hmmnet = HMMNet(APN())
    logp = hmmnet.forward(s_i, actions)
    assert math.isclose(torch.exp(logp), 0.2 * 0.88 / 16, abs_tol=1E-7)


def viterbi(hmm: HMMNet, s_i, actions):
    """
    hmm: a trained HMMNet
    s_i: (T+1, s) tensor
    actions: (T,) tensor of ints

    output: most likely path of options over the sequence
    """
    T = len(actions)
    b = hmm.b
    assertEqual(s_i.shape[0], T + 1)
    # (T+1, t), (T+1, b, n), (T+1, b, 2), (T+1, b), (T+1, T+1, b)
    t_i, action_logps, stop_logps, start_logps, consistency_penalties = hmm.abstract_policy_net(s_i)

    f_matrix = torch.zeros((T, b))
    f_matrix[0] = start_logps[0] + action_logps[0, :, actions[0]]
    pointer_matrix = torch.zeros((T, b))

    for t in range(1, T):
        trans_fn = calc_trans_fn(stop_logps, start_logps, t)
        f_new = rearrange(f_matrix[t-1], 'b -> b 1') + trans_fn
        pointers = torch.argmax(f_new, axis=0)

        f_matrix[t] = action_logps[t, :, actions[t]] + torch.max(f_new, axis=0)[0]
        pointer_matrix[t] = pointers

    path = [0] * T
    path[-1] = int(torch.argmax(f_matrix[-1]))
    for t in range(T-1, 0, -1):
        path[t-1] = int(pointer_matrix[t, path[t]])

    return path


if __name__ == '__main__':
    forward_test()
