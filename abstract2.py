import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import einops
from utils import assert_equal, DEVICE, assert_shape
import torch
from box_world import STOP_IX, CONTINUE_IX
import math


class HmmTrajNet2(nn.Module):
    def __init__(self, abstract_policy_net):
        super().__init__()
        self.abstract_policy_net = abstract_policy_net
        self.b = abstract_policy_net.b

    def forward(self, s_i_batch, actions_batch, lengths):
        """
        s_i: (B, max_T+1, s) tensor
        actions: (B, max_T,) tensor of ints


        lengths: T for each traj in the batch

        returns: negative logp of all trajs in batch
        """
        # return self.forward_old(s_i_batch, actions_batch, lengths)
        B, max_T = actions_batch.shape[0:2]
        assert_equal((B, max_T+1), s_i_batch.shape[0:2])

        # (B, max_T+1, t), (B, max_T+1, b, n), (B, max_T+1, b, 2), (B, max_T+1, b), (B, max_T+1, max_T+1, b)
        t_i, action_logps, stop_logps, start_logps, causal_penalties = self.abstract_policy_net.forward_batched(s_i_batch)

        # Z_t is (b, i, 2); one for each timestep and batch
        f_i = torch.zeros((max_T, B, self.b, max_T, 2), device=DEVICE)
        # initial prob is 0, but 1 for e_0 = 1, c_0 = 0, so set e_0=0 to -inf
        f_i[0] = torch.full((B, self.b, max_T, 2), float('-inf'), device=DEVICE)
        f_i[0, :, 0, 0, 1] = torch.full((B, ), 1, device=DEVICE)

        for i in range(1, max_T):
            assert torch.allclose(torch.logsumexp(f_i[i-1], dim=(1, 2, 3)), torch.zeros((B,)))

            trans_fn = calc_trans_fn_batched(stop_logps, start_logps, i)

            f_unsummed = (rearrange(f_i[i-1], 'B b -> B b 1')
                          + trans_fn
                          + rearrange(action_logps[range(B), i, :, actions_batch[:, i]], 'B b -> B 1 b'))
            f_i[i] = torch.logsumexp(f_unsummed, axis=1)

        # max_T length would be out of bounds since we zero-index
        x0 = f_i[lengths-1, range(B)]
        assert_shape(x0, (B, self.b))
        # max_T will give last element of (max_T + 1) axis
        x1 = stop_logps[range(B), lengths, :, STOP_IX]
        assert_shape(x1, (B, self.b))
        total_logps = torch.logsumexp(x0 + x1, axis=1)  # (B, )
        return -torch.sum(total_logps)

class VanillaController(nn.Module):
    def __init__(self, a, net, batched: bool = False):
        super().__init__()
        self.a = a
        self.b = 1
        self.net = net
        self.batched = batched

    def forward(self, s_i_batch):
        """
        s_i_batch: (B, T, *s) tensor of states
        outputs:
            (B, T, 1, a) tensor of action logps,
            None for stop logps,
            None for start logps
        """
        if not self.batched:
            return self.forward_unbatched(self, s_i_batch)

        B, T, *s = s_i_batch.shape
        action_logps = self.net(s_i_batch.reshape(B * T, *s))
        assert_equal(action_logps.shape, (B * T, self.a))
        action_logps = F.log_softmax(action_logps, dim=1)
        action_logps = action_logps.reshape(B, T, 1, self.a)

        return action_logps, None, None

    def forward_unbatched(self, s_i):
        """
        s_i: (t, *s) tensor of states
        outputs:
            (t, 1, a) tensor of action logps,
            none for stop logps,
            none for start logps
        """
        t = s_i.shape[0]
        action_logps = self.net(s_i)
        assert_equal(action_logps.shape, (t, self.a))
        action_logps = F.log_softmax(action_logps, dim=1)
        action_logps = rearrange(action_logps, 't a -> t 1 a')

        return action_logps, None, None


class Controller(nn.Module):
    def __init__(self, a, b, net, batched: bool = False):
        super().__init__()
        self.a = a
        self.b = b
        self.net = net
        self.batched = batched

    def forward(self, s_i_batch):
        """
        s_i: (B, T, s) tensor of states
        lengths: (B, ) tensor of lengths
        outputs:
            (B, T, b, a) tensor of action logps,
            (B, T, b, 2) tensor of stop logps,
            (B, T, b) tensor of start logps,
        """
        if not self.batched:
            return self.unbatched_forward(s_i_batch)

        B, T, *s = s_i_batch.shape
        out = self.net(s_i_batch.reshape(B * T, *s)).reshape(B, T, -1)
        assert_equal(out.shape[-1], self.a * self.b + 2 * self.b + self.b)
        action_logits = out[:, :, :self.b * self.a].reshape(B, T, self.b, self.a)
        stop_logits = out[:, :, self.b * self.a:self.b * self.a + 2 * self.b].reshape(B, T, self.b, 2)
        start_logits = out[:, :, self.b * self.a + 2 * self.b:]
        assert_equal(start_logits.shape[-1], self.b)
        action_logps = F.log_softmax(action_logits, dim=3)
        stop_logps = F.log_softmax(stop_logits, dim=3)
        start_logps = F.log_softmax(start_logits, dim=2)

        return action_logps, stop_logps, start_logps

    def unbatched_forward(self, s_i):
        """
        s_i: (T, s) tensor of states
        outputs:
            (T, b, a) tensor of action logps
            (T, b, 2) tensor of stop logps,
            (T, b) tensor of start logps
        """
        T = s_i.shape[0]
        out = self.net(s_i)
        assert_equal(out.shape, (T, self.a * self.b + 2 * self.b + self.b))
        action_logits = out[:, :self.b * self.a].reshape(T, self.b, self.a)
        stop_logits = out[:, self.b * self.a:self.b * self.a + 2 * self.b].reshape(T, self.b, 2)
        start_logits = out[:, self.b * self.a + 2 * self.b:]
        assert_equal(start_logits.shape[1], self.b)

        action_logps = F.log_softmax(action_logits, dim=2)
        stop_logps = F.log_softmax(stop_logits, dim=2)
        start_logps = F.log_softmax(start_logits, dim=1)

        return action_logps, stop_logps, start_logps

    def eval_obs(self, s_i):
        """
        For evaluation when we act for a single state.

        s_i: (s, ) tensor

        Returns:
            (b, a) tensor of action logps
            (b, 2) tensor of stop logps
            (b, ) tensor of start logps
        """
        # (1, b, a), (1, b, 2), (1, b)
        action_logps, stop_logps, start_logps = self.unbatched_forward(s_i.unsqueeze(0))
        return action_logps[0], stop_logps[0], start_logps[0]


class Controller2(nn.Module):
    def __init__(self, apn):
        super().__init__()
        self.apn = apn

    def forward(self, s_i_batch):
        """
        s_i: (B, T, s) tensor of states
        lengths: (B, ) tensor of lengths
        outputs:
            (B, T, b, a) tensor of action logps,
            (B, T, b, 2) tensor of stop logps,
            (B, T, b) tensor of start logps,
        """
        if not self.batched:
            return self.unbatched_forward(s_i_batch)

        B, T, *s = s_i_batch.shape
        out = self.net(s_i_batch.reshape(B * T, *s)).reshape(B, T, -1)
        assert_equal(out.shape[-1], self.a * self.b + 2 * self.b + self.b)
        action_logits = out[:, :, :self.b * self.a].reshape(B, T, self.b, self.a)
        stop_logits = out[:, :, self.b * self.a:self.b * self.a + 2 * self.b].reshape(B, T, self.b, 2)
        start_logits = out[:, :, self.b * self.a + 2 * self.b:]
        assert_equal(start_logits.shape[-1], self.b)
        action_logps = F.log_softmax(action_logits, dim=3)
        stop_logps = F.log_softmax(stop_logits, dim=3)
        start_logps = F.log_softmax(start_logits, dim=2)

        return action_logps, stop_logps, start_logps

    def unbatched_forward(self, s_i):
        """
        s_i: (T, s) tensor of states
        outputs:
            (T, b, a) tensor of action logps
            (T, b, 2) tensor of stop logps,
            (T, b) tensor of start logps
        """
        T = s_i.shape[0]
        out = self.net(s_i)
        assert_equal(out.shape, (T, self.a * self.b + 2 * self.b + self.b))
        action_logits = out[:, :self.b * self.a].reshape(T, self.b, self.a)
        stop_logits = out[:, self.b * self.a:self.b * self.a + 2 * self.b].reshape(T, self.b, 2)
        start_logits = out[:, self.b * self.a + 2 * self.b:]
        assert_equal(start_logits.shape[1], self.b)

        action_logps = F.log_softmax(action_logits, dim=2)
        stop_logps = F.log_softmax(stop_logits, dim=2)
        start_logps = F.log_softmax(start_logits, dim=1)

        return action_logps, stop_logps, start_logps

    def eval_obs(self, s_i):
        """
        For evaluation when we act for a single state.

        s_i: (s, ) tensor

        Returns:
            (b, a) tensor of action logps
            (b, 2) tensor of stop logps
            (b, ) tensor of start logps
        """
        # (1, b, a), (1, b, 2), (1, b)
        action_logps, stop_logps, start_logps = self.unbatched_forward(s_i.unsqueeze(0))
        return action_logps[0], stop_logps[0], start_logps[0]


class TrajNet(nn.Module):
    """
    Like HMMNet, but no abstract model or anything.
    """
    def __init__(self, control_net):
        super().__init__()
        self.control_net = control_net
        assert control_net.batched
        self.b = control_net.b

    def eval_obs(self, s_i):
        """
        s_i: single observation
        returns: (4, ) of action log probabilities
        """
        # (1, b, 4)
        action_logps, _, _, = self.control_net.unbatched_forward(s_i.unsqueeze(0))
        action_logps = action_logps[0, 0]
        assert_shape(action_logps, (4, ))
        return action_logps

    def forward2(self, s_i_batch, actions_batch, lengths):
        """
        uses eval_obs so we can check eval_obs is implemented the same as
        training, by checking that this gives the same loss as training
        """
        total_logp = 0
        total_correct = 0
        for s_i, actions, length in zip(s_i_batch, actions_batch, lengths):
            for s, a in zip(s_i[:length], actions[:length]):
                action_logps = self.eval_obs(s)
                pred = torch.argmax(action_logps)
                correct = pred == a
                if correct:
                    total_correct += 1
                logp = action_logps[a]
                total_logp += logp

        return -total_logp  # , total_correct

    def forward(self, s_i_batch, actions_batch, lengths):
        """
        s_i: (B, max_T+1, s) tensor
        actions: (B, max_T,) tensor of ints
        lengths: T for each traj in the batch

        returns: negative logp of all trajs in batch
        """
        B, max_T = actions_batch.shape[0:2]
        assert_equal((B, max_T+1), s_i_batch.shape[0:2])

        # (B, max_T+1, b, n), (B, max_T+1, b, 2), (B, max_T+1, b)
        action_logps, stop_logps, start_logps = self.control_net(s_i_batch)

        total_logp = 0
        total_correct = 0
        for i, length in enumerate(lengths):
            logp = torch.sum(action_logps[i, range(length), 0, actions_batch[i, :length]])
            choices = action_logps[i, :length, 0]
            preds = torch.argmax(choices, dim=1)
            assert_equal(preds.shape, actions_batch[i, :length].shape)
            correct = torch.sum(preds == actions_batch[i, :length])
            total_correct += correct
            total_logp += logp

        return -total_logp  # , total_correct


class UnbatchedTrajNet(nn.Module):
    """
    Like HMMNet, but no abstract model or anything.
    """
    def __init__(self, control_net):
        super().__init__()
        self.control_net = control_net
        self.b = control_net.b

    def forward(self, s_i, actions):
        """
        s_i: (T+1, s) tensor
        actions: (T,) tensor of ints

        outputs: negative logp of sequence
        """
        T = actions.shape[0]
        assert_equal(T+1, s_i.shape[0])

        # (T+1, b, n), (T+1, b, 2), (T+1, b)
        action_logps, stop_logps, start_logps = self.control_net(s_i)
        return -torch.sum(action_logps[range(T), 0, actions])


class HMMTrajNet(nn.Module):
    """
    batched Class for doing the HMM calculations for learning options.
    for now just wraps unbactched.
    """
    def __init__(self, control_net):
        super().__init__()
        self.control_net = control_net
        # assert not control_net.batched
        self.b = control_net.b

    def forward(self, s_i_batch, actions_batch, lengths):
        """
        s_i: (B, max_T+1, s) tensor
        actions: (B, max_T,) tensor of ints
        lengths: T for each traj in the batch

        returns: negative logp of all trajs in batch
        """
        # return self.forward_old(s_i_batch, actions_batch, lengths)
        B, max_T = actions_batch.shape[0:2]
        assert_equal((B, max_T+1), s_i_batch.shape[0:2])

        # (B, max_T+1, b, n), (B, max_T+1, b, 2), (B, max_T+1, b)
        action_logps, stop_logps, start_logps = self.control_net(s_i_batch)

        f_i = torch.zeros((max_T, B, self.b, ), device=DEVICE)
        f_i[0] = start_logps[:, 0] + action_logps[range(B), 0, :, actions_batch[:, 0]]
        for i in range(1, max_T):
            # (B, b, b)
            trans_fn = calc_trans_fn_batched(stop_logps, start_logps, i)

            f_unsummed = (rearrange(f_i[i-1], 'B b -> B b 1')
                          + trans_fn
                          + rearrange(action_logps[range(B), i, :, actions_batch[:, i]], 'B b -> B 1 b'))
            f_i[i] = torch.logsumexp(f_unsummed, axis=1)

        # max_T length would be out of bounds since we zero-index
        x0 = f_i[lengths-1, range(B)]
        assert_shape(x0, (B, self.b))
        # max_T will give last element of (max_T + 1) axis
        x1 = stop_logps[range(B), lengths, :, STOP_IX]
        assert_shape(x1, (B, self.b))
        total_logps = torch.logsumexp(x0 + x1, axis=1)  # (B, )
        return -torch.sum(total_logps)

    def forward_old(self, s_i_batch, actions_batch, lengths):
        """
        s_i: (B, max_T+1, s) tensor
        actions: (B, max_T,) tensor of ints
        lengths: T for each traj in the batch

        returns: negative logp of all trajs in batch
        """
        B, max_T = actions_batch.shape[0:2]
        assert_equal((B, max_T+1), s_i_batch.shape[0:2])
        assert B == 1, 'for now stick with batch size 1'

        s_i = s_i_batch.squeeze(0)
        actions = actions_batch.squeeze(0)
        T = lengths[0]
        s_i = s_i[:T+1]
        actions = actions[:T]

        # (T+1, b, n), (T+1, b, 2), (T+1, b)
        action_logps, stop_logps, start_logps = self.control_net(s_i)

        f_0 = start_logps[0] + action_logps[0, :, actions[0]]
        f_prev = f_0
        for i in range(1, T):
            action = actions[i]
            trans_fn = calc_trans_fn(stop_logps, start_logps, i)

            f_unsummed = (rearrange(f_prev, 'b -> b 1')
                          + trans_fn
                          + rearrange(action_logps[i, :, action], 'b -> 1 b'))
            f = torch.logsumexp(f_unsummed, axis=0)
            assert_equal(f.shape, (self.b, ))
            f_prev = f

        assert_equal(T+1, stop_logps.shape[0])
        total_logp = torch.logsumexp(f + stop_logps[T, :, STOP_IX], axis=0)
        print(f"total_logp: {total_logp}")
        return -total_logp


def calc_trans_fn_batched(stop_logps, start_logps, i):
    """
    stop_logps: (B, T+1, b, 2) output from control_net
    start_logps: (B, T+1, b) output from control_net
    i: current time step

    output: (B, b, b) matrix whose k, i,j'th entry is the probability of transitioning
            from option i to option j for trajectory k in the batch.
    """
    (B, _, b) = start_logps.shape
    # for each b' -> b
    beta = stop_logps[:, i, :, STOP_IX]  # (B, b,)
    one_minus_beta = stop_logps[:, i, :, CONTINUE_IX]  # (B, b,)

    continue_trans_fn = torch.full((B, b, b), float('-inf'), device=DEVICE)
    continue_trans_fn[:, torch.arange(b), torch.arange(b)] = one_minus_beta
    # (B, b, b)
    trans_fn = torch.logaddexp(rearrange(beta, 'B b -> B b 1') + rearrange(start_logps[:, i], 'B b -> B 1 b'),
                               continue_trans_fn)
    assert torch.allclose(torch.logsumexp(trans_fn, axis=2), torch.zeros((B, b,), device=DEVICE), atol=1E-6)
    return trans_fn


def calc_trans_fn(stop_logps, start_logps, i):
    """
    stop_logps: (T+1, b, 2) output from control_net
    start_logps: (T+1, b) output from control_net
    i: current time step

    output: (b, b) matrix whose i,j'th entry is the probability of transitioning
            from option i to option j.
    """
    b = start_logps.shape[1]
    # for each b' -> b
    beta = stop_logps[i, :, STOP_IX]  # (b,)
    one_minus_beta = stop_logps[i, :, CONTINUE_IX]  # (b,)

    continue_trans_fn = torch.full((b, b), float('-inf'), device=DEVICE)
    continue_trans_fn[torch.arange(b), torch.arange(b)] = one_minus_beta
    trans_fn = torch.logaddexp(rearrange(beta, 'b -> b 1') + rearrange(start_logps[i], 'b -> 1 b'),
                               continue_trans_fn)
    assert torch.allclose(torch.logsumexp(trans_fn, axis=1), torch.zeros(b, device=DEVICE), atol=1E-6)
    return trans_fn


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
    assert STOP_IX == 0
    stop_probs = torch.stack((stop_probs, one_minus_stop_probs))
    assert_equal(stop_probs.shape, (2, s, b))
    stop_probs = rearrange(stop_probs, 't b s -> s b t')
    stop_probs = torch.stack([stop_probs[s] for s in s_i])
    assert_equal(stop_probs.shape, (4, b, 2))

    start_probs = torch.tensor([[0.5, 0.4], [0.5, 0.6]])
    start_probs = rearrange(start_probs, 'b s -> s b')
    start_probs = torch.stack([start_probs[s] for s in s_i])
    assert_equal(start_probs.shape, (4, b))

    class APN():
        def __init__(self):
            self.a = 2
            self.b = 2
            self.t = 10

        def __call__(self, s_i):
            return None, torch.log(action_probs), torch.log(stop_probs), torch.log(start_probs), None

    hmmnet = HMMTrajNet(APN())
    logp = hmmnet.forward(s_i, actions)
    assert math.isclose(torch.exp(logp), 0.2 * 0.88 / 16, abs_tol=1E-7)


def viterbi(hmm: HMMTrajNet, s_i, actions):
    """
    hmm: a trained HMMNet
    s_i: (T+1, s) tensor
    actions: (T,) tensor of ints

    output: most likely path of options over the sequence
    """
    T = len(actions)
    b = hmm.b
    assert_equal(s_i.shape[0], T + 1)
    # (T+1, b, n), (T+1, b, 2), (T+1, b)
    action_logps, stop_logps, start_logps = hmm.control_net(s_i)

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
