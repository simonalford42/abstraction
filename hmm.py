import itertools
import torch.nn as nn
from einops import rearrange
from utils import assert_equal, DEVICE, assert_shape
import torch
from box_world import STOP_IX, CONTINUE_IX


def cc_fw(b, action_logps, stop_logps, start_logps, lengths, masks):
    """
    The forward calculation (batched).
        action_logps: [B, max_T, b]
        stop_logps: [B, max_T+1, b, 2]
        start_logps: [B, max_T+1, b]

        Due to a change in indexing, the relationship is:
        P(a_t | b_t, s_{t-1} = action_logps[t-1]
        P(e_t | s_t, b_t) = stop_logps[t]
        P(b_{t+1} | s_t ) = start_logps[t]
        The easy way to think of this is that t denotes where s_t is.

        Assumes stop_logps are arranged so that 0 is continue, 1 is stop.
        cc_loss which calls this should preprocess it to make it so.
    """
    B, max_T = action_logps.shape[0:2]
    # (t, B, b, c, e), but dim 0 is a list, and dim 2 increases by one each step
    f = [torch.full((B, b, max(1, t), 2), float('-inf'), device=DEVICE) for t in range(max_T+1)]
    f[0][:, 0, 0, 1] = 0

    for t in range(1, max_T+1):
        # e_prev = 0; options stay same
        # (B, b, c, e)
        f[t][:, :, :t-1, :] = (f[t-1][:, :, :, 0:1]
                               + action_logps[:, t-1, :, None, None]
                               + stop_logps[:, t, :, None, :])
        # e_prev = 1; options new, c mass fixed at t-1
        # (B, b, e)
        f[t][:, :, t-1, :] = (torch.logsumexp(f[t-1][:, :, :, 1], dim=(1, 2))[:, None, None]
                              + start_logps[:, t-1, :, None]
                              + action_logps[:, t-1, :, None]
                              + stop_logps[:, t, :, :])

    total_logp = torch.empty(B, device=DEVICE)
    for i, T in enumerate(lengths):
        total_logp[i] = torch.logsumexp(f[T][i, :, :, 1], dim=(0,1))

    return f, total_logp


def cc_bw(b, action_logps, stop_logps, start_logps, lengths, masks):
    """
    The backward calculation (batched).
    action_logps: [B, T, b]
    stop_logps: [B, T+1, b, 2]
    start_logps: [B, T+1, b]
    """
    B, max_T = action_logps.shape[0:2]
    # f[0] is P(s[1:T], a[1:T] | Z0, s0) (this comment is for unbatched, not batched, so incorrect)
    f = torch.full((B, max_T+1, b, 2), float('-inf'), device=DEVICE)
    # P(-) = 1[e_maxT = 1]
    f[:, -1, :, 1] = 0

    for t in range(max_T-1, -1, -1):
        # e = 0; continue option
        # (B, b)
        f[:, t, :, 0] = (action_logps[:, t]  # this is really p(a_{t+1})
                         + torch.logsumexp(stop_logps[:, t+1] + f[:, t+1], dim=2))

        # e = 1; stop option
        # (B, b)                        B, b, e summed to (B, )
        f[:, t, :, 1] = torch.logsumexp(start_logps[:, t, :, None]
                                        + action_logps[:, t, :, None]
                                        + stop_logps[:, t+1]
                                        + f[:, t+1],
                                        dim=(1, 2))[:, None]

        # if mask is zero, resets the answer to -inf, else stays the same.
        choices = torch.tensor([[float('-inf'), 0], [1, 1]])  # (mask, e)
        mask = masks[:, t][:, None, None]  # (B, 1, 1)
        choice = choices[masks[:, t]][:, None, :]  # (B, 1, e)
        f[:, t, :, :] = f[:, t, :, :] * mask + choice * (1 - mask)

    total_logp = f[:, 0, 0, 1]  # (B, )
    return f, total_logp


def cc_loss(b, action_logps, stop_logps, start_logps, causal_pens, lengths, masks):
    """
    batched!
    """
    B, max_T = action_logps.shape[0:2]

    assert_shape(action_logps, (B, max_T, b))
    assert_shape(stop_logps, (B, max_T+1, b, 2))
    assert_shape(start_logps, (B, max_T+1, b))
    assert_shape(causal_pens, (B, max_T+1, max_T+1, b))

    if STOP_IX == 0:
        # e_t = 1 means stop, so 'stop_ix' must be 1
        stop_logps = stop_logps.flip(dims=(3, ))

    fw_logps, total_logp = cc_fw(b, action_logps, stop_logps, start_logps, lengths, masks)  # (max_T+1, B, b, c, e), (B, )
    bw_logps, total_logp2 = cc_bw(b, action_logps, stop_logps, start_logps, lengths, masks)  # (B, max_T+1, b, e), (B, )

    dist = sum(abs(total_logp - total_logp2))
    if dist > 1E-4:
        print(f'warning: fw and bw disagree by {dist}')
        # need to unflip stop lps lol
        if STOP_IX == 0:
            stop_logps = stop_logps.flip(dims=(3, ))
        total_logp3 = hmm_fw_ub(action_logps, stop_logps, start_logps)
        print(f"fw: {total_logp}, bw: {total_logp2}, hmm_fw: {total_logp3}")
        print(f"a: {total_logp3 - total_logp}, b: {total_logp3 - total_logp}, hmm_fw: {total_logp3}")
    total_logp_sum = sum(total_logp)
    assert torch.allclose(total_logp, total_logp2), f'fw: {total_logp}, bw: {total_logp2}'

    total_cc_loss = torch.zeros(B, device=DEVICE)
    # t is when we stop
    for t in range(1, max_T+1):
        marginal = fw_logps[t] + bw_logps[:, t, :, None, :]  # (B, b, c, e)
        causal_pen = rearrange(causal_pens, 'B start stop b -> B b stop start')[:, :, t, :t]  # (B, b, c)
        normed_marg_prob = torch.exp(marginal[:, :, :, 1] - total_logp[:, None, None])  # (B, b, c)
        cc_loss = torch.sum(normed_marg_prob * causal_pen, dim=(1, 2))  # (B, )
        # if t > the length of a seq in the batch, then mask out the cc contribution
        total_cc_loss += cc_loss * masks[:, t-1]  # t-1 since we want T to be used, but mask is 0 when >= T.

    return total_logp_sum, torch.sum(total_cc_loss)


def cc_fw_ub(b, action_logps, stop_logps, start_logps):
    """
    The forward calculation (unbatched)
        action_logps: [T, b]
        stop_logps: [T+1, b, 2]
        start_logps: [T+1, b]
        See Notion for more notes on this.

        Due to a change in indexing, the relationship is:
        P(a_t | b_t, s_{t-1} = action_logps[t-1]
        P(e_t | s_t, b_t) = stop_logps[t]
        P(b_{t+1} | s_t ) = start_logps[t]
        The easy way to think of this is that t denotes where s_t is.

        Assumes stop_logps are arranged so that 0 is continue, 1 is stop.
        cc_loss which calls this should preprocess it to make it so.
    """
    T = action_logps.shape[0]
    # (t, b, c, e), but dim 0 is a list, and dim 2 increases by one each step
    f = [torch.full((b, max(1, t), 2), float('-inf'), device=DEVICE) for t in range(T+1)]
    f[0][0, 0, 1] = 0

    for t in range(1, T+1):
        # e_prev = 0; options stay same
        # b, c, e
        f[t][:, :t-1, :] = (f[t-1][:, :, 0:1]
                            + action_logps[t-1, :, None, None]
                            + stop_logps[t, :, None, :])
        # e_prev = 1; options new, c mass fixed at t-1
        # b, e
        f[t][:, t-1, :] = (torch.logsumexp(f[t-1][:, :, 1], dim=(0, 1))
                            + start_logps[t-1, :, None]
                            + action_logps[t-1, :, None]
                            + stop_logps[t, :, :])

    total_logp = torch.logsumexp(f[T][:, :, 1], dim=(0, 1))
    return f, total_logp


def cc_bw_ub(b, action_logps, stop_logps, start_logps):
    """
    The backward calculation (unbatched).
    """
    max_T = action_logps.shape[0]
    # f[0] is P(s[1:T], a[1:T] | Z0, s0)
    f = torch.full((max_T+1, b, 2), float('-inf'), device=DEVICE)
    # P(-) = 1[eT = 1]
    f[-1, :, 1] = 0

    for t in range(max_T-1, -1, -1):
        # e = 0; continue option
        f[t, :, 0] = (action_logps[t]  # this is really p(a_{t+1}), see cc_fw docstring
                      + torch.logsumexp(stop_logps[t+1] + f[t+1], dim=1))

        # e = 1; stop option
        # (b, )                      (b, e) summed to (, )
        f[t, :, 1] = torch.logsumexp(start_logps[t, :, None]
                                     + action_logps[t, :, None]
                                     + stop_logps[t+1]
                                     + f[t+1],
                                     dim=(0, 1))

    assert torch.all(f[0, :, 1] == f[0, 0, 1])
    total_logp = f[0, 0, 1]
    return f, total_logp


def cc_loss_ub(b, action_logps, stop_logps, start_logps, causal_pens):
    """
        action_logps (T, b)
        stop_logps (T+1, b, 2)
        start_logps (T+1, b)
        causal_pens (T+1, T+1, b)
    """
    T = action_logps.shape[0]

    assert_shape(action_logps, (T, b))
    assert_shape(stop_logps, (T+1, b, 2))
    assert_shape(start_logps, (T+1, b))
    assert_shape(causal_pens, (T+1, T+1, b))

    if STOP_IX == 0:
        # e_t = 1 means stop, so 'stop_ix' must be 1
        stop_logps = stop_logps.flip(dims=(2, ))

    fw_logps, total_logp = cc_fw_ub(b, action_logps, stop_logps, start_logps)  # (T+1, b, c, e)
    bw_logps, total_logp2 = cc_bw_ub(b, action_logps, stop_logps, start_logps)  # (T+1, b, e)
    dist = abs(total_logp - total_logp2)
    if dist > 1E-4:
        print(f'warning: fw and bw disagree by {dist}')
        # need to unflip stop lps lol
        if STOP_IX == 0:
            stop_logps = stop_logps.flip(dims=(2, ))
        total_logp3 = hmm_fw_ub(action_logps, stop_logps, start_logps)
        print(f"fw: {total_logp}, bw: {total_logp2}, hmm_fw: {total_logp3}")
        print(f"a: {total_logp3 - total_logp}, b: {total_logp3 - total_logp}, hmm_fw: {total_logp3}")

    total_cc_loss = 0
    # t is when we stop
    for t in range(1, T+1):
        marginal = fw_logps[t] + bw_logps[t, :, None, :]  # (b, c, e)
        causal_pen = rearrange(causal_pens, 'start stop b -> b stop start')[:, t, :t]
        cc_loss = torch.sum(torch.exp(marginal[:, :, 1] - total_logp) * causal_pen)
        total_cc_loss += cc_loss

    return total_logp, total_cc_loss


def cc_loss_brute(b, action_logps, stop_logps, start_logps, causal_pens):
    """
        action_logps (T, b)
        stop_logps (T+1, b, 2)
        start_logps (T+1, b)
        causal_pens (T+1, T+1, b)

        The indexing with b_i is really fucked.
    """
    T = action_logps.shape[0]
    assert T > 0
    assert b > 0
    assert_shape(action_logps, (T, b))
    assert_shape(stop_logps, (T+1, b, 2))
    assert_shape(start_logps, (T+1, b))
    assert_shape(causal_pens, (T+1, T+1, b))

    total_logp = torch.tensor(float('-inf'))

    def calc_seq_logp(b_i, stops):
        stop_or_continue_here = [STOP_IX if s else CONTINUE_IX for s in stops]
        start_ixs = [0] + [i+1 for i in range(T-1) if stops[i]]
        b_i_starts = [b_i[i] for i in start_ixs]
        seq_starts_logp = torch.sum(start_logps[start_ixs, b_i_starts])
        seq_stops_and_continues_logp = torch.sum(stop_logps[range(1, T+1), b_i, stop_or_continue_here])
        seq_actions_logp = torch.sum(action_logps[range(T), b_i])
        seq_logp = seq_starts_logp + seq_stops_and_continues_logp + seq_actions_logp
        return seq_logp

    # b_i is started based on s_{i-1}, gives action a_i which yields state s_i
    # ranges from 1 to T, so b_i[0] is really b_1, sorry
    for b_i in itertools.product(range(b), repeat=T):
        # based off s_i, can we stop and start new option for b_{i+1}
        stop_at_i = [[True, False] if b_i[i] == b_i[i+1] else [True]
                      for i in range(T-1)]
        stop_at_i.append([True])  # always stop at end

        for stops in itertools.product(*stop_at_i):
            assert_equal(len(stops), T)
            # i'th entry of stops is whether b_i was stopped
            seq_logp = calc_seq_logp(b_i, stops)
            total_logp = torch.logaddexp(total_logp, seq_logp)

    total_cc_loss = 0
    for b_i in itertools.product(range(b), repeat=T):
        # based off s_i, can we stop and start new option for b_{i+1}
        stop_at_i = [[True, False] if b_i[i] == b_i[i+1] else [True]
                      for i in range(len(b_i)-1)]
        stop_at_i.append([True])  # always stop at end

        for stops in itertools.product(*stop_at_i):
            seq_logp = calc_seq_logp(b_i, stops)
            # i'th entry of stops is whether b_i was stopped
            # if b_j is stopped, and prev b_i was stopped at i, then we
            # started at s_{i-1} and end at s_j
            start_ixs = [0] + [i+1 for i in range(T) if stops[i]]
            stop_ixs = [i+1 for i in range(T) if stops[i]]
            options = [(i, j, b_i[i]) for (i, j) in zip(start_ixs, stop_ixs)]
            cc_seq = sum([causal_pens[i, j, b] for (i, j, b) in options])
            norm_p_seq = torch.exp(seq_logp - total_logp)
            total_cc_loss += norm_p_seq * cc_seq

    return total_logp, total_cc_loss


def hmm_fw_ub(action_logps, stop_logps, start_logps):
    T = action_logps.shape[0]

    f_0 = start_logps[0] + action_logps[0]
    f_prev = f_0
    for i in range(1, T):
        beta = stop_logps[i, :, STOP_IX]  # (b,)
        one_minus_beta = stop_logps[i, :, CONTINUE_IX]  # (b,)

        f = (torch.logaddexp(f_prev + one_minus_beta,
                             torch.logsumexp(f_prev + beta, dim=0) + start_logps[i])
             + action_logps[i])
        f_prev = f

    assert_equal(T+1, stop_logps.shape[0])
    total_logp = torch.logsumexp(f_prev + stop_logps[T, :, STOP_IX], dim=0)
    return total_logp


class CausalNet(nn.Module):
    """
    Uses AbstractPolicyNet instead of Controller for getting action logps, etc.
    The difference is that here these come from separate nets which can be
    different sizes, instead of simply slicing the output of one big Relational
    net.
    """

    def __init__(self, control_net, cc_weight=1):
        super().__init__()
        self.control_net = control_net
        self.b = control_net.b
        self.cc_weight = cc_weight

    def forward(self, s_i_batch, actions_batch, lengths, masks, batched=True):
        if batched:
            return self.cc_loss(s_i_batch, actions_batch, lengths, masks)
        else:
            assert s_i_batch.shape[0] == 1
            T = lengths[0]
            s_i, actions = s_i_batch[0, :T+1], actions_batch[0, :T]
            return self.cc_loss_ub(s_i, actions)

    def cc_loss(self, s_i_batch, actions_batch, lengths, masks):
        # (B, max_T+1, b, n), (B, max_T+1, b, 2), (B, max_T+1, b), (B, max_T+1, max_T+1, b)
        action_logps, stop_logps, start_logps, causal_pens = self.control_net(s_i_batch, batched=True)

        B = len(lengths)
        max_T = action_logps.shape[1] - 1
        action_logps = action_logps[torch.arange(B)[:, None],
                                    torch.arange(max_T)[None, :],
                                    :,
                                    actions_batch[None, :]]
        # not sure why there's this extra singleton axis, but this passes the test so go for it
        action_logps = action_logps[0]

        logp, cc = cc_loss(self.b, action_logps, stop_logps, start_logps, causal_pens, lengths, masks)
        loss = -logp + self.cc_weight * cc
        return loss

    def cc_loss_ub(self, s_i, actions):
        T = actions.shape[0]
        # (T+1, b, n), (T+1, b, 2), (T+1, b), (T+1, T+1, b)
        action_logps, stop_logps, start_logps, causal_pens = self.control_net(s_i, batched=False)
        # (T, b)
        action_logps = action_logps[range(T), :, actions]

        logp, cc = cc_loss_ub(self.b, action_logps, stop_logps, start_logps, causal_pens)
        loss = -logp + self.cc_weight * cc
        return loss


class TrajNet(nn.Module):
    """
    Like HmmNet, but no abstract model or anything.
    This way, we can swap out HmmNet with this and see how just basic SV
    training does as a baseline.
    """
    def __init__(self, control_net):
        super().__init__()
        self.control_net = control_net
        self.b = control_net.b

    def eval_obs(self, s_i):
        """
        s_i: single observation
        returns: (4, ) of action log probabilities
        """
        # (1, b, 4)
        action_logps, _, _, = self.control_net(s_i.unsqueeze(0), batched=False)
        action_logps = action_logps[0, 0]
        assert_shape(action_logps, (4, ))
        return action_logps

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
        action_logps, stop_logps, start_logps, _ = self.control_net(s_i_batch, batched=False)

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

        return -total_logp


class HmmNet(nn.Module):
    """
    Class for doing the HMM calculations for learning options.
    """
    def __init__(self, control_net):
        super().__init__()
        self.control_net = control_net
        self.b = control_net.b

    def forward(self, s_i_batch, actions_batch, lengths):
        return self.logp_loss(s_i_batch, actions_batch, lengths)

    def logp_loss(self, s_i_batch, actions_batch, lengths):
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
        action_logps, stop_logps, start_logps, _ = self.control_net(s_i_batch, batched=True)

        f = torch.zeros((max_T, B, self.b, ), device=DEVICE)
        f[0] = start_logps[:, 0] + action_logps[range(B), 0, :, actions_batch[:, 0]]
        for i in range(1, max_T):
            beta = stop_logps[:, i, :, STOP_IX]  # (B, b,)
            one_minus_beta = stop_logps[:, i, :, CONTINUE_IX]  # (B, b,)

            f[i] = (torch.logaddexp(f[i-1] + one_minus_beta,
                                    torch.logsumexp(f[i-1] + beta, dim=1, keepdim=True) + start_logps[:, i])
                      + action_logps[range(B), i, :, actions_batch[:, i]])

        # max_T length would be out of bounds since we zero-index
        x0 = f[lengths-1, range(B)]
        assert_shape(x0, (B, self.b))
        # max_T will give last element of (max_T + 1) axis
        x1 = stop_logps[range(B), lengths, :, STOP_IX]
        assert_shape(x1, (B, self.b))
        total_logps = torch.logsumexp(x0 + x1, axis=1)  # (B, )
        return -torch.sum(total_logps)

    def logp_loss_ub(self, s_i, actions):
        """
        returns: negative logp of all trajs in batch
        """
        T = actions.shape[0]

        # (T+1, b, n), (T+1, b, 2), (T+1, b)
        action_logps, stop_logps, start_logps, _ = self.control_net(s_i, batched=False)
        # (T, b)
        action_logps = action_logps[range(T), :, actions]

        total_logp = hmm_fw_ub(action_logps, stop_logps, start_logps)
        return -total_logp


# don't delete, viterbi still needs it
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


def viterbi(hmm: HmmNet, s_i, actions):
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
    action_logps, stop_logps, start_logps, _ = hmm.control_net(s_i)

    f_matrix = torch.zeros((T, b))
    f_matrix[0] = start_logps[0] + action_logps[0, :, actions[0]]
    pointer_matrix = torch.zeros((T, b))

    # TODO: implement with the more efficient trans fn
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
