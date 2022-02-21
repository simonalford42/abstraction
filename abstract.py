import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from einops import rearrange, repeat
from utils import assert_equal, assert_shape
from modules import MicroNet, RelationalDRLNet, FC
import box_world
from box_world import STOP_IX, CONTINUE_IX


class AbstractPolicyNet(nn.Module):
    def __init__(self, a, b, t, tau_net, micro_net, macro_policy_net,
                 macro_transition_net):
        super().__init__()
        self.a = a  # number of actions
        self.b = b  # number of options
        # self.s = s  # state dim; not actually used
        self.t = t  # abstract state dim
        self.tau_net = tau_net  # s -> t
        self.micro_net = micro_net  # s -> (b * a +  2 * b)
        self.macro_policy_net = macro_policy_net  # t -> b  aka P(b | t)
        self.macro_transition_net = macro_transition_net  # (t + b) -> t abstract transition

    def forward(self, s_i):
        """
        s_i: (T, s) tensor of states
        outputs:
           (T, t) tensor of abstract states t_i
           (T, b, a) tensor of action logps
           (T, b, 2) tensor of stop logps
              the index corresponding to stop/continue are in
              STOP_IX, CONTINUE_IX
           (T, b) tensor of start logps
           (T, T, b) tensor of causal consistency penalties
        """
        t_i = self.tau_net(s_i)  # (T, t)
        micro_out = self.micro_net(s_i)
        action_logps = rearrange(micro_out[:, :self.b * self.a], 'T (b a) -> T b a', b=self.b)
        stop_logps = rearrange(micro_out[:, self.b * self.a:], 'T (b two) -> T b two', b=self.b)

        start_logps = self.macro_policy_net(t_i)  # (T, b) aka P(b | t)
        consistency_penalty = self.calc_consistency_penalty(t_i)  # (T, T, b)

        action_logps = F.log_softmax(action_logps, dim=2)
        stop_logps = F.log_softmax(stop_logps, dim=2)
        start_logps = F.log_softmax(start_logps, dim=1)

        return t_i, action_logps, stop_logps, start_logps, consistency_penalty

    def forward_batched(self, s_i_batch):
        """
        s_i: (B, T, s) tensor of states
        outputs:
           (B, T, t) tensor of abstract states t_i
           (B, T, b, a) tensor of action logps
           (B, T, b, 2) tensor of stop logps
              the index corresponding to stop/continue are in
              STOP_IX, CONTINUE_IX
           (B, T, b) tensor of start logps
           (B, T, T, b) tensor of causal consistency penalties
        """
        B, T, *s = s_i_batch.shape

        t_i = self.tau_net(s_i_batch.reshape(B * T, *s))
        t_i = t_i.reshape(B, T, -1)

        assert_shape(t_i, (B, T, self.t))
        micro_out = self.micro_net(s_i_batch.reshape(B * T, *s)).reshape(B, T, -1)
        assert_shape(micro_out, (B, T, self.b * self.a + self.b * 2))
        action_logps = rearrange(micro_out[:, :, :self.b * self.a], 'B T (b a) -> B T b a', a=self.a)
        stop_logps = rearrange(micro_out[:, :, self.b * self.a:], 'B T (b two) -> B T b two', b=self.b)

        start_logps = self.macro_policy_net(t_i.reshape(B * T, self.t)).reshape(B, T, self.b)

        consistency_penalty = self.calc_consistency_penalty_batched(t_i)  # (B, T, T, b)

        action_logps = F.log_softmax(action_logps, dim=3)
        stop_logps = F.log_softmax(stop_logps, dim=3)
        start_logps = F.log_softmax(start_logps, dim=2)

        return t_i, action_logps, stop_logps, start_logps, consistency_penalty

    def new_option_logps(self, t):
        """
        Input: an abstract state of shape (t,)
        Output: (b,) logp of different actions.
        """
        return self.start_net(t.unsqueeze(0)).reshape(self.b)

    def macro_transition(self, t, b):
        """
        Calculate a single abstract transition. Useful for test-time.
        """
        return self.macro_transitions(t.unsqueeze(0),
                                      torch.tensor([b])).reshape(self.t)

    def macro_transitions(self, t_i, bs):
        """
        input: t_i: (T, t) batch of abstract states.
               bs: 1D tensor of actions to try
        returns: (T, |bs|, self.t) batch of new abstract states for each
            option applied.
        """
        # TODO: recalculate with einops, compare calculations
        T = t_i.shape[0]
        nb = bs.shape[0]
        # calculate transition for each t_i + b pair
        t_i2 = t_i.repeat_interleave(nb, dim=0)  # (T*nb, t)
        assert_equal(t_i2.shape, (T * nb, self.t))
        b_onehots = F.one_hot(bs, num_classes=self.b).repeat(T, 1)  # (T*nb, b)
        assert_equal(b_onehots.shape, (T * nb, self.b))
        # b is "less significant', changes in 'inner loop'
        t_i2 = torch.cat((t_i2, b_onehots), dim=1)  # (T*nb, t + b)
        assert_equal(t_i2.shape, (T * nb, self.t + self.b))
        # (T * nb, t + b) -> (T * nb, t)
        t_i2 = self.macro_transition_net(t_i2)
        return t_i2.reshape(T, nb, self.t)

    def macro_transitions2(self, t_i, bs):
        """Returns (T, |bs|, self.t) batch of new abstract states for each option applied.

        Args:
            t_i: (T, t) batch of abstract states
            bs: 1D tensor of actions to try
        """
        T = t_i.shape[0]
        nb = bs.shape[0]
        # calculate transition for each t_i + b pair
        t_i2 = repeat(t_i, 'T t -> (T repeat) t', repeat=nb)
        b_onehots0 = F.one_hot(bs, num_classes=self.b)
        b_onehots = b_onehots0.repeat(T, 1)
        b_onehots2 = repeat(b_onehots0, 'nb b -> (repeat nb) b', repeat=T)
        assert torch.all(b_onehots == b_onehots2)
        # b is 'less significant', changes in 'inner loop'
        t_i2 = torch.cat((t_i2, b_onehots), dim=1)  # (T*nb, t + b)
        assert_equal(t_i2.shape, (T * nb, self.t + self.b))
        # (T * nb, t + b) -> (T * nb, t)
        t_i2 = self.macro_transition_net(t_i2)
        return rearrange(t_i2, '(T nb) t -> T nb t', T=T)

    def calc_consistency_penalty(self, t_i):
        """For each pair of indices and each option, calculates (t - alpha(b, t))^2

        Args:
            t_i: (T, t) tensor of abstract states

        Returns:
            (T, T, b) tensor of penalties
        """
        T = t_i.shape[0]
        # apply each action at each timestep.
        macro_trans = self.macro_transitions(t_i, torch.arange(self.b))
        macro_trans2 = self.macro_transitions2(t_i, torch.arange(self.b))
        assert torch.all(macro_trans == macro_trans2)

        macro_trans1 = macro_trans.reshape(T, 1, self.b, self.t)
        macro_trans2 = rearrange(macro_trans, 'T nb t -> T 1 nb t')
        assert torch.all(macro_trans1 == macro_trans2)

        t_i2 = t_i.reshape(1, T, 1, self.t)
        t_i3 = rearrange(t_i, 'T t -> 1 T 1 t')
        assert torch.all(t_i2 == t_i3)

        # (start, end, action, t value)
        penalty = (t_i2 - macro_trans1)**2
        assert_equal(penalty.shape, (T, T, self.b, self.t))
        # L1 norm
        penalty = penalty.sum(dim=-1)  # (T, T, self.b)
        return penalty

    def calc_consistency_penalty_batched(self, t_i_batch):
        """For each pair of indices and each option, calculates (t - alpha(b, t))^2

        Args:
            t_i_batch: (B, T, t) tensor of abstract states

        Returns:
            (B, T, T, b) tensor of penalties
        """
        B, T = t_i_batch.shape[0:2]

        t_i_flat = rearrange(t_i_batch, 'B T t -> (B T) t')
        macro_trans = self.macro_transitions2(t_i_flat,
                                              torch.arange(self.b))
        macro_trans = rearrange(macro_trans, '(B T) b t -> B T 1 b t', B=B)

        t_i2 = rearrange(t_i_batch, 'B T t -> B 1 T 1 t')

        # (start, end, action, t value)
        penalty = (t_i2 - macro_trans)**2
        assert_shape(penalty, (B, T, T, self.b, self.t))
        penalty = penalty.sum(dim=-1)  # (B, T, T, b)
        return penalty


def boxworld_relational_net(out_dim: int = 4):
    return RelationalDRLNet(input_channels=box_world.NUM_ASCII,
                            num_attn_blocks=2,
                            num_heads=4,
                            out_dim=out_dim)


def attention_apn(b, t):
    a = 4
    tau_net = boxworld_relational_net(out_dim=t)
    input_shape = (14, 14)  # assume default box world grid size
    micro_net = MicroNet(input_shape=input_shape,
                         input_channels=box_world.NUM_ASCII,
                         # both P(a | s, b) and beta(s, b)
                         out_dim=a * b + 2 * b)
    macro_policy_net = FC(input_dim=t, output_dim=b, num_hidden=3, hidden_dim=32)
    macro_transition_net = FC(input_dim=t + b, output_dim=t, num_hidden=3, hidden_dim=32)
    return AbstractPolicyNet(a, b, t, tau_net, micro_net, macro_policy_net, macro_transition_net)


class CCNet(nn.Moduile):
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
        T = len(actions)
        assert_equal(s_i.shape[0], T+1)
        # (T+1, t), (T+1, b, n), (T+1, b, 2), (T+1, b), (T+1, T+1, b)
        t_i, action_logps, stop_logps, start_logps, causal_penalties = self.abstract_policy_net(s_i)
        beta_logps = stop_logps[:, :, STOP_IX]
        one_minus_beta_logps = stop_logps[:, :, CONTINUE_IX]
        # (T+1, b)
        action_logps = action_logps[range(T+1), :, actions]

        return cc_loss(action_logps, beta_logps, one_minus_beta_logps, start_logps, causal_penalties)


def cc_test():
    a = 2
    b = 2
    T = 3
    s_i = (0, 1, 0, 1)
    a_i = (1, 1, 0)

    def p_a_given_s_and_b(a, s, b):
        if a == s and b == 0:
            return 0.75
        elif a != s and b == 1:
            return 0.75
        else:
            return 0.25

    def p_b_given_s(b, s):
        return [[1, 0], [0.75, 0.25]][s][b]

    def p_stop(b, s):
        return [[0.25, 0.75],[0.6, 0.4]][s][b]


    # (T, b)
    action_probs = torch.tensor([[p_a_given_s_and_b(a=a, s=s, b=b1)
                                  for b1 in range(b)]
                                  for a, s in zip(a_i, s_i[:-1])])
    start_probs = torch.tensor([[p_b_given_s(b=b1, s=s) for b1 in range(b)]
                                 for s in s_i[:-1]])
    stop_probs = torch.tensor([[p_stop(b=b1, s=s) for b1 in range(b)]
                                 for s in s_i[:-1]])
    causal_penalties = torch.arange((T+1)**2 * b).reshape(T+1, T+1, 2)

    assert_shape(action_probs, (T, b))
    assert_shape(start_probs, (T, b))
    assert_shape(stop_probs, (T, b))

    action_logps = torch.log(action_probs)
    start_logps = torch.log(start_probs)
    stop_logps = torch.log(stop_probs)
    beta_logps = stop_logps
    one_minus_beta_logps = torch.log(1 - stop_probs)

    # b_vec = 0, 0, 0
    p_start = 1
    p_stop_at_end = stop_probs[-1, 0]
    p_actions = torch.prod(action_probs[:, 0])
    p_dont_stop = torch.prod(stop_probs[1:, 0])
    p_0_0_0 = p_start * p_stop_at_end * p_actions * p_dont_stop
    cc_0_0_0 = causal_penalties[0, 3, 0]

    # b_vec = 0, 1, 0
    p_start = 1
    p_stop1 = stop_probs[1, 0]
    p_stop2 = stop_probs[2, 1]
    p_stop3 = stop_probs[3, 0]
    p_actions = action_probs[0, 0] * action_probs[1, 1] * action_probs[2, 0]
    p_0_1_0 = p_start * p_stop1 * p_stop2 * p_stop3 * p_actions
    cc_0_1_0 = causal_penalties[0, 1, 0] + causal_penalties[1, 2, 1] + causal_penalties[2, 3, 0]

    # b_vec = 0, 1, 1
    p_start = 1
    p_stop1 = stop_probs[1, 0]
    p_stop2 = stop_probs[3, 1]
    p_actions = action_probs[0, 0] * action_probs[1, 1] * action_probs[2, 1]
    p_0_1_1 = p_start * p_stop1 * p_stop2 * p_actions
    cc_0_1_1 = causal_penalties[0, 1, 0] + causal_penalties[1, 3, 1]

    cc_target = (p_0_0_0 * cc_0_0_0) + (p_0_1_0 * cc_0_1_0) + (p_0_1_1 * cc_0_1_1)
    cc = cc_loss(action_logps, beta_logps, one_minus_beta_logps, stop_probs, causal_penalties)
    assert_equal(cc_target, cc)



class Eq2Net(nn.Module):
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

        At a high level:
            - keeps track of distribution over abstract actions, and what timestep
              that abstract action started. so (i+1, b) shape.
            - uses this to calculate expected
        """
        T = len(actions)
        assert_equal(s_i.shape[0], T + 1)
        # (T+1, t), (T+1, b, n), (T+1, b, 2), (T+1, b), (T+1, T+1, b)
        t_i, action_logps, stop_logps, start_logps, consistency_penalties = self.abstract_policy_net(s_i)

        total_logp = 0.
        total_consistency_penalty = 0.
        # (i+1, b) dist over options keeps track of when option started.
        option_step_dist = start_logps[0].unsqueeze(0)
        for i, action in enumerate(actions):
            # invariant: prob dist should sum to 1
            # I was getting error of ~1E-7 which got triggered by default value
            # only applies if no abstract penalty
            if not self.abstract_penalty:
                assert torch.isclose((s := torch.logsumexp(option_step_dist, (0, 1))),
                                     torch.tensor(0.), atol=1E-5), \
                       f'Not quite zero: {s}'

            # transition before acting. this way the state at which an option
            # starts is where its first move happens
            # => skip transition for the first step
            if i > 0:
                stop_lps = stop_logps[i, :, STOP_IX]  # (b,)
                one_minus_stop_lps = stop_logps[i, :, CONTINUE_IX]  # (b,)
                start_lps = start_logps[i]  # (b,)

                # prob mass for options exiting which started at step i; broadcast
                option_step_stops = option_step_dist + stop_lps.reshape(1, self.b)  # (i+1, b)
                total_rearrange = torch.logsumexp(option_step_stops, dim=(0, 1))
                total_rearrange = total_rearrange - self.abstract_penalty
                # distribute new mass among new options. broadcast
                new_mass = start_lps + total_rearrange  # (b,)

                # mass that stays in place, aka doesn't stop; broadcast
                option_step_dist = option_step_dist + one_minus_stop_lps  # (T, b)
                # add new mass at new timestep; TODO: einops?
                option_step_dist = torch.cat((option_step_dist,
                                             new_mass.unsqueeze(0)))

                # causal consistency penalty; start up to current timestep, end here,
                consistency_pens = consistency_penalties[:i + 1, i, :]  # (i+1, b)
                assert_equal(consistency_pens.shape, (i + 1, self.b))
                consistency_penalty = torch.logsumexp(option_step_dist + consistency_pens, dim=(0, 1))
                # TODO: this needs to be a logsumexp
                total_consistency_penalty += consistency_penalty

            action_lps = action_logps[i, :, action]  # (b,)
            # in prob space, this is a sum of probs weighted by macro-dist
            logp = torch.logsumexp(action_lps + option_step_dist, dim=(0, 1))
            # TODO: does this need to be a logsumexp?
            total_logp += logp

        # all macro options need to stop at the very end.
        final_stop_lps = stop_logps[-1, :, STOP_IX]  # (b,)
        # broadcast
        total_logp += torch.logsumexp(final_stop_lps.reshape(1, self.b) + option_step_dist, dim=(0, 1))

        # maximize logp, minimize causal inconsistency
        loss = -total_logp + self.consistency_ratio * total_consistency_penalty
        return loss
