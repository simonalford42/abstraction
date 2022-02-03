import torch.nn as nn
import torch.nn.functional as F
import torch
from utils import assert_equal

STOP_NET_STOP_IX = 0
STOP_NET_CONTINUE_IX = 1


class AbstractPolicyNet(nn.Module):

    def __init__(self, a, b, t, tau_net, micro_net, stop_net, start_net,
                 alpha_net):
        super().__init__()
        self.a = a  # number of actions
        self.b = b  # number of options
        # self.s = s  # state dim; not actually used
        self.t = t  # abstract state dim
        self.tau_net = tau_net  # s -> t
        self.micro_net = micro_net  # s -> (b, a)  = P(a | b, s)
        self.stop_net = stop_net  # s -> 2b of (stop, 1 - stop) aka beta.
        self.start_net = start_net  # t -> b  aka P(b | t)
        self.alpha_net = alpha_net  # (t + b) -> t abstract transition

    def forward(self, s_i):
        """
        s_i: (T, s) tensor of states
        outputs:
           (T, t) tensor of abstract states t_i
           (T, b, a) tensor of action logps
           (T, b, 2) tensor of stop logps
              the index corresponding to stop/continue are in
              STOP_NET_STOP_IX, STOP_NET_CONTINUE_IX
           (T, b) tensor of start logps
           (T, b, t, t) tensor of causal consistency penalties
        """
        T = s_i.shape[0]

        t_i = self.tau_net(s_i)  # (T, t)
        action_logps = self.micro_net(s_i).reshape(T, self.b, self.a)
        stop_logps = self.stop_net(s_i).reshape(T, self.b, 2)
        start_logps = self.start_net(t_i)  # (T, b) aka P(b | t)
        consistency_penalty = self.calc_consistency_penalty(t_i)  # (T, T, b)

        action_logps = F.log_softmax(action_logps, dim=2)
        stop_logps = F.log_softmax(stop_logps, dim=2)
        start_logps = F.log_softmax(start_logps, dim=1)

        return t_i, action_logps, stop_logps, start_logps, consistency_penalty

    def new_option_logps(self, t):
        """
        Input: an abstract state of shape (t,)
        Output: (b,) logp of different actions.
        """
        return self.start_net(t.unsqueeze(0)).reshape(self.b)

    def alpha_transition(self, t, b):
        """
        Calculate a single abstract transition. Useful for test-time.
        """
        return self.alpha_transitions(t.unsqueeze(0),
                                      torch.tensor([b])).reshape(self.t)

    def alpha_transitions(self, t_i, bs):
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
        t_i2 = self.alpha_net(t_i2)
        return t_i2.reshape(T, nb, self.t)

    def calc_consistency_penalty(self, t_i):
        # TODO: recalculate with einops, compare calculations
        T = t_i.shape[0]
        # apply each action at each timestep.
        alpha_trans = self.alpha_transitions(t_i, torch.arange(self.b))
        alpha_trans = alpha_trans.reshape(T, 1, self.b, self.t)
        t_i2 = t_i.reshape(1, T, 1, self.t)
        # (start, end, action, t value)
        penalty = (t_i2 - alpha_trans)**2
        assert_equal(penalty.shape, (T, T, self.b, self.t))
        # L1 norm
        penalty = penalty.sum(dim=-1)  # (T, T, self.b)
        return penalty


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
                stop_lps = stop_logps[i, :, STOP_NET_STOP_IX]  # (b,)
                one_minus_stop_lps = stop_logps[i, :, STOP_NET_CONTINUE_IX]  # (b,)
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
        final_stop_lps = stop_logps[-1, :, STOP_NET_STOP_IX]  # (b,)
        # broadcast
        total_logp += torch.logsumexp(final_stop_lps.reshape(1, self.b) + option_step_dist, dim=(0, 1))

        # maximize logp, minimize causal inconsistency
        loss = -total_logp + self.consistency_ratio * total_consistency_penalty
        return loss
