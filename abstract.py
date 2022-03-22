from collections import namedtuple
import torch.nn as nn
import torch.nn.functional as F
import torch
from einops import rearrange, repeat
from utils import assert_equal, assert_shape, DEVICE, logaddexp
from modules import MicroNet, RelationalDRLNet, FC
import box_world


class ConsistencyStopController1(nn.Module):
    """
    alpha(b), beta(s), options take you to different states._
    """
    def __init__(self, a, b, t, tau_net, micro_net, macro_policy_net,
                 macro_transition_net):
        super().__init__()
        self.a = a  # number of actions
        self.b = b  # number of options
        # self.s = s  # state dim; not actually used
        self.t = t  # abstract state dim
        self.tau_net = tau_net  # s -> t
        self.micro_net = micro_net  # s -> (b * a)
        self.macro_policy_net = macro_policy_net  # t -> b  aka P(b | t)
        self.macro_transition_net = macro_transition_net  # b_i one hot -> t abstract transition
        self.b_input = F.one_hot(torch.arange(self.b)).float().to(DEVICE)  # (b, b)

    def forward(self, s_i_batch, batched=False):
        if batched:
            return self.forward_b(s_i_batch)
        else:
            return self.forward_ub(s_i_batch)

    def forward_ub(self, s_i):
        """
        s_i: (T, s) tensor of states
        outputs:
           (T, b, a) tensor of action logps
           (T, b, 2) tensor of stop logps
              the index corresponding to stop/continue are in
              box_world.STOP_IX (0), box_world.CONTINUE_IX (1)
           (T, b) tensor of start logps
           None causal consistency placeholder
        """
        T = s_i.shape[0]
        t_i = self.tau_net(s_i)  # (T, t)
        micro_out = self.micro_net(s_i)
        assert_shape(micro_out, (T, self.b * self.a))
        action_logps = rearrange(micro_out, 'T (b a) -> T b a', b=self.b)
        stop_logps = self.calc_stop_logps_ub(t_i)  # (T, b, 2)
        assert torch.allclose(torch.logsumexp(stop_logps, dim=2),
                              torch.zeros((T, self.b), device=DEVICE),
                              atol=1E-7), f'{stop_logps, torch.logsumexp(stop_logps, dim=2)}'
        start_logps = self.macro_policy_net(t_i)  # (T, b) aka P(b | t)

        action_logps = F.log_softmax(action_logps, dim=2)
        start_logps = F.log_softmax(start_logps, dim=1)
        return action_logps, stop_logps, start_logps, None

    def calc_stop_logps_ub(self, t_i):
        T = t_i.shape[0]
        goal_states = self.macro_transition_net(self.b_input)  # (b, t)
        assert_shape(goal_states, (self.b, self.t))
        assert_shape(t_i, (T, self.t))
        # (T, t, b) - (T, t, b), sum over t axis to get (T, b)
        stop_logps = -(rearrange(t_i, 'T t -> T t 1')
                       - rearrange(goal_states, 'b t -> 1 t b')) ** 2
        stop_logps = stop_logps.sum(dim=1)
        assert_shape(stop_logps, (T, self.b))
        one_minus_stop_logps = logaddexp(torch.zeros_like(stop_logps),
                                         stop_logps,
                                         mask=torch.tensor([1, -1]))
        assert_shape(one_minus_stop_logps, (T, self.b))
        if box_world.STOP_IX == 0:
            stack = (stop_logps, one_minus_stop_logps)
        else:
            stack = (one_minus_stop_logps, stop_logps)
        stop_logps = torch.stack(stack, dim=2)
        assert_shape(stop_logps, (T, self.b, 2))
        return stop_logps

    def forward_b(self, s_i_batch):
        """
        s_i: (B, T, s) tensor of states
        outputs:
           (B, T, b, a) tensor of action logps
           (B, T, b, 2) tensor of stop logps
              the index corresponding to stop/continue are in
              STOP_IX, CONTINUE_IX
           (B, T, b) tensor of start logps
           None CC placeholder
        """
        B, T, *s = s_i_batch.shape

        t_i = self.tau_net(s_i_batch.reshape(B * T, *s))
        t_i = t_i.reshape(B, T, -1)

        assert_shape(t_i, (B, T, self.t))
        micro_out = self.micro_net(s_i_batch.reshape(B * T, *s)).reshape(B, T, -1)
        assert_shape(micro_out, (B, T, self.b * self.a))
        action_logps = rearrange(micro_out, 'B T (b a) -> B T b a', a=self.a)
        stop_logps = self.calc_stop_logps_b(t_i)  # (B, T, b, 2)
        assert torch.allclose(torch.logsumexp(stop_logps, dim=3),
                              torch.zeros((B, T, self.b), device=DEVICE),
                              atol=1E-7), f'{stop_logps, torch.logsumexp(stop_logps, dim=3)}'
        start_logps = self.macro_policy_net(t_i.reshape(B * T, self.t)).reshape(B, T, self.b)

        action_logps = F.log_softmax(action_logps, dim=3)
        start_logps = F.log_softmax(start_logps, dim=2)

        return action_logps, stop_logps, start_logps, None

    def calc_stop_logps_b(self, t_i):
        B, T = t_i.shape[0:2]
        goal_states = self.macro_transition_net(self.b_input)  # (b, t)
        assert_shape(goal_states, (self.b, self.t))
        assert_shape(t_i, (B, T, self.t))
        # (B, T, t, b) - (B, T, t, b), sum over t axis to get (T, b)
        stop_logps = -(rearrange(t_i, 'B T t -> B T t 1')
                       - rearrange(goal_states, 'b t -> 1 1 t b')) ** 2
        stop_logps = stop_logps.sum(dim=2)
        assert_shape(stop_logps, (B, T, self.b))
        one_minus_stop_logps = logaddexp(torch.zeros_like(stop_logps),
                                         stop_logps,
                                         mask=torch.tensor([1, -1]))
        assert_shape(one_minus_stop_logps, (B, T, self.b))
        if box_world.STOP_IX == 0:
            stack = (stop_logps, one_minus_stop_logps)
        else:
            stack = (one_minus_stop_logps, stop_logps)
        stop_logps = torch.stack(stack, dim=3)
        assert_shape(stop_logps, (B, T, self.b, 2))
        return stop_logps


class ConsistencyStopController2(nn.Module):
    """
    alpha(b, tau(s)), beta(s_c, s_t). full abstract model, not markov
    """
    def __init__(self, a, b, t, tau_net, micro_net, macro_policy_net,
                 macro_transition_net):
        super().__init__()
        self.a = a  # number of actions
        self.b = b  # number of options
        # self.s = s  # state dim; not actually used
        self.t = t  # abstract state dim
        self.tau_net = tau_net  # s -> t
        self.micro_net = micro_net  # s -> (b * a)
        self.macro_policy_net = macro_policy_net  # t -> b  aka P(b | t)
        self.macro_transition_net = macro_transition_net  # (t + b) -> t abstract transition

    def forward(self, s_i_batch, batched=False):
        if batched:
            return self.forward_b(s_i_batch)
        else:
            return self.forward_ub(s_i_batch)

    def forward_ub(self, s_i):
        """
        s_i: (T, s) tensor of states
        outputs:
           (T, b, a) tensor of action logps
           (T, T, b, 2) tensor of stop logps
              (start, stop, b, 2)
              the index corresponding to stop/continue are in
              box_world.STOP_IX (0), box_world.CONTINUE_IX (1)
           (T, b) tensor of start logps
           None causal consistency placeholder
        """
        T = s_i.shape[0]
        t_i = self.tau_net(s_i)  # (T, t)
        micro_out = self.micro_net(s_i)
        assert_shape(micro_out, (T, self.b * self.a))
        action_logps = rearrange(micro_out, 'T (b a) -> T b a', b=self.b)
        stop_logps = self.calc_stop_logps_ub(t_i)  # (T, T, b, 2)
        assert torch.allclose(torch.logsumexp(stop_logps, dim=3),
                              torch.zeros((T, T, self.b), device=DEVICE),
                              atol=1E-7), f'{stop_logps, torch.logsumexp(stop_logps, dim=3)}'
        start_logps = self.macro_policy_net(t_i)  # (T, b) aka P(b | t)

        action_logps = F.log_softmax(action_logps, dim=2)
        start_logps = F.log_softmax(start_logps, dim=1)
        return action_logps, stop_logps, start_logps, None

    def macro_transitions(self, t_i, bs):
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

    def calc_stop_logps_ub(self, t_i):
        T = t_i.shape[0]
        alpha_out = self.macro_transitions(t_i, torch.arange(self.b, device=DEVICE))
        assert_shape(alpha_out, (self.T, self.b, self.t))
        assert_shape(t_i, (T, self.t))
        # (T, T, b, t) - (T, T, b, t), sum over t axis to get (T, T, b)
        stop_logps = -(rearrange(t_i,           'T t -> 1 T 1 t')
                       - rearrange(alpha_out, 'T b t -> T 1 b t')) ** 2
        stop_logps = stop_logps.sum(dim=3)
        assert_shape(stop_logps, (T, T, self.b))
        one_minus_stop_logps = logaddexp(torch.zeros_like(stop_logps),
                                         stop_logps,
                                         mask=torch.tensor([1, -1]))
        assert_shape(one_minus_stop_logps, (T, T, self.b))
        if box_world.STOP_IX == 0:
            stack = (stop_logps, one_minus_stop_logps)
        else:
            stack = (one_minus_stop_logps, stop_logps)
        stop_logps = torch.stack(stack, dim=3)
        assert_shape(stop_logps, (T, T, self.b, 2))
        return stop_logps

    def forward_b(self, s_i_batch):
        """
        s_i: (B, T, s) tensor of states
        outputs:
           (B, T, b, a) tensor of action logps
           (B, T, b, 2) tensor of stop logps
              the index corresponding to stop/continue are in
              STOP_IX, CONTINUE_IX
           (B, T, b) tensor of start logps
           None CC placeholder
        """
        B, T, *s = s_i_batch.shape

        t_i = self.tau_net(s_i_batch.reshape(B * T, *s))
        t_i = t_i.reshape(B, T, -1)

        assert_shape(t_i, (B, T, self.t))
        micro_out = self.micro_net(s_i_batch.reshape(B * T, *s)).reshape(B, T, -1)
        assert_shape(micro_out, (B, T, self.b * self.a))
        action_logps = rearrange(micro_out, 'B T (b a) -> B T b a', a=self.a)
        stop_logps = self.calc_stop_logps_b(t_i)  # (B, T, b, 2)
        assert torch.allclose(torch.logsumexp(stop_logps, dim=3),
                              torch.zeros((B, T, self.b), device=DEVICE),
                              atol=1E-7), f'{stop_logps, torch.logsumexp(stop_logps, dim=3)}'
        start_logps = self.macro_policy_net(t_i.reshape(B * T, self.t)).reshape(B, T, self.b)

        action_logps = F.log_softmax(action_logps, dim=3)
        start_logps = F.log_softmax(start_logps, dim=2)

        return action_logps, stop_logps, start_logps, None

    def calc_stop_logps_b(self, t_i):
        B, T = t_i.shape[0:2]

        t_i_flat = rearrange(t_i, 'B T t -> (B T) t')
        alpha_out = self.macro_transitions(t_i_flat, torch.arange(self.b, device=DEVICE))
        assert_shape(alpha_out, (B * T, self.b, self.t))
        assert_shape(t_i, (B, T, self.t))
        # (B, T, T, b, t) - (B, T, T, b, t), sum over t axis to get (B, T, T, b)
        stop_logps = -(rearrange(t_i,             'B T t -> B 1 T 1 t')
                       - rearrange(alpha_out, '(B T) b t -> B T 1 b t', B=B)) ** 2
        stop_logps = stop_logps.sum(dim=2)
        assert_shape(stop_logps, (B, T, self.b))
        one_minus_stop_logps = logaddexp(torch.zeros_like(stop_logps),
                                         stop_logps,
                                         mask=torch.tensor([1, -1]))
        assert_shape(one_minus_stop_logps, (B, T, self.b))
        if box_world.STOP_IX == 0:
            stack = (stop_logps, one_minus_stop_logps)
        else:
            stack = (one_minus_stop_logps, stop_logps)
        stop_logps = torch.stack(stack, dim=3)
        assert_shape(stop_logps, (B, T, self.b, 2))
        return stop_logps


class HeteroController(nn.Module):
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

    def forward(self, s_i_batch, batched=False):
        if batched:
            return self.forward_b(s_i_batch)
        else:
            return self.forward_ub(s_i_batch)

    def forward_ub(self, s_i):
        """
        s_i: (T, s) tensor of states
        outputs:
           (T, b, a) tensor of action logps
           (T, b, 2) tensor of stop logps
              the index corresponding to stop/continue are in
              box_world.STOP_IX (0), box_world.CONTINUE_IX (1)
           (T, b) tensor of start logps
           (T, T, b) tensor of causal consistency penalties
        """
        t_i = self.tau_net(s_i)  # (T, t)
        micro_out = self.micro_net(s_i)
        action_logps = rearrange(micro_out[:, :self.b * self.a], 'T (b a) -> T b a', b=self.b)
        stop_logps = rearrange(micro_out[:, self.b * self.a:], 'T (b two) -> T b two', b=self.b)

        start_logps = self.macro_policy_net(t_i)  # (T, b) aka P(b | t)
        causal_pens = self.calc_causal_pens_ub(t_i)  # (T, T, b)

        action_logps = F.log_softmax(action_logps, dim=2)
        stop_logps = F.log_softmax(stop_logps, dim=2)
        start_logps = F.log_softmax(start_logps, dim=1)

        return action_logps, stop_logps, start_logps, causal_pens

    def forward_b(self, s_i_batch):
        """
        s_i: (B, T, s) tensor of states
        outputs:
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

        causal_pens = self.calc_causal_pens_b(t_i)  # (B, T, T, b)

        action_logps = F.log_softmax(action_logps, dim=3)
        stop_logps = F.log_softmax(stop_logps, dim=3)
        start_logps = F.log_softmax(start_logps, dim=2)

        return action_logps, stop_logps, start_logps, causal_pens

    def macro_transition(self, t, b):
        """
        Calculate a single abstract transition. Useful for test-time.
        """
        return self.macro_transitions(t.unsqueeze(0),
                                      torch.tensor([b], device=DEVICE)).reshape(self.t)

    def tau_embed(self, s):
        """
        Calculate the embedding of a single state. Returns (t, ) tensor.
        """
        return self.tau_net(s.unsqueeze(0))[0]

    def macro_transitions(self, t_i, bs):
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

    def calc_causal_pens_ub(self, t_i):
        """For each pair of indices and each option, calculates (t - alpha(b, t))^2

        Args:
            t_i: (T, t) tensor of abstract states

        Returns:
            (T, T, b) tensor of penalties
        """
        T = t_i.shape[0]
        # apply each action at each timestep.
        macro_trans = self.macro_transitions(t_i, torch.arange(self.b, device=DEVICE))
        macro_trans1 = rearrange(macro_trans, 'T nb t -> T 1 nb t')
        t_i2 = rearrange(t_i, 'T t -> 1 T 1 t')
        # (start, end, action, t value)
        penalty = (t_i2 - macro_trans1)**2
        assert_equal(penalty.shape, (T, T, self.b, self.t))
        # L1 norm
        penalty = penalty.sum(dim=-1)  # (T, T, self.b)
        return penalty

    def calc_causal_pens_b(self, t_i_batch):
        """For each pair of indices and each option, calculates (t - alpha(b, t))^2

        Args:
            t_i_batch: (B, T, t) tensor of abstract states

        Returns:
            (B, T, T, b) tensor of penalties
        """
        B, T = t_i_batch.shape[0:2]

        t_i_flat = rearrange(t_i_batch, 'B T t -> (B T) t')
        macro_trans = self.macro_transitions(t_i_flat,
                                              torch.arange(self.b, device=DEVICE))
        macro_trans = rearrange(macro_trans, '(B T) b t -> B T 1 b t', B=B)

        t_i2 = rearrange(t_i_batch, 'B T t -> B 1 T 1 t')

        # (start, end, action, t value)
        penalty = (t_i2 - macro_trans)**2
        assert_shape(penalty, (B, T, T, self.b, self.t))
        penalty = penalty.sum(dim=-1)  # (B, T, T, b)
        return penalty

    def eval_obs(self, s_i):
        """
        For evaluation when we act for a single state.

        s_i: (*s, ) tensor

        Returns:
            (b, a) tensor of action logps
            (b, 2) tensor of stop logps
            (b, ) tensor of start logps
        """
        # (1, b, a), (1, b, 2), (1, b), (1, 1, b)
        action_logps, stop_logps, start_logps, _ = self.forward_ub(s_i.unsqueeze(0))
        return action_logps[0], stop_logps[0], start_logps[0]


class HomoController(nn.Module):
    """Controller as in microcontroller and macrocontroller.
    Homo because all of the outputs come from one big network.
    Really this one should be phased out if I can show just as good results from the Hetero one.
    Given the state, we give action logps, start logps, stop logps, etc.
    """

    def __init__(self, a, b, net):
        super().__init__()
        self.a = a
        self.b = b
        self.net = net

    def forward(self, s_i_batch, batched=True):
        if batched:
            return self.forward_b(s_i_batch)
        else:
            return self.forward_ub(s_i_batch)

    def forward_b(self, s_i_batch):
        """
        s_i: (B, T, s) tensor of states
        lengths: (B, ) tensor of lengths
        outputs:
            (B, T, b, a) tensor of action logps,
            (B, T, b, 2) tensor of stop logps,
            (B, T, b) tensor of start logps,
            None as causal pens placeholder
        """

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

        return action_logps, stop_logps, start_logps, None

    def forward_ub(self, s_i):
        """
        s_i: (T, s) tensor of states
        outputs:
            (T, b, a) tensor of action logps
            (T, b, 2) tensor of stop logps,
            (T, b) tensor of start logps
            None as causal pen placeholder
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

        return action_logps, stop_logps, start_logps, None

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
        action_logps, stop_logps, start_logps, _, = self.forward_ub(s_i.unsqueeze(0))
        return action_logps[0], stop_logps[0], start_logps[0]


def boxworld_relational_net(out_dim: int = 4):
    return RelationalDRLNet(input_channels=box_world.NUM_ASCII,
                            num_attn_blocks=2,
                            num_heads=4,
                            out_dim=out_dim)


def boxworld_homocontroller(b):
    # a * b for action probs, 2 * b for stop probs, b for start probs
    a = 4
    out_dim = a * b + 2 * b + b
    relational_net = RelationalDRLNet(input_channels=box_world.NUM_ASCII,
                                      num_attn_blocks=2,
                                      num_heads=4,
                                      out_dim=out_dim).to(DEVICE)
    control_net = HomoController(
        a=4,
        b=b,
        net=relational_net,
    )
    return control_net


def boxworld_controller(b, t=16, typ='hetero'):
    """
    type: hetero, homo, ccts1, or ccts2
    """
    a = 4
    assert typ in ['hetero', 'homo', 'ccts1', 'ccts2']
    if typ == 'homo':
        return boxworld_homocontroller(b)
    if typ == 'ccts1':
        micro_out_dim = a * b
        macro_trans_in_dim = b
        model = ConsistencyStopController1
    elif typ == 'ccts2':
        micro_out_dim = a * b
        macro_trans_in_dim = b
        model = ConsistencyStopController2
    else:
        assert typ == 'hetero'
        # both P(a | s, b) and beta(s, b)
        micro_out_dim = a * b + 2 * b
        macro_trans_in_dim = b + t
        model = HeteroController

    tau_net = boxworld_relational_net(out_dim=t)
    input_shape = (14, 14)  # assume default box world grid size
    micro_net = MicroNet(input_shape=input_shape,
                         input_channels=box_world.NUM_ASCII,
                         out_dim=micro_out_dim)
    macro_policy_net = FC(input_dim=t, output_dim=b, num_hidden=2, hidden_dim=32)
    macro_transition_net = FC(input_dim=macro_trans_in_dim, output_dim=t, num_hidden=2, hidden_dim=32)
    return model(a, b, t, tau_net, micro_net, macro_policy_net, macro_transition_net)
