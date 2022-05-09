import torch.nn as nn
import torch.nn.functional as F
import torch
import data
from einops import rearrange, repeat
import utils
from utils import assert_equal, assert_shape, DEVICE, logaddexp
from modules import MicroNet, RelationalDRLNet, FC, RelationalMacroNet
import box_world

TT = torch.Tensor

"""
Note: T used in Controller classes for length of sequence i.e. s_i.shape = (T, *s) etc. is not the same
T as that used in hmm.py, where the number of states is T+1 or max_T +1.
"""


def full_fine_tune_loss(t_i_batch, b_i_batch, control_net, masks=None, weights=None, loss='kl'):
    '''
    t_i_batch: (B, max_T + 1, t)
    b_i_batch: (B, max_T, )
    weights: weight loss from each item in batch
    loss: either 'kl', 'log', or 'kl-flip'
    '''

    def log_dist_loss(a, b):
        return -(a + b).exp()

    def kl_dist_loss(a, b):
        return F.kl_div(input=a, target=b, reduction='none', log_target=True)

    def kl_flip_dist_loss(a, b):
        return F.kl_div(input=b, target=a, reduction='none', log_target=True)

    if loss == 'kl':
        dist_loss_fn = kl_dist_loss
    elif loss == 'kl-flip':
        dist_loss_fn = kl_flip_dist_loss
    elif loss == 'log':
        dist_loss_fn = log_dist_loss
    else:
        raise ValueError("Bad arg for loss")

    def solved_and_start_logps(t_i_batch, control_net):
        (B, T, t) = t_i_batch.shape
        t_i_flattened = t_i_batch.reshape(B * T, t)
        solved = control_net.solved_net(t_i_flattened).reshape(B, T, 2)
        start_logps = control_net.macro_policy_net(t_i_flattened).reshape(B, T, control_net.b)
        return solved, start_logps

    (B, T) = b_i_batch.shape
    assert B > 1
    assert_shape(t_i_batch, (B, T + 1, control_net.t))
    if weights is not None:
        assert_shape(weights, (B, ))
    t_i_pred = t_i_batch[:, 0]
    t_i_preds = [t_i_pred]

    for i in range(T):
        b_i = b_i_batch[:, i]
        assert_shape(b_i, (B, ))
        t_i_pred = control_net.macro_transitions2(t_i_pred, b_i)
        t_i_preds.append(t_i_pred)

    t_i_pred_batch = torch.stack(t_i_preds, dim=1)
    assert_equal(t_i_pred_batch.shape, t_i_batch.shape)

    solved_pred_dist, macro_pred_dist = solved_and_start_logps(t_i_pred_batch, control_net)
    solved_target_dist, macro_target_dist = solved_and_start_logps(t_i_batch, control_net)
    loss_batch = (dist_loss_fn(solved_pred_dist, solved_target_dist).sum(dim=-1)
                  + dist_loss_fn(macro_pred_dist, macro_target_dist).sum(dim=-1))

    cc_loss_batch = (t_i_pred_batch - t_i_batch) ** 2
    cc_loss_batch = cc_loss_batch.sum(dim=-1)

    assert_shape(cc_loss_batch, (B, T + 1))
    assert_shape(loss_batch, (B, T + 1))

    loss_batch = loss_batch + cc_loss_batch

    if weights is not None:
        loss_batch = loss_batch * weights[:, None]
    if masks is not None:
        loss_batch = loss_batch * masks

    return loss_batch.sum()


def fine_tune_loss(t_i_batch, b_i_batch, control_net, weights=None):
    '''
    t_i_batch: (B, max_T + 1, t)
    b_i_batch: (B, max_T, )
    weights: weight loss from each item in batch
    '''
    (B, T) = b_i_batch.shape
    assert_shape(t_i_batch, (B, T + 1, control_net.t))
    if weights is not None:
        assert_shape(weights, (B, ))
    t_i_pred = t_i_batch[:, 0]
    t_i_preds = [t_i_pred]

    for i in range(b_i_batch.shape[1]):
        b_i = b_i_batch[:, i]
        assert_shape(b_i, (B, ))
        assert_equal(t_i_pred.shape[:-1], (B, ))
        t_i_pred = control_net.macro_transitions2(t_i_pred, b_i)
        t_i_preds.append(t_i_pred)

    t_i_pred_batch = torch.stack(t_i_preds, dim=1)
    assert_equal(t_i_pred_batch.shape, t_i_batch.shape)
    loss_batch = (t_i_pred_batch - t_i_batch) ** 2
    loss_batch = loss_batch.sum(dim=2)
    assert_shape(loss_batch, (B, T + 1))
    if weights is not None:
        loss_batch = loss_batch * weights[:, None]
    loss = loss_batch.sum()
    return loss


class ConsistencyStopControllerReduced(nn.Module):
    """
    alpha(b), beta(s), options take you to different states._
    (reduced compared to full abstract model)
    """
    def __init__(self, a, b, t, tau_net, micro_net, macro_policy_net,
                 macro_transition_net, solved_net, tau_lp_norm=1):
        super().__init__()
        self.a = a  # number of actions
        self.b = b  # number of options
        # self.s = s  # state dim; not actually used
        self.t = t  # abstract state dim
        self.tau_net = tau_net  # s -> t
        self.micro_net = micro_net  # s -> (b * a)
        self.macro_policy_net = macro_policy_net  # t -> b  aka P(b | t)
        self.macro_transition_net = macro_transition_net  # b_i one hot -> t abstract transition
        self.solved_net = solved_net  # t -> 2
        self.b_input = F.one_hot(torch.arange(self.b)).float().to(DEVICE)  # (b, b)
        self.tau_lp_norm = tau_lp_norm

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
              data.STOP_IX (0), data.CONTINUE_IX (1)
           (T, b) tensor of start logps
           None causal consistency placeholder
           solved: (T, 2)
        """
        # T = s_i.shape[0]
        t_i = self.tau_net(s_i)  # (T, t)
        # torch.testing.assert_close(torch.linalg.vector_norm(t_i, ord=self.tau_lp_norm, dim=1),
        #                            torch.ones(T, device=DEVICE))
        micro_out = self.micro_net(s_i)
        # assert_shape(micro_out, (T, self.b * self.a))
        action_logps = rearrange(micro_out, 'T (b a) -> T b a', b=self.b)
        stop_logps = self.calc_stop_logps_ub(t_i)  # (T, b, 2)
        # assert torch.allclose(torch.logsumexp(stop_logps, dim=2),
        #                       torch.zeros((T, self.b), device=DEVICE),
        #                       atol=1E-7), f'{stop_logps, torch.logsumexp(stop_logps, dim=2)}'
        start_logps = self.macro_policy_net(t_i)  # (T, b) aka P(b | t)

        action_logps = F.log_softmax(action_logps, dim=2)
        start_logps = F.log_softmax(start_logps, dim=1)
        solved = self.solved_net(t_i)
        return action_logps, stop_logps, start_logps, None, solved

    def calc_stop_logps_ub(self, t_i):
        # T = t_i.shape[0]
        goal_states = self.macro_transition_net(self.b_input)  # (b, t)
        # assert_shape(goal_states, (self.b, self.t))
        # assert_shape(t_i, (T, self.t))
        # (T, t, b) - (T, t, b), sum over t axis to get (T, b)
        stop_logps = -(rearrange(t_i, 'T t -> T t 1')
                       - rearrange(goal_states, 'b t -> 1 t b')) ** 2
        stop_logps = stop_logps.sum(dim=1)
        # assert_shape(stop_logps, (T, self.b))
        one_minus_stop_logps = logaddexp(torch.zeros_like(stop_logps),
                                         stop_logps,
                                         mask=torch.tensor([1, -1]))
        # assert_shape(one_minus_stop_logps, (T, self.b))
        if data.STOP_IX == 0:
            stack = (stop_logps, one_minus_stop_logps)
        else:
            stack = (one_minus_stop_logps, stop_logps)
        stop_logps = torch.stack(stack, dim=2)
        # assert_shape(stop_logps, (T, self.b, 2))
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
           solved: (B, T, 2)
        """
        B, T, *s = s_i_batch.shape
        s_i_flattened = s_i_batch.reshape(B * T, *s)

        t_i_flattened = self.tau_net(s_i_flattened)
        t_i = t_i_flattened.reshape(B, T, self.t)
        # torch.testing.assert_close(torch.linalg.vector_norm(t_i, ord=self.tau_lp_norm, dim=2),
        #                            torch.ones(B, T, device=DEVICE))

        micro_out = self.micro_net(s_i_flattened).reshape(B, T, -1)
        # assert_shape(micro_out, (B, T, self.b * self.a))
        action_logps = rearrange(micro_out, 'B T (b a) -> B T b a', a=self.a)
        stop_logps = self.calc_stop_logps_b(t_i)  # (B, T, b, 2)
        # assert torch.allclose(torch.logsumexp(stop_logps, dim=3),
        #                       torch.zeros((B, T, self.b), device=DEVICE),
        #                       atol=1E-7), f'{stop_logps, torch.logsumexp(stop_logps, dim=3)}'
        start_logps = self.macro_policy_net(t_i_flattened).reshape(B, T, self.b)
        solved = self.solved_net(t_i_flattened).reshape(B, T, 2)

        action_logps = F.log_softmax(action_logps, dim=3)
        start_logps = F.log_softmax(start_logps, dim=2)

        return action_logps, stop_logps, start_logps, None, solved

    def calc_stop_logps_b(self, t_i):
        B, T = t_i.shape[0:2]
        goal_states = self.macro_transition_net(self.b_input)  # (b, t)
        # assert_shape(goal_states, (self.b, self.t))
        # assert_shape(t_i, (B, T, self.t))
        # (B, T, t, b) - (B, T, t, b), sum over t axis to get (T, b)
        stop_logps = -(rearrange(t_i, 'B T t -> B T t 1')
                       - rearrange(goal_states, 'b t -> 1 1 t b')) ** 2
        stop_logps = stop_logps.sum(dim=2)
        # assert_shape(stop_logps, (B, T, self.b))
        one_minus_stop_logps = logaddexp(torch.zeros_like(stop_logps),
                                         stop_logps,
                                         mask=torch.tensor([1, -1]))
        # assert_shape(one_minus_stop_logps, (B, T, self.b))
        if data.STOP_IX == 0:
            stack = (stop_logps, one_minus_stop_logps)
        else:
            stack = (one_minus_stop_logps, stop_logps)
        stop_logps = torch.stack(stack, dim=3)
        # assert_shape(stop_logps, (B, T, self.b, 2))
        return stop_logps


class ConsistencyStopController(nn.Module):
    """
    alpha(b, tau(s)), beta(s_c, s_t). full abstract model, not markov
    """
    def __init__(self, a, b, t, tau_net, micro_net, macro_policy_net,
                 macro_transition_net, solved_net, tau_lp_norm, tau_noise_std):
        super().__init__()
        self.a = a  # number of actions
        self.b = b  # number of options
        # self.s = s  # state dim; not actually used
        self.t = t  # abstract state dim
        self.tau_net = tau_net  # s -> t
        self.micro_net = micro_net  # s -> (b * a)
        self.macro_policy_net = macro_policy_net  # t -> b  aka P(b | t)
        self.macro_transition_net = macro_transition_net  # (t + b) -> t abstract transition
        self.solved_net = solved_net  # t -> 2
        self.tau_lp_norm = tau_lp_norm
        self.tau_noise_std = tau_noise_std
        assert tau_noise_std == 0

    def freeze_microcontroller(self):
        self.micro_net.requires_grad_(False)

    def freeze_all_controllers(self):
        """
        Freezes all but the macro transition net and the solved net.
        """
        self.micro_net.requires_grad_(False)
        self.macro_policy_net.requires_grad_(False)
        self.tau_net.requires_grad_(False)

    def unfreeze(self):
        """
        Unfreezes micro_net, macro_policy_net, tau_net.
        """
        self.micro_net.requires_grad_(True)
        self.macro_policy_net.requires_grad_(True)
        self.tau_net.requires_grad_(True)

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
              data.STOP_IX (0), data.CONTINUE_IX (1)
           (T, b) tensor of start logps
           None causal consistency placeholder
           solved: (T, 2)
        """
        # T = s_i.shape[0]
        t_i = self.tau_net(s_i)  # (T, t)
        # torch.testing.assert_close(torch.linalg.vector_norm(t_i, ord=self.tau_lp_norm, dim=1),
        #                            torch.ones(T, device=DEVICE))
        action_logps = self.micro_net(s_i)
        # assert_shape(action_logps, (T, self.b, self.a))
        stop_logps = self.calc_stop_logps_ub(t_i)  # (T, T, b, 2)
        # assert torch.allclose(torch.logsumexp(stop_logps, dim=3),
        #                       torch.zeros((T, T, self.b), device=DEVICE),
        #                       atol=1E-6), f'{stop_logps, torch.logsumexp(stop_logps, dim=3)}'
        start_logps = self.macro_policy_net(t_i)  # (T, b) aka P(b | t)
        # assert_shape(start_logps, (T, self.b))
        solved = self.solved_net(t_i)
        return action_logps, stop_logps, start_logps, None, solved, t_i

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
        # b_onehots2 = repeat(b_onehots0, 'nb b -> (repeat nb) b', repeat=T)
        # assert torch.all(b_onehots == b_onehots2)
        # b is 'less significant', changes in 'inner loop'
        t_i2 = torch.cat((t_i2, b_onehots), dim=1)  # (T*nb, t + b)
        assert_equal(t_i2.shape, (T * nb, self.t + self.b))
        # (T * nb, t + b) -> (T * nb, t)
        t_i2 = self.macro_transition_net(t_i2)
        new_t_i = rearrange(t_i2, '(T nb) t -> T nb t', T=T) + t_i[:, None, :]
        return new_t_i

    def macro_transition(self, t, b):
        """
        Calculate a single abstract transition. Useful for test-time.
        Does not apply noise to abstract state.
        """
        return self.macro_transitions(t.unsqueeze(0),
                                      torch.tensor([b], device=DEVICE)).reshape(self.t)

    def macro_transitions2(self, t_i, bs):
        """
        Returns (T, self.t) batch of new abstract states.

        Args:
            t_i: (T, t) batch of abstract states
            bs: (T, ) tensor of actions for each abstract state
        """
        T = t_i.shape[0]
        b_onehots = F.one_hot(bs, num_classes=self.b)
        assert_shape(b_onehots, (T, self.b))
        t_i2 = torch.cat((t_i, b_onehots), dim=1)
        assert_shape(t_i2, (T, self.t + self.b))
        t_i2 = self.macro_transition_net(t_i2)
        assert_shape(t_i2, (T, self.t))
        return t_i2

    def calc_stop_logps_ub(self, t_i):
        # T = t_i.shape[0]
        alpha_out = self.macro_transitions(t_i, torch.arange(self.b, device=DEVICE))
        # assert_shape(alpha_out, (T, self.b, self.t))
        # assert_shape(t_i, (T, self.t))
        # (T, T, b, t) - (T, T, b, t), sum over t axis to get (T, T, b)
        stop_logps = -(rearrange(t_i, 'T t -> 1 T 1 t')
                       - rearrange(alpha_out, 'T b t -> T 1 b t')) ** 2
        stop_logps = stop_logps.sum(dim=3)
        # assert_shape(stop_logps, (T, T, self.b))
        one_minus_stop_logps = logaddexp(torch.zeros_like(stop_logps),
                                         stop_logps,
                                         mask=torch.tensor([1, -1]))
        # assert_shape(one_minus_stop_logps, (T, T, self.b))
        if data.STOP_IX == 0:
            stack = (stop_logps, one_minus_stop_logps)
        else:
            stack = (one_minus_stop_logps, stop_logps)
        stop_logps = torch.stack(stack, dim=3)
        # assert_shape(stop_logps, (T, T, self.b, 2))
        return stop_logps

    def forward_b(self, s_i_batch):
        """
        s_i: (B, T, s) tensor of states
        outputs:
           (B, T, b, a) tensor of action logps
           (B, T, T, b, 2) tensor of stop logps
              the index corresponding to stop/continue are in
              STOP_IX, CONTINUE_IX
           (B, T, b) tensor of start logps
           None CC placeholder
           solved: (B, T, 2)
        """
        B, T, *s = s_i_batch.shape
        s_i_flattened = s_i_batch.reshape(B * T, *s)

        t_i_flattened = self.tau_net(s_i_flattened)
        t_i = t_i_flattened.reshape(B, T, self.t)
        # torch.testing.assert_close(torch.linalg.vector_norm(t_i, ord=self.tau_lp_norm, dim=2),
        #                            torch.ones(B, T, device=DEVICE))

        action_logps = self.micro_net(s_i_flattened).reshape(B, T, self.b, self.a)
        stop_logps = self.calc_stop_logps_b(t_i)  # (B, T, T, b, 2)

        # assert torch.allclose(torch.logsumexp(stop_logps, dim=4),
        #                       torch.zeros((B, T, T, self.b), device=DEVICE),
        #                       atol=1E-6), \
        #        f'{stop_logps, torch.logsumexp(stop_logps, dim=4)}'

        start_logps = self.macro_policy_net(t_i_flattened).reshape(B, T, self.b)
        solved = self.solved_net(t_i_flattened).reshape(B, T, 2)

        return action_logps, stop_logps, start_logps, None, solved, t_i

    def calc_stop_logps_b(self, t_i):
        B, T = t_i.shape[0:2]

        t_i_flat = rearrange(t_i, 'B T t -> (B T) t')
        alpha_out = self.macro_transitions(t_i_flat, torch.arange(self.b, device=DEVICE))
        # assert_shape(alpha_out, (B * T, self.b, self.t))
        # assert_shape(t_i, (B, T, self.t))
        # (B, T, T, b, t) - (B, T, T, b, t), sum over t axis to get (B, T, T, b)
        stop_logps = -(rearrange(t_i, 'B T t -> B 1 T 1 t')
                       - rearrange(alpha_out, '(B T) b t -> B T 1 b t', B=B)) ** 2
        stop_logps = stop_logps.sum(dim=-1)
        # assert_shape(stop_logps, (B, T, T, self.b))
        one_minus_stop_logps = logaddexp(torch.zeros_like(stop_logps),
                                         stop_logps,
                                         mask=torch.tensor([1, -1]))
        # assert_shape(one_minus_stop_logps, (B, T, T, self.b))
        if data.STOP_IX == 0:
            stack = (stop_logps, one_minus_stop_logps)
        else:
            stack = (one_minus_stop_logps, stop_logps)
        stop_logps = torch.stack(stack, dim=4)
        # assert_shape(stop_logps, (B, T, T, self.b, 2))
        return stop_logps

    def tau_embed(self, s):
        """
        Calculate the embedding of a single state. Returns (t, ) tensor.
        Does not apply noise to abstract state.
        """
        return self.tau_net(s.unsqueeze(0))[0]

    def eval_obs(self, s_i, option_start_s):
        """
        For evaluation when we act for a single state.

        s_i: (*s, ) tensor

        Returns:
            (b, a) tensor of action logps
            (b, 2) tensor of stop logps
            (b, ) tensor of start logps
            (2, ) solved logits (solved is at abstract.SOLVED_IX)
        """
        # (2, b, a), (2, 2, b, 2), (2, b), (1, 1, b), (2, 2)
        action_logps, stop_logps, start_logps, _, solved_logits = self.forward_ub(torch.stack((option_start_s, s_i)))

        return action_logps[1], stop_logps[0, 1], start_logps[1], solved_logits[1]

    def eval_abstract_policy(self, t_i):
        """
        t_i: (t, ) tensor
        Returns:
            (b, ) tensor of logp for each abstract action
            (b, t) tensor of new tau for each abstract action
            (b, 2) tensor of logp new tau is solved
        """
        b, t = self.b, self.t
        start_logps: TT[b, ] = self.macro_policy_net(t_i.unsqueeze(0))[0]
        new_taus: TT[b, t] = self.macro_transitions(t_i.unsqueeze(0),
                                                    torch.arange(self.b, device=DEVICE))[0]
        # assert_shape(new_taus, (b, t))
        solveds = self.solved_net(new_taus)
        # assert_shape(solveds, (b, 2))
        return start_logps, new_taus, solveds

    def solved_logps(self, t_i):
        """
        t_i: (t, ) tensor
        Returns: (2, ) logps of probability solved/unsolved (use box_world.[UN]SOLVED_IX)
        """
        solved_logps = self.solved_net(t_i.unsqueeze(0))[0]
        return solved_logps

    def micro_policy(self, s_i, b, c_i):
        """
        s_i: single state
        c_i: time option b started
        outputs:
            (a,) action logps
            (2,) stop logps
        """
        raise NotImplementedError()


def noisify_tau(t_i, noise_std):
    # utils.warn('tau noise disabled')
    # return t_i
    return t_i + torch.normal(torch.zeros_like(t_i), torch.tensor(noise_std, device=DEVICE))


class HeteroController(nn.Module):
    def __init__(self, a, b, t, tau_net, micro_net, macro_policy_net,
                 macro_transition_net, solved_net, tau_lp_norm, tau_noise_std):
        super().__init__()
        self.a = a  # number of actions
        self.b = b  # number of options
        # self.s = s  # state dim; not actually used
        self.t = t  # abstract state dim
        self.tau_net = tau_net  # s -> t
        self.micro_net = micro_net  # s -> ((a, b), (2*b,))
        self.macro_policy_net = macro_policy_net  # t -> b aka P(b | t)
        self.macro_transition_net = macro_transition_net  # (t + b) -> t abstract transition
        self.tau_lp_norm = tau_lp_norm
        self.solved_net = solved_net  # t -> 2
        self.tau_noise_std = tau_noise_std

    def freeze_microcontroller(self):
        self.micro_net.requires_grad_(False)

    def unfreeze_microcontroller(self):
        self.micro_net.requires_grad_(True)

    def freeze_all_controllers(self):
        """
        Freezes all but the macro transition net and the solved net.
        """
        self.micro_net.requires_grad_(False)
        self.macro_policy_net.requires_grad_(False)
        self.tau_net.requires_grad_(False)

    def unfreeze(self):
        """
        Unfreezes micro_net, macro_policy_net, tau_net.
        """
        self.micro_net.requires_grad_(True)
        self.macro_policy_net.requires_grad_(True)
        self.tau_net.requires_grad_(True)

    def forward(self, s_i_batch, batched=True, tau_noise=True):
        if batched:
            return self.forward_b(s_i_batch, tau_noise=tau_noise)
        else:
            return self.forward_ub(s_i_batch, tau_noise=tau_noise)

    def forward_ub(self, s_i, tau_noise=True):
        """
        s_i: (T, s) tensor of states
        outputs:
           (T, b, a) tensor of action logps
           (T, b, 2) tensor of stop logps
              the index corresponding to stop/continue are in
              data.STOP_IX (0), data.CONTINUE_IX (1)
           (T, b) tensor of start logps
           (T, T, b) tensor of causal consistency penalties
           solved: (T, 2)
           (T, t) t_i abstract state embeddings
        """
        # T = s_i.shape[0]
        t_i = self.tau_net(s_i)  # (T, t)
        # torch.testing.assert_close(torch.linalg.vector_norm(t_i, ord=self.tau_lp_norm, dim=1),
        #                            torch.ones(T, device=DEVICE))

        noise = self.tau_noise_std if tau_noise else 0
        noised_t_i = noisify_tau(t_i, noise)

        action_logps, stop_logps = self.micro_net(s_i)
        start_logps = self.macro_policy_net(noised_t_i)  # (T, b) aka P(b | t)
        causal_pens = self.calc_causal_pens_ub(t_i, noised_t_i)  # (T, T, b)
        solved = self.solved_net(noised_t_i)

        return action_logps, stop_logps, start_logps, causal_pens, solved, t_i

    def forward_b(self, s_i_batch, tau_noise=True):
        """
        s_i: (B, T, s) tensor of states
        outputs:
           (B, T, b, a) tensor of action logps
           (B, T, b, 2) tensor of stop logps
              the index corresponding to stop/continue are in
              STOP_IX, CONTINUE_IX
           (B, T, b) tensor of start logps
           (B, T, T, b) tensor of causal consistency penalties
           (B, T, 2) solved
           (B, T, t) t_i abstract state embeddings
        """
        B, T, *s = s_i_batch.shape
        s_i_flattened = s_i_batch.reshape(B * T, *s)
        t_i_flattened = self.tau_net(s_i_flattened)

        noise = self.tau_noise_std if tau_noise else 0
        noised_t_i_flattened = noisify_tau(t_i_flattened, noise)

        t_i = t_i_flattened.reshape(B, T, self.t)
        # noised_t_i = noised_t_i_flattened.reshape(B, T, self.t)
        # torch.testing.assert_close(torch.linalg.vector_norm(t_i, ord=self.tau_lp_norm, dim=2),
        #                            torch.ones(B, T, device=DEVICE))

        action_logps, stop_logps = self.micro_net(s_i_flattened)
        action_logps = action_logps.reshape(B, T, self.b, self.a)
        stop_logps = stop_logps.reshape(B, T, self.b, 2)
        start_logps = self.macro_policy_net(noised_t_i_flattened).reshape(B, T, self.b)
        solved = self.solved_net(noised_t_i_flattened).reshape(B, T, 2)

        causal_pens = self.calc_causal_pens_b(t_i, noised_t_i_flattened)  # (B, T, T, b)

        return action_logps, stop_logps, start_logps, causal_pens, solved, t_i

    def macro_transition(self, t, b):
        """
        Calculate a single abstract transition. Useful for test-time.
        Does not apply noise to abstract state.
        """
        return self.macro_transitions(t.unsqueeze(0),
                                      torch.tensor([b], device=DEVICE)).reshape(self.t)

    def tau_embed(self, s):
        """
        Calculate the embedding of a single state. Returns (t, ) tensor.
        Does not apply noise to abstract state.
        """
        return self.tau_net(s.unsqueeze(0))[0]

    def macro_transitions(self, t_i, bs):
        """
        Returns (T, |bs|, self.t) batch of new abstract states for each option applied.
        Does not apply noise to abstract state.


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
        # b_onehots2 = repeat(b_onehots0, 'nb b -> (repeat nb) b', repeat=T)
        # assert torch.all(b_onehots == b_onehots2)
        # b is 'less significant', changes in 'inner loop'
        t_i2 = torch.cat((t_i2, b_onehots), dim=1)  # (T*nb, t + b)
        # assert_equal(t_i2.shape, (T * nb, self.t + self.b))
        # (T * nb, t + b) -> (T * nb, t)
        t_i2 = self.macro_transition_net(t_i2)
        # more stable for alpha to compute Delta, then add original t
        new_t_i = rearrange(t_i2, '(T nb) t -> T nb t', T=T) + t_i[:, None, :]
        return new_t_i

    def macro_transitions2(self, t_i, bs):
        """
        Returns (T, self.t) batch of new abstract states.

        Args:
            t_i: (T, t) batch of abstract states
            bs: (T, ) tensor of actions for each abstract state
        """
        T = t_i.shape[0]
        assert_shape(t_i, (T, self.t))
        assert_shape(bs, (T, ))
        b_onehots = F.one_hot(bs, num_classes=self.b)
        assert_shape(b_onehots, (T, self.b))
        t_i2 = torch.cat((t_i, b_onehots), dim=1)
        assert_shape(t_i2, (T, self.t + self.b))
        out = self.macro_transition_net(t_i2)
        assert_shape(out, (T, self.t))
        # residual
        out = out + t_i
        return out

    def calc_causal_pens_ub(self, t_i, noised_t_i):
        """
        For each pair of indices and each option, calculates (t - alpha(b, t))^2

        Args:
            t_i: (T, t) tensor of abstract states
            noised_t_i: (T, t) tensor of noisified abstract states

        Returns:
            (T, T, b) tensor of penalties
        """
        # T = t_i.shape[0]
        # apply each action at each timestep.
        macro_trans = self.macro_transitions(noised_t_i, torch.arange(self.b, device=DEVICE))
        macro_trans2 = rearrange(macro_trans, 'T nb t -> T 1 nb t')
        t_i2 = rearrange(t_i, 'T t -> 1 T 1 t')
        # (start, end, action, t value)
        penalty = (t_i2 - macro_trans2)**2
        # assert_equal(penalty.shape, (T, T, self.b, self.t))
        # L1 norm
        penalty = penalty.sum(dim=-1)  # (T, T, self.b)
        return penalty

    def calc_causal_pens_b(self, t_i_batch, noised_t_i_flattened):
        """For each pair of indices and each option, calculates (t - alpha(b, t))^2

        Args:
            t_i_batch: (B, T, t) tensor of abstract states
            noised_t_i_flattened: (B * T, t) tensor of abstract states

        Returns:
            (B, T, T, b) tensor of penalties
        """
        B, T = t_i_batch.shape[0:2]

        macro_trans = self.macro_transitions(noised_t_i_flattened,
                                             torch.arange(self.b, device=DEVICE))
        macro_trans2 = rearrange(macro_trans, '(B T) b t -> B T 1 b t', B=B)

        t_i2 = rearrange(t_i_batch, 'B T t -> B 1 T 1 t')

        # (start, end, action, t value)
        penalty = (t_i2 - macro_trans2)**2
        # assert_shape(penalty, (B, T, T, self.b, self.t))
        penalty = penalty.sum(dim=-1)  # (B, T, T, b)
        return penalty

    def eval_obs(self, s_i, option_start_s=None):
        """
        For evaluation when we act for a single state.

        s_i: (*s, ) tensor

        Returns:
            (b, a) tensor of action logps
            (b, 2) tensor of stop logps
            (b, ) tensor of start logps
            (2, ) solved logits (solved is at abstract.SOLVED_IX)
        """
        # (1, b, a), (1, b, 2), (1, b), (1, 1, b), (1, 2)
        action_logps, stop_logps, start_logps, _, solved_logits, _ = self.forward_ub(s_i.unsqueeze(0), tau_noise=False)

        return action_logps[0], stop_logps[0], start_logps[0], solved_logits[0]

    def eval_abstract_policy(self, t_i):
        """
        t_i: (t, ) tensor
        Returns:
            (b, ) tensor of logp for each abstract action
            (b, t) tensor of new tau for each abstract action
            (b, 2) tensor of logp new tau is solved
        """
        b, t = self.b, self.t
        start_logps: TT[b, ] = self.macro_policy_net(t_i.unsqueeze(0))[0]
        new_taus: TT[b, t] = self.macro_transitions(t_i.unsqueeze(0),
                                                    torch.arange(self.b, device=DEVICE))[0]
        # assert_shape(new_taus, (b, t))
        solveds = self.solved_net(new_taus)
        # assert_shape(solveds, (b, 2))
        return start_logps, new_taus, solveds

    def solved_logps(self, t_i):
        """
        t_i: (t, ) tensor
        Returns: (2, ) logps of probability solved/unsolved (use box_world.[UN]SOLVED_IX)
        """
        solved_logps = self.solved_net(t_i.unsqueeze(0))[0]
        return solved_logps

    def micro_policy(self, s_i, b):
        """
        s_i: single state
        outputs:
            (a,) action logps
            (2,) stop logps
        """
        action_logps, stop_logps = self.micro_net(s_i.unsqueeze(0))
        return action_logps[0, b], stop_logps[0, b]


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
            None as solveds placeholder
        """

        B, T, *s = s_i_batch.shape
        out = self.net(s_i_batch.reshape(B * T, *s)).reshape(B, T, -1)
        # assert_equal(out.shape[-1], self.a * self.b + 2 * self.b + self.b)
        action_logits = out[:, :, :self.b * self.a].reshape(B, T, self.b, self.a)
        stop_logits = out[:, :, self.b * self.a:self.b * self.a + 2 * self.b].reshape(B, T, self.b, 2)
        start_logits = out[:, :, self.b * self.a + 2 * self.b:]
        # assert_equal(start_logits.shape[-1], self.b)
        action_logps = F.log_softmax(action_logits, dim=3)
        stop_logps = F.log_softmax(stop_logits, dim=3)
        start_logps = F.log_softmax(start_logits, dim=2)

        return action_logps, stop_logps, start_logps, None, 'None', None

    def forward_ub(self, s_i):
        """
        s_i: (T, s) tensor of states
        outputs:
            (T, b, a) tensor of action logps
            (T, b, 2) tensor of stop logps,
            (T, b) tensor of start logps
            None as causal pen placeholder
            'None' as solveds placeholder
        """
        T = s_i.shape[0]
        out = self.net(s_i)
        # assert_equal(out.shape, (T, self.a * self.b + 2 * self.b + self.b))
        action_logits = out[:, :self.b * self.a].reshape(T, self.b, self.a)
        stop_logits = out[:, self.b * self.a:self.b * self.a + 2 * self.b].reshape(T, self.b, 2)
        start_logits = out[:, self.b * self.a + 2 * self.b:]
        # assert_equal(start_logits.shape[1], self.b)

        action_logps = F.log_softmax(action_logits, dim=2)
        stop_logps = F.log_softmax(stop_logits, dim=2)
        start_logps = F.log_softmax(start_logits, dim=1)

        return action_logps, stop_logps, start_logps, None, 'None', None

    def eval_obs(self, s_i, option_start_s=None):
        """
        For evaluation when we act for a single state.

        s_i: (s, ) tensor

        Returns:
            (b, a) tensor of action logps
            (b, 2) tensor of stop logps
            (b, ) tensor of start logps
            None solved placeholder
        """
        # (1, b, a), (1, b, 2), (1, b)
        action_logps, stop_logps, start_logps, _, _, _ = self.forward_ub(s_i.unsqueeze(0))
        return action_logps[0], stop_logps[0], start_logps[0], None


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
    control_net = HomoController(a=a, b=b, net=relational_net)
    return control_net


class ActionsMicroNet(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.micro_net = MicroNet(input_shape=box_world.DEFAULT_GRID_SIZE,
                                  input_channels=box_world.NUM_ASCII,
                                  out_dim=a * b)
        self.b = b

    def forward(self, x):
        x = self.micro_net(x)
        x = rearrange(x, 'B (b a) -> B b a', b=self.b)
        x = F.log_softmax(x, dim=-1)
        return x


class ActionsAndStopsMicroNet(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.micro_net = MicroNet(input_shape=box_world.DEFAULT_GRID_SIZE,
                                  input_channels=box_world.NUM_ASCII,
                                  out_dim=a * b + 2 * b)
        self.a = a
        self.b = b

    def forward(self, x):
        x = self.micro_net(x)
        action_logps = x[:, :self.a * self.b]
        stop_logps = x[:, self.a * self.b:]
        action_logps = rearrange(action_logps, 'B (b a) -> B b a', b=self.b)
        stop_logps = rearrange(stop_logps, 'B (b two) -> B b two', b=self.b)
        action_logps = F.log_softmax(action_logps, dim=-1)
        stop_logps = F.log_softmax(stop_logps, dim=-1)
        return action_logps, stop_logps


class NormModule(nn.Module):
    def __init__(self, p, dim):
        super().__init__()
        self.p = p
        self.dim = dim

    def forward(self, x):
        utils.warn('tau norm dim disabled')
        return F.normalize(x, dim=-1, p=self.p)  # * self.dim


def boxworld_controller(b, t=16, typ='hetero', tau_lp_norm=1, gumbel=False, tau_noise_std=0,
                        attentional_trans_net=False, batch_norm=False):
    """
    typ: hetero, homo, ccts, or ccts-reduced
    """
    a = 4
    tau_lp_norm = 1
    gumbel = False
    fc_hidden_dim = 64

    if gumbel:
        raise NotImplementedError()

    assert typ in ['hetero', 'homo', 'ccts', 'ccts-reduced']

    if typ == 'homo':
        return boxworld_homocontroller(b)

    if typ in ['ccts', 'ccts-reduced']:
        micro_net = ActionsMicroNet(a, b)
    else:
        micro_net = ActionsAndStopsMicroNet(a, b)

    if typ == 'ccts-reduced':
        macro_trans_in_dim = b
        model = ConsistencyStopControllerReduced
    elif typ == 'ccts':
        macro_trans_in_dim = b + t
        model = ConsistencyStopController
    else:
        assert typ == 'hetero'
        macro_trans_in_dim = b + t
        model = HeteroController

    tau_module = NormModule(p=tau_lp_norm, dim=t)
    tau_net = nn.Sequential(boxworld_relational_net(out_dim=t, ),
                            tau_module)

    macro_policy_net = nn.Sequential(FC(input_dim=t, output_dim=b, num_hidden=3,
                                        hidden_dim=fc_hidden_dim, batch_norm=batch_norm),
                                     nn.LogSoftmax(dim=-1))

    macro_transition_net = FC(input_dim=macro_trans_in_dim,
                              output_dim=t,
                              num_hidden=3,
                              hidden_dim=fc_hidden_dim,
                              batch_norm=batch_norm)
    macro_transition_net = nn.Sequential(macro_transition_net, tau_module)

    solved_net = nn.Sequential(FC(input_dim=t, output_dim=2, num_hidden=3,
                                  hidden_dim=fc_hidden_dim, batch_norm=batch_norm),
                               nn.LogSoftmax(dim=-1))

    return model(a=a, b=b, t=t,
                 tau_net=tau_net,
                 micro_net=micro_net,
                 macro_policy_net=macro_policy_net,
                 macro_transition_net=macro_transition_net,
                 solved_net=solved_net,
                 tau_lp_norm=tau_lp_norm,
                 tau_noise_std=tau_noise_std)
