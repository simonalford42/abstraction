import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import tensor
from abstract import boxworld_homocontroller

import abstract
import hmm
import box_world
from hmm import HmmNet, cc_loss, cc_loss_brute, cc_loss_ub, hmm_fw_ub
from box_world import CONTINUE_IX, STOP_IX
from utils import DEVICE, assert_equal, assert_shape

import hypothesis
from hypothesis import example, given, settings, strategies as st
from hypothesis.strategies import composite
from hypothesis.extra import numpy as np_st

# mlflow gives a lot of warnings from its dependencies, ignore them
import warnings
warnings.filterwarnings("module", category=DeprecationWarning)


@composite
def cc_input(draw, max_b, max_T):
    """
        action_logps (T, b)
        stop_logps (T+1, b, 2)
        start_logps (T+1, b)
        causal_pens (T+1, T+1, b)
    """
    b = draw(st.integers(min_value=1, max_value=max_b))
    T = draw(st.integers(min_value=1, max_value=max_T))
    logp_strategy = st.floats(min_value=-10, max_value=0, width=32)
    causal_pen_strategy = st.integers(min_value=0, max_value=10)
    action_logps = torch.tensor(draw(np_st.arrays(np.dtype(np.float32), (T, b), elements=logp_strategy)),
                                dtype=torch.float32)
    stop_logps = torch.tensor(draw(np_st.arrays(np.dtype(np.float32), (T+1, b, 2), elements=logp_strategy)),
                              dtype=torch.float32)
    start_logps = torch.tensor(draw(np_st.arrays(np.dtype(np.float32), (T+1, b), elements=logp_strategy)),
                               dtype=torch.float32)
    causal_pens = torch.tensor(draw(np_st.arrays(np.dtype(np.float32), (T+1, T+1, b), elements=causal_pen_strategy)),
                               dtype=torch.float32)

    # action_logps = F.log_softmax(action_logps, dim=2)
    stop_logps = F.log_softmax(stop_logps, dim=2)
    start_logps = F.log_softmax(start_logps, dim=1)

    return tuple([b, action_logps, stop_logps, start_logps, causal_pens])


@example(args=(1,
         tensor([[0.]]),  # action_logps
         tensor([[[0., 0.]],  # stop_logps
                 [[0., 0.]]]),
         tensor([[0.], [0.]]),  # start logps
         tensor([[[0.], [1.]],  # cc pens
                 [[0.], [0.]]])),)
@example(args=(1,
         tensor([[0.]]),  # action logps
         tensor([[[0., 0.]],  # stop logps
                 [[0., 0.]]]),
         tensor([[1.],  # start logps
                 [1.]]),
         tensor([[[0.], [0.]],  # cc
                 [[0.], [0.]]])),)
@example(args=(1,
         tensor([[0.], [0.]]),  # action logps
         tensor([[[0., 0.]],  # stop_logps
                 [[0., 0.]],
                 [[0., 0.]]]),
         tensor([[0.], [0.], [0.]]),  # start logps
         tensor([[[0.], [0.], [0.]],  # cc pen
                 [[0.], [0.], [0.]],
                 [[0.], [0.], [0.]]])),)
@example(args=(2,
         tensor([[0., 0.],
                 [0., 0.]]),
         tensor([[[0., 0.],
                 [0., 0.]],
                 [[0., 0.],
                 [0., 0.]],
                 [[0., 0.],
                 [0., 0.]]]),
         tensor([[0., 0.],
                 [0., 0.],
                 [0., 0.]]),
         tensor([[[1., 1.],
                 [1., 1.],
                 [1., 1.]],
                 [[1., 1.],
                 [1., 1.],
                 [1., 1.]],
                 [[1., 1.],
                 [1., 1.],
                 [1., 1.]]])),)
@example(args=(2,
         tensor([[0., 0.],
                 [0., 0.]]),
         tensor([[[0., 0.],
                  [0., 0.]],
                 [[0., 0.],
                  [0., 0.]],
                 [[0., 0.],
                  [0., 0.]]]),
         tensor([[0., 0.],
                 [0., 0.],
                 [0., 0.]]),
         tensor([[[0., 0.],
                  [0., 0.],
                  [1., 0.]],  # start at 0, stop at 3, b = 0.
                 [[0., 0.],
                  [0., 0.],
                  [0., 0.]],
                 [[0., 0.],
                  [0., 0.],
                  [0., 0.]]])),)
@example(args=(2,
         tensor([[0., 0.]]),
         tensor([[[0.0000, -3.4967],
                  [-2.7204, 0.0000]],
                 [[-2.1027, -0.5299],
                  [-4.1779, 0.0000]]]),
         tensor([[0.0000, -3.1113],
                 [0.0000, 0.0000]]),
         tensor([[[4., 7.],
                  [6., 6.]],
                 [[6., 4.],
                  [1., 10.]]])),)
@example(args=(1,  # b = 1
         tensor([[0.],  # T = 3; action logps
                 [0.],
                 [0.]]),
         tensor([[[0., 0.]],  # stop logps
                 [[0., 0.]],
                 [[0., 0.]],
                 [[0., 0.]]]),
         tensor([[0.0000],  # start logps
                 [-1.6095],  # force an indexing error for the probability?
                 [0.0000],
                 [0.0000]]),
         tensor([[[3.], [3.], [3.], [3.]],
                 [[3.], [3.], [3.], [0.]],  # start = 1, stop = end
                 [[3.], [3.], [3.], [3.]],
                 [[3.], [3.], [3.], [3.]]])),)
@example(args=(3,
         tensor([[0.0000, -10, -10],
                 [-10, -10, -10],
                 [-10, -10, -10],
                 [-10, -10, -10],
                 [-10, -10, -10]]),
         tensor([[[0., 0.], [0., 0.], [0., 0.]],
                 [[0., 0.], [0., 0.], [0., 0.]],
                 [[0., 0.], [0., 0.], [0., 0.]],
                 [[0., 0.], [0., 0.], [0., 0.]],
                 [[0., 0.], [0., 0.], [0., 0.]],
                 [[0., 0.], [0., 0.], [0., 0.]]]),
         tensor([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]),
         tensor([[[1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.]],
                 [[1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.]],
                 [[1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.]],
                 [[1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.]],
                 [[1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.]],
                 [[1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.]]])),)
@given(cc_input(max_b=3, max_T=5))
@settings(
    # verbosity=hypothesis.Verbosity.verbose,
    # phases=[hypothesis.Phase.explicit],
    # phases=[hypothesis.Phase.explicit, hypothesis.Phase.reuse],
    max_examples=200,
)
def test_brute_vs_hmm_cc_loss(args):
    b, action_logps, stop_logps, start_logps, causal_pens = args
    hmm_logp, hmm_out = cc_loss_ub(b, action_logps, stop_logps, start_logps, causal_pens)
    brute_logp, brute_out = cc_loss_brute(b, action_logps, stop_logps, start_logps, causal_pens)

    # action_logps = action_logps.unsqueeze(0)
    # stop_logps = stop_logps.unsqueeze(0)
    # start_logps = start_logps.unsqueeze(0)
    # causal_pens = causal_pens.unsqueeze(0)
    # batch_logp, batch_out = cc_loss(b, action_logps, stop_logps, start_logps, causal_pens)
    assert torch.isclose(hmm_logp, brute_logp, rtol=1E-4), f'hmm: {hmm_logp}, brute: {brute_logp}'
    assert torch.isclose(hmm_out, brute_out, rtol=1E-4), f'hmm: {hmm_out}, brute: {brute_out}'
    # assert torch.isclose(hmm_logp, batch_logp, rtol=5E-4), f'hmm: {hmm_logp}, brute: {batch_logp}'
    # assert torch.isclose(hmm_out, batch_out, rtol=5E-4), f'hmm: {hmm_out}, brute: {batch_out}'


@given(cc_input(max_b=10, max_T=70))
def test_cc_logp_vs_hmm_logp(args):
    b, action_logps, stop_logps, start_logps, causal_pens = args
    logp, cc = cc_loss_ub(b, action_logps, stop_logps, start_logps, causal_pens)
    hmm_logp = hmm_fw_ub(action_logps, stop_logps, start_logps)
    assert torch.isclose(logp, hmm_logp), f'cc: {logp}, hmm: {hmm_logp}'


def test_cc():
    # a = 2
    b = 2
    T = 3
    s_i = (0, 1, 1, 2)
    a_i = (1, 1, 1)

    def p_a_given_s_and_b(a, s, b):
        if a == s and b == 0:
            return 0.75
        elif a != s and b == 1:
            return 0.75
        else:
            return 0.25

    def p_b_given_s(b, s):
        return [[1, 1], [0, 0], [0, 0]][s][b]

    def p_stop(b, s):
        a = [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]
        return a[s][b]

    # (T, b)
    action_probs = torch.tensor([[p_a_given_s_and_b(a=a, s=s, b=b1)
                                  for b1 in range(b)]
                                  for a, s in zip(a_i, s_i[:-1])])
    start_probs = torch.tensor([[p_b_given_s(b=b1, s=s) for b1 in range(b)]
                                 for s in s_i])
    stop_probs = torch.tensor([[p_stop(b=b1, s=s) for b1 in range(b)]
                                 for s in s_i])
    stop_probs = stop_probs[:, :, None]
    if STOP_IX == 0:
        stop_probs = torch.cat((stop_probs, 1 - stop_probs), dim=2)
    else:
        stop_probs = torch.cat((1 - stop_probs, stop_probs), dim=2)
    # causal_pens = torch.arange((T+1)**2 * b).reshape(T+1, T+1, 2)
    causal_pens = torch.full((T+1, T+1, 2), 1)

    assert_shape(action_probs, (T, b))
    assert_shape(start_probs, (T+1, b))
    assert_shape(stop_probs, (T+1, b, 2))

    action_logps = torch.log(action_probs)
    start_logps = torch.log(start_probs)
    stop_logps = torch.log(stop_probs)

    logp, cc = cc_loss_ub(b, action_logps, stop_logps, start_logps, causal_pens)
    logp2, cc2 = cc_loss_brute(b, action_logps, stop_logps, start_logps, causal_pens)
    np.testing.assert_approx_equal(cc2, cc, significant=6)
    np.testing.assert_approx_equal(logp2, logp, significant=6)


def test_cc2():
    # a = 2
    b = 1
    T = 1
    s_i = (0, 0)
    a_i = (0, )

    def p_a_given_s_and_b(a, s, b):
        return 1

    def p_b_given_s(b, s):
        return 1

    def p_stop(b, s):
        return 1

    # (T, b)
    action_probs = torch.tensor([[p_a_given_s_and_b(a=a, s=s, b=b1)
                                  for b1 in range(b)]
                                  for a, s in zip(a_i, s_i[:-1])])
    start_probs = torch.tensor([[p_b_given_s(b=b1, s=s) for b1 in range(b)]
                                 for s in s_i])
    stop_probs = torch.tensor([[p_stop(b=b1, s=s) for b1 in range(b)]
                                 for s in s_i])
    stop_probs = stop_probs[:, :, None]

    if STOP_IX == 0:
        stop_probs = torch.cat((stop_probs, 1 - stop_probs), dim=2)
    else:
        stop_probs = torch.cat((1 - stop_probs, stop_probs), dim=2)
    causal_pens = torch.full((T+1, T+1, 1), 3)

    assert_shape(action_probs, (T, b))
    assert_shape(start_probs, (T+1, b))
    assert_shape(stop_probs, (T+1, b, 2))

    action_logps = torch.log(action_probs)
    start_logps = torch.log(start_probs)
    stop_logps = torch.log(stop_probs)

    # b_vec = 0
    p_start = 1
    p_stop_at_end = 1
    p_actions = 1
    p_0_0_0 = p_start * p_stop_at_end * p_actions
    cc_0_0_0 = 3

    cc_target = torch.tensor(p_0_0_0 * cc_0_0_0)
    logp, cc = cc_loss_ub(b, action_logps, stop_logps, start_logps, causal_pens)
    logp2, cc2 = cc_loss_brute(b, action_logps, stop_logps, start_logps, causal_pens)
    np.testing.assert_approx_equal(cc2, cc_target, significant=6)
    np.testing.assert_approx_equal(cc_target, cc, significant=6)
    np.testing.assert_approx_equal(logp2, logp, significant=6)


def test_cc3():
    # a = 2
    b = 1
    T = 5
    s_i = tuple([1 if i == T else 0 for i in range(T+1)])
    a_i = s_i[:-1]

    def p_a_given_s_and_b(a, s, b):
        return 1

    def p_b_given_s(b, s):
        return 1

    def p_stop(b, s):
        return 1 if s else 0

    # (T, b)
    action_probs = torch.tensor([[p_a_given_s_and_b(a=a, s=s, b=b1)
                                  for b1 in range(b)]
                                  for a, s in zip(a_i, s_i[:-1])])
    start_probs = torch.tensor([[p_b_given_s(b=b1, s=s) for b1 in range(b)]
                                 for s in s_i])
    stop_probs = torch.tensor([[p_stop(b=b1, s=s) for b1 in range(b)]
                                 for s in s_i])
    stop_probs = stop_probs[:, :, None]

    if STOP_IX == 0:
        stop_probs = torch.cat((stop_probs, 1 - stop_probs), dim=2)
    else:
        stop_probs = torch.cat((1 - stop_probs, stop_probs), dim=2)
    causal_pens = torch.full((T+1, T+1, 1), 3)

    assert_shape(action_probs, (T, b))
    assert_shape(start_probs, (T+1, b))
    assert_shape(stop_probs, (T+1, b, 2))

    action_logps = torch.log(action_probs)
    start_logps = torch.log(start_probs)
    stop_logps = torch.log(stop_probs)

    # b_vec = 0
    p_start = 1
    p_stop_at_end = 1
    p_actions = 1
    p_dont_stop = torch.prod(stop_probs[1:-1, 0, CONTINUE_IX])
    p_0_0_0 = p_start * p_stop_at_end * p_actions * p_dont_stop
    cc_0_0_0 = 3

    cc_target = (p_0_0_0 * cc_0_0_0)
    logp, cc = cc_loss_ub(b, action_logps, stop_logps, start_logps, causal_pens)
    logp2, cc2 = cc_loss_brute(b, action_logps, stop_logps, start_logps, causal_pens)
    np.testing.assert_approx_equal(cc2, cc_target, significant=6)
    np.testing.assert_approx_equal(cc2, cc_target, significant=6)
    np.testing.assert_approx_equal(logp2, logp, significant=6)
    # print('passed cc test3')


def test_cc4():
    # a = 2
    b = 2
    T = 2
    s_i = tuple([0 if i < T else 1 for i in range(T+1)])
    a_i = s_i[:-1]

    def p_a_given_s_and_b(a, s, b):
        return 1 if b else 0

    def p_b_given_s(b, s):
        return 1 if b else 0

    def p_stop(b, s):
        return 1 if s else 0

    # (T, b)
    action_probs = torch.tensor([[p_a_given_s_and_b(a=a, s=s, b=b1)
                                  for b1 in range(b)]
                                  for a, s in zip(a_i, s_i[:-1])])
    start_probs = torch.tensor([[p_b_given_s(b=b1, s=s) for b1 in range(b)]
                                 for s in s_i])
    stop_probs = torch.tensor([[p_stop(b=b1, s=s) for b1 in range(b)]
                                 for s in s_i])
    stop_probs = stop_probs[:, :, None]

    if STOP_IX == 0:
        stop_probs = torch.cat((stop_probs, 1 - stop_probs), dim=2)
    else:
        stop_probs = torch.cat((1 - stop_probs, stop_probs), dim=2)
    causal_pens = torch.full((T+1, T+1, b), 3)

    assert_shape(action_probs, (T, b))
    assert_shape(start_probs, (T+1, b))
    assert_shape(stop_probs, (T+1, b, 2))

    action_logps = torch.log(action_probs)
    start_logps = torch.log(start_probs)
    stop_logps = torch.log(stop_probs)

    # b_vec = 1
    p_start = 1
    p_stop_at_end = 1
    p_actions = 1
    p_dont_stop = torch.prod(stop_probs[1:-1, 1, CONTINUE_IX])
    p_0_0_0 = p_start * p_stop_at_end * p_actions * p_dont_stop
    cc_0_0_0 = 3

    cc_target = (p_0_0_0 * cc_0_0_0)
    logp, cc = cc_loss_ub(b, action_logps, stop_logps, start_logps, causal_pens)
    logp2, cc2 = cc_loss_brute(b, action_logps, stop_logps, start_logps, causal_pens)
    np.testing.assert_approx_equal(cc2, cc_target, significant=6)
    np.testing.assert_approx_equal(cc2, cc_target, significant=6)
    np.testing.assert_approx_equal(logp2, logp, significant=6)
    # print('passed cc test4')


def test_cc5():
    # a = 2
    b = 2
    T = 1
    s_i = [0, 0]
    a_i = [0, ]

    def p_a_given_s_and_b(a, s, b):
        return 1 if b else 0.5

    def p_b_given_s(b, s):
        return 0.75 if b else 0.25

    def p_stop(b, s):
        return 0.5 if b else 0.25

    # (T, b)
    action_probs = torch.tensor([[p_a_given_s_and_b(a=a, s=s, b=b1)
                                  for b1 in range(b)]
                                  for a, s in zip(a_i, s_i[:-1])])
    start_probs = torch.tensor([[p_b_given_s(b=b1, s=s) for b1 in range(b)]
                                 for s in s_i])
    stop_probs = torch.tensor([[p_stop(b=b1, s=s) for b1 in range(b)]
                                 for s in s_i])
    stop_probs = stop_probs[:, :, None]

    if STOP_IX == 0:
        stop_probs = torch.cat((stop_probs, 1 - stop_probs), dim=2)
    else:
        stop_probs = torch.cat((1 - stop_probs, stop_probs), dim=2)
    causal_pens = torch.full((T+1, T+1, b), 3)

    assert_shape(action_probs, (T, b))
    assert_shape(start_probs, (T+1, b))
    assert_shape(stop_probs, (T+1, b, 2))

    action_logps = torch.log(action_probs)
    start_logps = torch.log(start_probs)
    stop_logps = torch.log(stop_probs)

    # b_vec = 0
    p_start = start_probs[0, 0]
    p_stop_at_end = p_stop(b=0, s=0)
    p_actions = p_a_given_s_and_b(a=0, s=0, b=0)
    p_0 = p_start * p_stop_at_end * p_actions

    # b_vec = 1
    p_start = start_probs[0, 1]
    p_stop_at_end = p_stop(b=1, s=0)
    p_actions = p_a_given_s_and_b(a=0, s=0, b=1)
    p_1 = p_start * p_stop_at_end * p_actions

    cc_0 = (p_0 / (p_0 + p_1)) * 3
    cc_1 = (p_1 / (p_0 + p_1)) * 3

    cc_target = cc_0 + cc_1
    logp, cc = cc_loss_ub(b, action_logps, stop_logps, start_logps, causal_pens)
    logp2, cc2 = cc_loss_brute(b, action_logps, stop_logps, start_logps, causal_pens)
    np.testing.assert_approx_equal(cc2, cc_target, significant=6)
    np.testing.assert_approx_equal(cc2, cc_target, significant=6)
    np.testing.assert_approx_equal(logp2, logp, significant=6)


def test_cc6():
    # a = 2
    b = 3
    T = 5
    # (T, b)
    action_probs = torch.arange(b * T).reshape(T, b).to(torch.float32)
    action_probs /= b * T
    start_probs = torch.tensor([[[0.5, 0.3, 0.1][b] + 0.05 * t
                                 for b in range(b)]
                                for t in range(T+1)])
    stop_probs = torch.tensor([[[0.5, 0.3, 0.1][b] + 0.05 * t
                                 for b in range(b)]
                                for t in range(T+1)])
    stop_probs = stop_probs[:, :, None]

    if STOP_IX == 0:
        stop_probs = torch.cat((stop_probs, 1 - stop_probs), dim=2)
    else:
        stop_probs = torch.cat((1 - stop_probs, stop_probs), dim=2)
    causal_pens = torch.arange((T+1)*(T+1)*b).reshape((T+1, T+1, b), 3)

    assert_shape(action_probs, (T, b))
    assert_shape(start_probs, (T+1, b))
    assert_shape(stop_probs, (T+1, b, 2))

    action_logps = torch.log(action_probs)
    start_logps = torch.log(start_probs)
    stop_logps = torch.log(stop_probs)

    stop_logps = F.log_softmax(stop_logps, dim=2)
    start_logps = F.log_softmax(start_logps, dim=1)

    logp, cc = cc_loss_ub(b, action_logps, stop_logps, start_logps, causal_pens)
    logp2, cc2 = cc_loss_brute(b, action_logps, stop_logps, start_logps, causal_pens)
    np.testing.assert_approx_equal(cc2, cc, significant=6)
    np.testing.assert_approx_equal(logp2, logp, significant=6)


def test_hmm_and_cc():
    control_net = abstract.boxworld_controller(b=3)

    causal_net = hmm.CausalNet(control_net, cc_weight=0.)
    hmm_net = hmm.HmmNet(control_net)

    env = box_world.BoxWorldEnv()
    dataloader = box_world.box_world_dataloader(env=env, n=20, traj=True, batch_size=1)

    causal_net.to(DEVICE)
    hmm_net.to(DEVICE)

    for s_i, actions, lengths, masks in dataloader:
        s_i, actions, lengths = s_i.to(DEVICE), actions.to(DEVICE), lengths.to(DEVICE)
        causal_loss = causal_net(s_i, actions, lengths)
        hmm_loss = hmm_net(s_i, actions, lengths)
        assert torch.isclose(causal_loss, hmm_loss)


def test_hmm_batched():
    control_net = abstract.boxworld_homocontroller(b=3)
    net = hmm.HmmNet(control_net)

    env = box_world.BoxWorldEnv()
    dataloader = box_world.box_world_dataloader(env=env, n=100, traj=True, batch_size=10)

    net.to(DEVICE)
    optimizer = torch.optim.Adam(net.parameters(), lr=1E-4)

    for s_i, actions, lengths, _ in dataloader:
        optimizer.zero_grad()
        s_i, actions, lengths = s_i.to(DEVICE), actions.to(DEVICE), lengths.to(DEVICE)
        total_loss = net(s_i, actions, lengths)

        total_loss2 = 0
        for s_i, action, T in zip(s_i, actions, lengths):
            s_i = s_i[0:T+1]
            action = action[0:T]
            loss = net.logp_loss_ub(s_i, action)
            total_loss2 += loss

        assert torch.isclose(total_loss, total_loss2), f'{total_loss=}, {total_loss2=}'

        # loss = total_loss2
        loss = total_loss
        print(loss)
        loss.backward()
        optimizer.step()


def test_cc_batched():
    b = 3
    B = 5
    random.seed(1)
    torch.manual_seed(1)
    np.random.seed(1)

    control_net = abstract.boxworld_controller(b=b)

    net = hmm.CausalNet(control_net, cc_weight=1.)

    env = box_world.BoxWorldEnv(seed=1)
    dataloader = box_world.box_world_dataloader(env=env, n=50, traj=True, batch_size=B)

    net.to(DEVICE)

    optimizer = torch.optim.Adam(net.parameters(), lr=8E-4)

    for s_i, actions, lengths, masks in dataloader:
        optimizer.zero_grad()
        s_i, actions, lengths, masks = s_i.to(DEVICE), actions.to(DEVICE), lengths.to(DEVICE), masks.to(DEVICE)
        total_loss2 = 0
        for s, action, T in zip(s_i, actions, lengths):
            s = s[0:T+1]
            action = action[0:T]
            loss = net.cc_loss_ub(s, action)
            # print(f'ub loss: {loss}')
            total_loss2 += loss

        total_loss = net(s_i, actions, lengths, masks, batched=True)
        assert torch.isclose(total_loss, total_loss2), f'{total_loss=}, {total_loss2=}'

        # loss = total_loss2
        loss = total_loss
        print(loss)
        loss.backward()
        optimizer.step()


def test_actions_batch():
    b = 4
    B = 5
    control_net = abstract.boxworld_controller(b=b)

    env = box_world.BoxWorldEnv()
    dataloader = box_world.box_world_dataloader(env=env, n=10, traj=True, batch_size=B)

    control_net.to(DEVICE)

    for s_i, actions, lengths in dataloader:
        s_i, actions, lengths = s_i.to(DEVICE), actions.to(DEVICE), lengths.to(DEVICE)
        # (B, max_T+1, b, n)
        action_logps, stop_logps, start_logps, causal_pens = control_net(s_i, batched=True)

        T = action_logps.shape[1] - 1

        action_logps2 = []
        for t in range(T):
            # (B, b)
            action_logps2.append(action_logps[range(B), t, :, actions[:, t]])

        action_logps2 = torch.stack(action_logps2, dim=1)
        assert_shape(action_logps2, (B, T, b))

        action_logps = action_logps[torch.arange(B)[:, None],
                                    torch.arange(T)[None, :],
                                    :,
                                    actions[None, :]]
        assert_shape(action_logps[0], (B, T, b))
        assert torch.equal(action_logps2, action_logps[0])


def test_cc_batched2():
    b = 10
    B = 10
    control_net = abstract.boxworld_controller(b=b)

    env = box_world.BoxWorldEnv()
    dataloader = box_world.box_world_dataloader(env=env, n=50, traj=True, batch_size=B)

    control_net.to(DEVICE)

    for s_i, actions, lengths, masks in dataloader:
        s_i, actions, lengths, masks = s_i.to(DEVICE), actions.to(DEVICE), lengths.to(DEVICE), masks.to(DEVICE)
        # (B, max_T+1, b, n)
        action_logps, stop_logps, start_logps, causal_pens = control_net(s_i, batched=True)

        max_T = action_logps.shape[1] - 1

        action_logps = action_logps[torch.arange(B)[:, None],
                                    torch.arange(max_T)[None, :],
                                    :,
                                    actions[None, :]]
        # not sure why there's this extra singleton axis, but this passes the test so go for it
        action_logps = action_logps[0]

        fw_logps, total_logp_fw = hmm.cc_fw(b, action_logps, stop_logps, start_logps, lengths, masks)
        total_logp_fw = sum(total_logp_fw)
        bw_logps, total_logp_bw = hmm.cc_bw(b, action_logps, stop_logps, start_logps, lengths, masks)
        total_logp_bw = sum(total_logp_bw)
        total_logp, total_cc_loss = hmm.cc_loss(b, action_logps, stop_logps, start_logps, causal_pens, lengths, masks)

        total_logp_ub = 0
        total_cc_loss_ub = 0
        i = 0
        for s_i, action, T in zip(s_i, actions, lengths):
            s_i = s_i[0:T+1]
            action = action[0:T]
            action_logps, stop_logps, start_logps, causal_pens = control_net(s_i, batched=False)
            action_logps = action_logps[range(T), :, action]
            f2, logp_ub = hmm.cc_fw_ub(b, action_logps, stop_logps, start_logps)
            b2, logp_ub2 = hmm.cc_bw_ub(b, action_logps, stop_logps, start_logps)
            _, cc_loss_ub = hmm.cc_loss_ub(b, action_logps, stop_logps, start_logps, causal_pens)
            total_logp_ub += logp_ub
            total_cc_loss_ub += cc_loss_ub
            i += 1

        assert torch.isclose(total_logp_fw, total_logp_ub), f'{total_logp_fw=}, {total_logp_ub=}'
        assert torch.isclose(total_logp_bw, total_logp_ub), f'{total_logp_bw=}, {total_logp_ub=}'
        assert torch.isclose(total_cc_loss, total_cc_loss_ub), f'{total_cc_loss=}, {total_cc_loss_ub=}'


if __name__ == '__main__':
    # test_actions_batch()
    test_cc_batched()
    # test_hmm_batched()
    # test_hmm_and_cc()
    # test_cc_batched2()
    # test_cc_logp_vs_hmm_logp()
    # test_cc2()
    # test_cc3()
    # test_cc4()
    # test_cc5()
    # test_cc()
    # test_cc6()
