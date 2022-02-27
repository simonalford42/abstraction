import numpy as np
import torch
import torch.nn.functional as F
from torch import tensor

from hmm import HmmNet, cc_loss, cc_loss_brute
from box_world import CONTINUE_IX, STOP_IX
from utils import DEVICE, assert_equal, assert_shape

import hypothesis
from hypothesis import example, given, settings, strategies as st
from hypothesis.strategies import composite
from hypothesis.extra import numpy as np_st

# mlflow gives a shit ton of warnings from its dependencies
import warnings
warnings.filterwarnings("module", category=DeprecationWarning)


@composite
def cc_input(draw):
    """
        action_logps (T, b)
        stop_logps (T+1, b, 2)
        start_logps (T+1, b)
        causal_pens (T+1, T+1, b)
    """
    max_b = 3
    max_T = 5
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
@given(cc_input())
@settings(
    # verbosity=hypothesis.Verbosity.verbose,
    # phases=[hypothesis.Phase.explicit],
    # phases=[hypothesis.Phase.explicit, hypothesis.Phase.reuse],
    # max_examples=50,
)
def test_brute_vs_hmm_cc_loss(args):
    b, action_logps, stop_logps, start_logps, causal_pens = args
    hmm_logp, hmm_out = cc_loss(b, action_logps, stop_logps, start_logps, causal_pens)
    brute_logp, brute_out = cc_loss_brute(b, action_logps, stop_logps, start_logps, causal_pens)
    print(f'hmm_out: {hmm_out}')
    print(f'brute_out: {brute_out}')
    print(f'hmm_logp: {hmm_logp}')
    print(f'brute_logp: {brute_logp}')
    assert torch.isclose(hmm_logp, brute_logp, rtol=5E-4), f'hmm: {hmm_out}, brute: {brute_out}'
    assert torch.isclose(hmm_out, brute_out, rtol=5E-4), f'hmm: {hmm_out}, brute: {brute_out}'


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

    logp, cc = cc_loss(b, action_logps, stop_logps, start_logps, causal_pens)
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
    logp, cc = cc_loss(b, action_logps, stop_logps, start_logps, causal_pens)
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
    logp, cc = cc_loss(b, action_logps, stop_logps, start_logps, causal_pens)
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
    logp, cc = cc_loss(b, action_logps, stop_logps, start_logps, causal_pens)
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
    logp, cc = cc_loss(b, action_logps, stop_logps, start_logps, causal_pens)
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

    logp, cc = cc_loss(b, action_logps, stop_logps, start_logps, causal_pens)
    logp2, cc2 = cc_loss_brute(b, action_logps, stop_logps, start_logps, causal_pens)
    np.testing.assert_approx_equal(cc2, cc, significant=6)
    np.testing.assert_approx_equal(logp2, logp, significant=6)


def test_hmm_and_cc():
    import abstract
    import hmm
    import box_world
    control_net = abstract.boxworld_controller(b=3)

    net = hmm.CausalNet(control_net, cc_weight=0.)
    net2 = hmm.HmmNet(control_net)

    env = box_world.BoxWorldEnv()
    dataloader = box_world.box_world_dataloader(env=env, n=10, traj=True, batch_size=1)

    net.to(DEVICE)
    net2.to(DEVICE)

    for s_i, actions, lengths in dataloader:
        s_i, actions, lengths = s_i.to(DEVICE), actions.to(DEVICE), lengths.to(DEVICE)
        loss = net(s_i, actions, lengths)
        loss2 = net2(s_i, actions, lengths)
        assert torch.isclose(loss, loss2)


if __name__ == '__main__':
    test_hmm_and_cc()
    # test_cc2()
    # test_cc3()
    # test_cc4()
    # test_cc5()
    # test_cc()
    # test_cc6()
