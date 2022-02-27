import random
import numpy as np
import abstract
import torch
import torch.nn.functional as F
from torch import tensor
from torch.utils.data import DataLoader
import torch.nn as nn

from abstract2 import Controller, HMMTrajNet, TrajNet, UnbatchedTrajNet, cc_loss, cc_loss_brute
import box_world
from box_world import CONTINUE_IX, STOP_IX
from modules import RelationalDRLNet
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
    verbosity=hypothesis.Verbosity.verbose,
    # phases=[hypothesis.Phase.explicit],
    # phases=[hypothesis.Phase.explicit, hypothesis.Phase.reuse],
    # max_examples=50,
)
def test_brute_vs_hmm_cc_loss(args):
    b, action_logps, stop_logps, start_logps, causal_pens = args
    hmm_out = cc_loss(b, action_logps, stop_logps, start_logps, causal_pens)
    print(f'hmm_out: {hmm_out}')
    brute_out = cc_loss_brute(b, action_logps, stop_logps, start_logps, causal_pens)
    print(f'brute_out: {brute_out}')
    # print(f'o: : {torch.isclose(hmm_out, brute_out, atol=0.9)}')
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

    cc = cc_loss(b, action_logps, stop_logps, start_logps, causal_pens)
    cc2 = cc_loss_brute(b, action_logps, stop_logps, start_logps, causal_pens)
    np.testing.assert_approx_equal(cc2, cc, significant=6)


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
    cc = cc_loss(b, action_logps, stop_logps, start_logps, causal_pens)
    cc2 = cc_loss_brute(b, action_logps, stop_logps, start_logps, causal_pens)
    assert_equal(cc_target, cc)
    # print('first passed')
    assert_equal(cc_target, cc2)
    # print('passed cc test2')


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
    cc = cc_loss(b, action_logps, stop_logps, start_logps, causal_pens)
    cc2 = cc_loss_brute(b, action_logps, stop_logps, start_logps, causal_pens)
    assert_equal(cc_target, cc2)
    assert_equal(cc_target, cc)
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
    cc = cc_loss(b, action_logps, stop_logps, start_logps, causal_pens)
    cc2 = cc_loss_brute(b, action_logps, stop_logps, start_logps, causal_pens)
    assert_equal(cc_target, cc)
    assert_equal(cc_target, cc2)
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
    cc = cc_loss(b, action_logps, stop_logps, start_logps, causal_pens)
    cc2 = cc_loss_brute(b, action_logps, stop_logps, start_logps, causal_pens)
    assert_equal(cc_target, cc)
    # print('passed first')
    assert_equal(cc_target, cc2)
    # print('passed cc test5')


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

    cc = cc_loss(b, action_logps, stop_logps, start_logps, causal_pens)
    # print(f'cc: {cc}')
    cc2 = cc_loss_brute(b, action_logps, stop_logps, start_logps, causal_pens)
    # print(f'cc2: {cc2}')
    assert torch.isclose(cc, cc2, rtol=1E-5)


def test_forward():
    torch.manual_seed(1)
    import box_world
    from modules import RelationalDRLNet, abstract_out_dim
    from utils import Timing

    n = 5
    b = 10
    batch_size = 10

    relational_net = RelationalDRLNet(input_channels=box_world.NUM_ASCII,
                                      num_attn_blocks=2,
                                      num_heads=4,
                                      out_dim=abstract_out_dim(a=4, b=b)).to(DEVICE)
    control_net = Controller(
        a=4,
        b=b,
        net=relational_net,
        batched=True,
    )

    net = HMMTrajNet(control_net).to(DEVICE)

    env = box_world.BoxWorldEnv(seed=1)

    dataloader = box_world.box_world_dataloader(env=env, n=n, traj=True, batch_size=batch_size)
    for s_i_batch, actions_batch, lengths in dataloader:
        with Timing('loss1'):
            loss1 = net.logp(s_i_batch, actions_batch, lengths)
        with Timing('loss2'):
            loss2 = net.logp2(s_i_batch, actions_batch, lengths)
        # they should be equal
        print(f'loss1: {loss1}')
        print(f'loss2: {loss2}')


def batched_comparison():
    random.seed(0)
    torch.manual_seed(0)

    a = 4
    b = 1
    relational_net = RelationalDRLNet(input_channels=box_world.NUM_ASCII,
                                      num_attn_blocks=4,
                                      num_heads=4,
                                      out_dim=a * b + 2 * b + b).to(DEVICE)

    control_net = Controller(
        a=4,
        b=1,
        net=relational_net,
        batched=False,
    )
    unbatched_traj_net = UnbatchedTrajNet(control_net)

    control_net = Controller(
        a=4,
        b=1,
        net=relational_net,
        batched=True,
    )
    traj_net = TrajNet(control_net)

    env = box_world.BoxWorldEnv()
    dataloader = box_world.box_world_dataloader(env, n=3, traj=True, batch_size=2)
    data = box_world.BoxWorldDataset(env, n=3, traj=True)
    dataloader = DataLoader(data, batch_size=2, shuffle=False, collate_fn=box_world.traj_collate)

    total = 0
    for d in dataloader:
        negative_logp = traj_net(*d)
        total += negative_logp

    print(f'total0: {total}')

    total2 = 0
    for s_i, actions in zip(data.traj_states, data.traj_moves):
        negative_logp = unbatched_traj_net(s_i, actions)
        total2 += negative_logp

    print(f'total2: {total2}')

    data.traj = False
    dataloader = DataLoader(data, batch_size=1, shuffle=False)
    criterion = nn.CrossEntropyLoss(reduction='sum')
    total3 = 0
    for s_i, actions in dataloader:
        pred = relational_net(s_i)
        loss = criterion(pred[:, :4], actions)
        total3 += loss
    print(f'total3: {total3}')

    total4 = 0
    for s_i, actions in dataloader:
        pred = relational_net(s_i)
        logps = torch.log_softmax(pred[:, :4], dim=1)
        loss = -torch.sum(logps[range(len(actions)), actions])
        total4 += loss
    print(f'total4: {total4}')


def batched_comparison2():
    random.seed(1)
    torch.manual_seed(2)

    a = 4
    b = 10
    t = 50
    env = box_world.BoxWorldEnv()
    data = box_world.BoxWorldDataset(env, n=10, traj=True)
    dataloader = DataLoader(data, batch_size=1, shuffle=False, collate_fn=box_world.traj_collate)

    apn = abstract.attention_apn(b, t)
    t1, a1, s1, st1, cc1 = [], [], [], [], []

    for s_i_batch, actions_batch, lengths in dataloader:
        t_i, a, s, st, cc = apn(s_i_batch[0])
        t1.append(t_i)
        a1.append(a)
        s1.append(s)
        st1.append(st)
        cc1.append(cc)

    t1 = torch.cat(t1)
    a1 = torch.cat(a1)
    s1 = torch.cat(s1)
    st1 = torch.cat(st1)

    dataloader = DataLoader(data, batch_size=5, shuffle=False, collate_fn=box_world.traj_collate)
    t2, a2, s2, st2, cc2 = [], [], [], [], []

    for s_i_batch, actions_batch, lengths in dataloader:
        t_i, a, s, st, cc = apn.forward_batched(s_i_batch)
        for i, max_T in enumerate(lengths):
            print(max_T)
            t2.append(t_i[i, :max_T+1])
            a2.append(a[i, :max_T+1])
            s2.append(s[i, :max_T+1])
            st2.append(st[i, :max_T+1])
            cc2.append(cc[i, :max_T+1, :max_T+1])

    t2 = torch.cat(t2)
    a2 = torch.cat(a2)
    s2 = torch.cat(s2)
    st2 = torch.cat(st2)

    torch.testing.assert_allclose(t1, t2)
    torch.testing.assert_allclose(a1, a2)
    torch.testing.assert_allclose(s1, s2)
    torch.testing.assert_allclose(st1, st2)
    for c1, c2 in zip(cc1, cc2):
        torch.testing.assert_allclose(c1, c2)
    print('all good')


if __name__ == '__main__':
    pass
    # test_cc2()
    # test_cc3()
    # test_cc4()
    # test_cc5()
    # test_cc()
    # test_cc6()
