import box_world as bw
import math
import numpy as np
import random
import data
import utils
from pyDatalog import pyDatalog as pyd
from modules import RelationalDRLNet
from utils import assert_equal, DEVICE, assert_shape, log1minus
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import main
from einops import rearrange
import einops
import modules

STATE_EMBED_TRUE_IX = 0
STATE_EMBED_FALSE_IX = 1

NUM_PREDICATES = 2
PRECONDITION_LITERALS = 3
POSTCONDITION_LITERALS = 3

HELD_KEY_IX = 0
DOMINO_IX = 1

BW_WORLD_MODEL_PROGRAM = {'precondition':
                           [('held_key', 'X', False), ('domino', 'XY', False), ('action', 'Y', False)],
                          'effect':
                           [('held_key', 'Y', False), ('held_key', 'X', True), ('domino', 'XY', True)]}


def abstractify(obs):
    '''
    Returns dict of form {'predicate_str': [args]}
    '''
    dominoes = list(bw.get_dominoes(obs).keys())
    dominoes = [d.lower()[::-1] for d in dominoes]
    held_key = bw.get_held_key(obs)

    # the agent
    dominoes.remove('.')
    # sort dominoes by length, then alphabetically
    dominoes = sorted(dominoes, key=lambda s: (len(s), s))
    # if there's an open key, treat it as a held key
    if len(dominoes) > 0 and len(dominoes[0]) == 1:
        assert held_key is None
        held_key = dominoes[0]
        dominoes = dominoes[1:]

    # map each domino from e.g. 'AB' to ('A', 'B')
    dominoes = [tuple(domino) for domino in dominoes]

    if held_key is None:
        held_keys = []
    else:
        held_keys = [(held_key,)]

    out = {}
    if held_keys:
        out['held_key'] = held_keys
    if dominoes:
        out['domino'] = dominoes
    return out


def get_post_transition_facts():
    """
    returns dict of facts. key is the predicate and values are list of arg tuples
    """
    all_facts = {}
    # action facts don't get passed on
    for predicate in [held_key, domino, neg_held_key, neg_domino]:
        for arity in [1, 2]:
            try:
                args = [X, Y] if arity == 2 else [X]
                facts: list[tuple] = predicate(*args).ask()
                facts = sorted(facts)
                all_facts[str(predicate)] = facts
            except AttributeError:
                pass

    neg_pairs = [('neg_held_key', 'held_key'),
                  ('neg_domino', 'domino')]
    for neg_predicate, predicate in neg_pairs:
        if neg_predicate in all_facts:
            for args in all_facts[neg_predicate]:
                if args in all_facts[predicate]:
                    all_facts[predicate].remove(args)
                    if len(all_facts[predicate]) == 0:
                        del all_facts[predicate]
            del all_facts[neg_predicate]

    return all_facts


def transition_datalog(abs_obs: dict[str, list[tuple[str, str]]], abs_action: tuple[str]):
    '''
    abs_obs: dict of facts, {'predicate': [args]}
    abs_action: a tuple of args for the built-in 'action' predicate
    returns a new abs_obs.
    '''
    pyd.clear()
    # pyd.create_terms('X', 'Y', 'held_key', 'domino', 'action', 'neg_held_key', 'neg_domino')
    + action(*abs_action)
    if 'held_key' in abs_obs:
        for args in abs_obs['held_key']:
            + held_key(*args)
    if 'domino' in abs_obs:
        for args in abs_obs['domino']:
            + domino(*args)

    held_key(Y) <= held_key(X) & domino(X, Y) & action(Y)
    neg_held_key(X) <= held_key(X) & domino(X, Y) & action(Y)
    neg_domino(X, Y) <= held_key(X) & domino(X, Y) & action(Y)

    return get_post_transition_facts()


def check_datalog_consistency(states, moves):
    abs_states = [abstractify(state) for state in states]
    abs_moves = [(move[0].lower(), ) for move in moves]
    for i, (abs_state, abs_move) in enumerate(zip(abs_states, abs_moves)):
        abs_state2 = transition_datalog(abs_state, abs_move)
        assert_equal(abs_state2, abs_states[i+1])


def tensorize_symbolic_state(abs_state):
    '''
    Input: dict of facts, {'held_key': [args], 'domino': [args]}
        where args are tuples of the form (X, Y) and X, Y are color strings
    Output: 2 x C x C pytorch tensor A, where C is the number of colors
        and A[0, i, 0] is 1 iff the held key is i 1
        and A[1, i, j] is 1 iif (i, j) is a domino
        A[0, i, 1] is always 0
    '''
    colors = bw.COLORS
    A = torch.zeros(2, len(colors), len(colors))
    if 'held_key' in abs_state:
        for args in abs_state['held_key']:
            A[HELD_KEY_IX, colors.index(args[0]), 0] = 1
    if 'domino' in abs_state:
        for args in abs_state['domino']:
            A[DOMINO_IX, colors.index(args[0]), colors.index(args[1])] = 1
    return A
    # testing witih just predicting the held-key
    # return A[0:1]


def parse_symbolic_tensor(state):
    '''
    Input: 2 x C x C pytorch tensor A, where C is the number of colors
        and A[0, i, 0] is 1 iff the held key is i 1
        and A[1, i, j] is 1 iff (i, j) is a domino
        A[0, i, 1] is always 0
    Output: dict of facts, {'held_key': [args], 'domino': [args]}
        where args are tuples of the form (X, Y) and X, Y are color strings
    '''
    colors = bw.COLORS
    abs_state = {}
    if state[HELD_KEY_IX, :, 0].sum() > 0:
        abs_state['held_key'] = [(colors[i],) for i in range(len(colors)) if state[HELD_KEY_IX, i, 0]]
    if state[DOMINO_IX, :, :].sum() > 0:
        abs_state['domino'] = []
        for i in range(len(colors)):
            for j in range(len(colors)):
                if state[DOMINO_IX, i, j] == 1:
                    abs_state['domino'].append((colors[i], colors[j]))

        # sort domino args
        abs_state['domino'] = sorted(abs_state['domino'])
    return abs_state


def supervised_symbolic_state_abstraction_data(env, n) -> list[tuple]:
    '''
    Creates symbolic state abstraction data for n episodes of the environment.
    Returns a list of tuples of the form (symbolic_state, tensorized_embedding)
    '''
    trajs: list[list] = [bw.generate_abstract_traj(env) for _ in range(n)]

    # plural so we don't overwrite the data import
    datas = []
    for traj in trajs:
        states, moves = traj
        check_datalog_consistency(states, moves)
        state_tensors = [data.obs_to_tensor(state) for state in states]
        abs_states = [abstractify(state) for state in states]
        embed_states = [tensorize_symbolic_state(abs_state) for abs_state in abs_states]
        unembed_states = [parse_symbolic_tensor(embed_state) for embed_state in embed_states]
        for state, unembed in zip(abs_states, unembed_states):
            assert_equal(state, unembed)

        datas += list(zip(state_tensors, embed_states))

    return datas


class ListDataset(Dataset):
    '''
    Generic Dataset class for data stored in a list.
    '''
    def __init__(self, lst):
        self.lst = lst

    def __len__(self):
        return len(self.lst)

    def __getitem__(self, idx):
        return self.lst[idx]


class AbstractEmbedNet(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        out = self.net(x)
        C = bw.NUM_COLORS
        # out = out.reshape(out.shape[0], 1, C, C)
        # reshape to (batch_size, 2, C, C)
        out = out.reshape(out.shape[0], 2, C, C, 2)
        # out = F.log_softmax(out, dim=-1)
        out = F.softmax(out, dim=-1)
        # out = torch.sigmoid(out)
        return out


# seems like I have to do this outside of the function to get it to work?
pyd.create_terms('X', 'Y', 'held_key', 'domino', 'action', 'neg_held_key', 'neg_domino')


def world_model_step_prob(state_embeds, moves, world_model_program):
    '''
    Run the world model program on the state embeddings and return the new state prediction.

    state_embeds: [batch_size, 2, C, C, 2] tensor whose [:, P, A, B, STATE_EMBED_TRUE_IX] tells whether P(A, B) is true.
    likewise for STATE_EMBED_FALSE_IX.
    moves: [batch_size, ] tensor of colors [0, C-1w]
    world_model_program: a dict with keys 'precondition' and 'effect'.
        each are a length list of tuples (predicate, args, is_negated)
        where predicate is an index 0 or 1
              args is a string, either 'X', 'Y', or 'XY'
              is_negated is a boolean

    Returns: [batch_size, 2, C, C, 2] tensor of new state predictions.
    '''
    # example: held_key(Y), neg_held_key(X), neg_domino(X, Y) <= held_key(X) & domino(X, Y) & action(Y)
    P = state_embeds
    B, C = P.shape[0], P.shape[2]
    assert_shape(P, (B, 2, C, C, 2))
    # print(f'{moves=}')
    # print(f'{P[:, :, :, :, STATE_EMBED_TRUE_IX]=}')

    # pre args is either 'X', 'Y', or 'XY'
    precondition_probs = torch.ones(B, C, C)

    for predicate, args, is_negated in world_model_program['precondition']:
        if predicate == 'action':
            # if arg is 'X', then we can only allow P to be 1 when the first arg equals the action
            # if the arg is 'Y', then we can only allow P to be 1 when the second arg equals the action
            probs = F.one_hot(moves, num_classes=C)
            assert_shape(probs, (B, C))
        elif predicate == 'held_key':
            probs = P[:, HELD_KEY_IX, :, 0, STATE_EMBED_FALSE_IX if is_negated else STATE_EMBED_TRUE_IX]
        elif predicate == 'domino':
            probs = P[:, DOMINO_IX, :, :, STATE_EMBED_FALSE_IX if is_negated else STATE_EMBED_TRUE_IX]
        else:
            raise ValueError(f'unknown predicate {predicate}')

        if args == 'X':
            probs = rearrange(probs, 'B C -> B C 1')
        elif args == 'Y':
            probs = rearrange(probs, 'B C -> B 1 C')
        elif args == 'XY':
            pass
        else:
            raise ValueError('args must be X or Y')

        # print(f"{predicate=} {probs=}")
        # (B, C, C) tells probability that precondition satisfied
        # print(f'{predicate=}, {probs.nonzero()=}')
        precondition_probs *= probs
        assert_shape(precondition_probs, (B, C, C))

    # print(f"{precondition_probs=}, {precondition_probs[:, :, 2]=}")
    # print(f'{precondition_probs.nonzero()=}')
    # each of the effects gets assigned this probability of being true.
    # otherwise, by default, things stay the same.
    Q = torch.zeros(B, 2, C, C, 2)

    for predicate, args, is_negated in world_model_program['effect']:
        if predicate == 'held_key':
            assert args in ['X', 'Y']
            # sum precondition probs over axis not in args
            Q[:, HELD_KEY_IX, :, 0, STATE_EMBED_FALSE_IX if is_negated else STATE_EMBED_TRUE_IX] = precondition_probs.sum(dim=2 if args == 'X' else 1)
            # print(f"{predicate=}, {args=}, {is_negated=}, {Q[:, HELD_KEY_IX, :, 0, STATE_EMBED_FALSE_IX if is_negated else STATE_EMBED_TRUE_IX]=}")

        else:
            assert predicate == 'domino' and args == 'XY'
            Q[:, DOMINO_IX, :, :, STATE_EMBED_FALSE_IX if is_negated else STATE_EMBED_TRUE_IX] = precondition_probs

    Q = torch.clamp(Q, max=1)
    # print Q values for held_key(Y), neg_held_key(Y), domino(X, Y)
    # print(f'{Q[:, HELD_KEY_IX, :, 0, STATE_EMBED_TRUE_IX]=}')
    # print(f'{Q[:, HELD_KEY_IX, :, 0, STATE_EMBED_FALSE_IX]=}')
    # print(f'{Q[:, DOMINO_IX, :, 0, STATE_EMBED_TRUE_IX]=}')

    # print(f'{Q[:, :, :, :, STATE_EMBED_TRUE_IX].nonzero()=}')

    P0, P1 = P[:, :, :, :, STATE_EMBED_FALSE_IX], P[:, :, :, :, STATE_EMBED_TRUE_IX]
    # print(f"{P0=}, {P1=}")
    Q0, Q1 = Q[:, :, :, :, STATE_EMBED_FALSE_IX], Q[:, :, :, :, STATE_EMBED_TRUE_IX]
    # print(f"{Q0=}, {Q1=}")
    # new_P0 = P0 * (1 - Q1) + P1 * Q0
    new_P0 = P0 * (1 - Q1) + P1 * Q0
    # print(f"{new_P0=}")
    # new_P1 = P1 * (1 - Q0) + P0 * Q1
    new_P1 = P1 * (1 - Q0) + P0 * Q1
    # print(f"{new_P1=}")
    new_P1_2 = 1 - new_P0
    # print(f"{new_P1_2=}")
    # print(f"{new_P1_2-new_P1=}")
    # print(f'{new_P0.nonzero()=}')
    # print(f'{new_P1.nonzero()=}')
    # assert torch.allclose(log1minus(new_log_P0), new_log_P1)
    torch.testing.assert_allclose(1 - new_P0, new_P1)
    if STATE_EMBED_FALSE_IX == 0:
        new_P = torch.stack([new_P0, new_P1], dim=-1)
    else:
        new_P = torch.stack([new_P1, new_P0], dim=-1)

    assert_shape(new_P, (B, 2, C, C, 2))
    return new_P


def world_model_step(state_embeds, moves, world_model_program):
    '''
    Run the world model program on the state embeddings and return the new state prediction.

    state_embeds: [batch_size, 2, C, C, 2] tensor whose [:, P, A, B, STATE_EMBED_TRUE_IX] tells whether P(A, B) is true.
    likewise for STATE_EMBED_FALSE_IX.
    moves: [batch_size, ] tensor of colors [0, C-1]
    world_model_program: a dict with keys 'precondition' and 'effect'.
        each are a length list of tuples (predicate, args, is_negated)
        where predicate is an index 0 or 1
              args is a string, either 'X', 'Y', or 'XY'
              is_negated is a boolean

    Returns: [batch_size, 2, C, C, 2] tensor of new state predictions.
    '''
    # example: held_key(Y), neg_held_key(X), neg_domino(X, Y) <= held_key(X) & domino(X, Y) & action(Y)
    log_P = state_embeds
    B, C = log_P.shape[0], log_P.shape[2]
    assert_shape(log_P, (B, 2, C, C, 2))
    assert_shape(moves, (B, ))
    assert max(moves) <= C - 1 and min(moves) >= 0, f'moves must be in [0, {C-1}] but instead are {[min(moves), max(moves)]}'
    print(f'{log_P[:, :, :, :, STATE_EMBED_TRUE_IX]=}')

    # pre args is either 'X', 'Y', or 'XY'
    precondition_logps = torch.zeros(B, C, C)

    for predicate, args, is_negated in world_model_program['precondition']:
        if predicate == 'action':
            # if arg is 'X', then we can only allow P to be 1 when the first arg equals the action
            # if the arg is 'Y', then we can only allow P to be 1 when the second arg equals the action
            logps = torch.log(F.one_hot(moves, num_classes=C))
            assert_shape(logps, (B, C))
        elif predicate == 'held_key':
            logps = log_P[:, HELD_KEY_IX, :, 0, STATE_EMBED_FALSE_IX if is_negated else STATE_EMBED_TRUE_IX]
        elif predicate == 'domino':
            logps = log_P[:, DOMINO_IX, :, :, STATE_EMBED_FALSE_IX if is_negated else STATE_EMBED_TRUE_IX]
        else:
            raise ValueError(f'unknown predicate {predicate}')

        if args == 'X':
            logps = rearrange(logps, 'B C -> B C 1')
        elif args == 'Y':
            logps = rearrange(logps, 'B C -> B 1 C')
        elif args == 'XY':
            pass
        else:
            raise ValueError('args must be X or Y')

        print(f"{predicate=} {logps=}")
        # (B, C, C) tells probability that precondition satisfied
        # print(f'{predicate=}, {probs.nonzero()=}')
        precondition_logps += logps
        assert_shape(precondition_logps, (B, C, C))

    print(f"{precondition_logps=}, {precondition_logps[:, :, 2]=}")
    # print(f'{precondition_probs.nonzero()=}')
    # each of the effects gets assigned this probability of being true.
    # otherwise, by default, things stay the same.
    log_Q = torch.log(torch.zeros(B, 2, C, C, 2))

    for predicate, args, is_negated in world_model_program['effect']:
        if predicate == 'held_key':
            assert args in ['X', 'Y']
            # sum precondition probs over axis not in args
            log_Q[:, HELD_KEY_IX, :, 0, STATE_EMBED_FALSE_IX if is_negated else STATE_EMBED_TRUE_IX] = precondition_logps.logsumexp(dim=2 if args == 'X' else 1)
            print(f"{predicate=}, {args=}, {is_negated=}, {log_Q[:, HELD_KEY_IX, :, 0, STATE_EMBED_FALSE_IX if is_negated else STATE_EMBED_TRUE_IX]=}")

        else:
            assert predicate == 'domino' and args == 'XY'
            log_Q[:, DOMINO_IX, :, :, STATE_EMBED_FALSE_IX if is_negated else STATE_EMBED_TRUE_IX] = precondition_logps

    log_Q = torch.clamp(log_Q, max=0)
    # print Q values for held_key(Y), neg_held_key(Y), domino(X, Y)
    print(f'{log_Q[:, HELD_KEY_IX, :, 0, STATE_EMBED_TRUE_IX]=}')
    print(f'{log_Q[:, HELD_KEY_IX, :, 0, STATE_EMBED_FALSE_IX]=}')
    print(f'{log_Q[:, DOMINO_IX, :, 0, STATE_EMBED_TRUE_IX]=}')

    # print(f'{Q[:, :, :, :, STATE_EMBED_TRUE_IX].nonzero()=}')

    log_P0, log_P1 = log_P[:, :, :, :, STATE_EMBED_FALSE_IX], log_P[:, :, :, :, STATE_EMBED_TRUE_IX]
    print(f"{log_P0=}, {log_P1=}")
    log_Q0, log_Q1 = log_Q[:, :, :, :, STATE_EMBED_FALSE_IX], log_Q[:, :, :, :, STATE_EMBED_TRUE_IX]
    print(f"{log_Q0=}, {log_Q1=}")
    # new_P0 = P0 * (1 - Q1) + P1 * Q0
    new_log_P0 = torch.logaddexp(log_P0 + log1minus(log_Q1), log_P1 + log_Q0)
    print(f"{new_log_P0=}")
    # new_P1 = P1 * (1 - Q0) + P0 * Q1
    new_log_P1 = torch.logaddexp(log_P1 + log1minus(log_Q0), log_P0 + log_Q1)
    print(f"{new_log_P1=}")
    new_log_P1_2 = log1minus(new_log_P0)
    print(f"{new_log_P1_2=}")
    print(f"{new_log_P1_2-new_log_P1=}")
    # print(f'{new_P0.nonzero()=}')
    # print(f'{new_P1.nonzero()=}')
    # assert torch.allclose(log1minus(new_log_P0), new_log_P1)
    torch.testing.assert_allclose(log1minus(new_log_P0), new_log_P1)
    if STATE_EMBED_FALSE_IX == 0:
        new_log_P = torch.stack([new_log_P0, new_log_P1], dim=-1)
    else:
        new_log_P = torch.stack([new_log_P1, new_log_P0], dim=-1)

    assert_shape(new_log_P, (B, 2, C, C, 2))
    return new_log_P


def world_model_data(env, n) -> list[tuple]:
    '''
    Creates symbolic state abstraction data for n episodes of the environment.
    Returns a list of tuples of the form (abs_state, abs_action, abs_next_state)
    where abs_state and abs_next state are tensor embeddings of the symbolic state.
    '''
    trajs: list[list] = [bw.generate_abstract_traj(env) for _ in range(n)]

    # plural so we don't overwrite the data import
    datas = []
    for traj in trajs:
        states, moves = traj
        check_datalog_consistency(states, moves)
        state_tensors = [data.obs_to_tensor(state) for state in states]
        move_ixs = [bw.COLORS.index(m[0]) for m in moves]
        datas += list(zip(state_tensors[:-1], move_ixs, state_tensors[1:]))

    return datas


def option_sv_data(env, n) -> list[tuple]:
    trajs: list[list] = [bw.generate_abstract_traj(env) for _ in range(n)]

    datas = []
    for traj in trajs:
        states, moves = traj
        tensorized_symbolic_states = [tensorize_symbolic_state(abstractify(state)) for state in states]
        move_color_ixs = [bw.COLORS.index(domino[0]) for domino in moves]
        datas += list(zip(tensorized_symbolic_states[:-1], move_color_ixs))

    return datas


class SVOptionNet(nn.Module):
    def __init__(self, num_colors, num_options, hidden_dim, num_hidden=2):
        super().__init__()
        self.in_shape = (2, num_colors, num_colors)
        self.in_dim = np.prod(self.in_shape)
        self.num_options = num_options
        self.hidden_dim = hidden_dim
        self.fc = modules.FC(self.in_dim, self.num_options, num_hidden=num_hidden, hidden_dim=hidden_dim)

    def forward(self, x):
        assert_equal(x.shape[1:], self.in_shape)
        return self.fc(x.reshape(-1, self.in_dim))


class SVOptionNet2(nn.Module):
    def __init__(self, num_colors, num_options, num_heads=4, num_attn_blocks=2, hidden_dim=64):
        super().__init__()
        self.num_colors = num_colors
        self.num_options = num_options
        self.in_shape = (2, num_colors, num_colors)
        self.in_dim = np.prod(self.in_shape)
        # dimension of embedded predicate: one hot each color and the predicate choice
        d = self.num_colors * 2 + 2 + 1
        self.embed_dim = 2**(math.ceil(math.log(d, 2)))
        self.hidden_dim = hidden_dim
        self.out_dim = num_options

        self.attn_block = nn.MultiheadAttention(embed_dim=self.embed_dim,
                                                num_heads=num_heads,
                                                batch_first=True)
        self.num_attn_blocks = num_attn_blocks

        self.fc = nn.Sequential(nn.Linear(self.embed_dim, self.hidden_dim),
                                nn.ReLU(),
                                # nn.BatchNorm1d(self.d),
                                nn.Linear(self.hidden_dim, self.hidden_dim),
                                nn.ReLU(),
                                # nn.BatchNorm1d(self.d),
                                nn.Linear(self.hidden_dim, self.hidden_dim),
                                nn.ReLU(),
                                # nn.BatchNorm1d(self.d),
                                nn.Linear(self.hidden_dim, self.hidden_dim),
                                nn.ReLU(),
                                nn.Linear(self.hidden_dim, self.out_dim),)

        self.pcc_one_hot = self.make_pcc_one_hot(p=2, c=self.num_colors)

    def make_pcc_one_hot(self, p, c):
        vecs = []
        for i in range(p):
            for j in range(c):
                for k in range(c):
                    vec = torch.cat([F.one_hot(torch.tensor(i), num_classes=p),
                                     F.one_hot(torch.tensor(j), num_classes=c),
                                     F.one_hot(torch.tensor(k), num_classes=c)],
                                    dim=0)
                    assert_shape(vec, (p + c + c, ))
                    vecs.append(vec)

        pcc_one_hot = torch.stack(vecs)
        assert_shape(pcc_one_hot, (p * c * c, p + c + c))

        return pcc_one_hot.to(DEVICE)


    def forward(self, x):
        B = x.shape[0]
        assert_equal(x.shape[1:], self.in_shape)
        x = self.embed_predicates(x)
        assert_shape(x, (B, self.num_colors * self.num_colors * 2, self.embed_dim))

        for _ in range(self.num_attn_blocks):
            x = x + self.attn_block(x, x, x, need_weights=False)[0]
            x = F.layer_norm(x, (self.embed_dim,))

        x = einops.reduce(x, 'n l d -> n d', 'max')
        x = self.fc(x)
        return x

    def embed_predicates(self, x):
        B = x.shape[0]
        # (predicate_one_hot | color1_one_hot | color2_one_hot | value)
        assert_shape(x, (B, 2, self.num_colors, self.num_colors))
        x = rearrange(x, 'B p C1 C2 -> B (p C1 C2) 1')
        assert_shape(x, (B, self.num_colors * self.num_colors * 2, 1))
        pcc_one_hot_batch = einops.repeat(self.pcc_one_hot, 'pcc d -> B pcc d', B=B)
        x = torch.cat([pcc_one_hot_batch, x], dim=2)
        d = self.num_colors * 2 + 2 + 1
        assert_shape(x, (B, self.num_colors * self.num_colors * 2, d))
        x = torch.cat([x, torch.zeros((B, self.num_colors * self.num_colors * 2, self.embed_dim - d)).to(DEVICE)], dim=-1)

        return x


def test_world_model():
    env = bw.BoxWorldEnv()
    world_data = []
    trajs: list[list] = [bw.generate_abstract_traj(env) for _ in range(5000)]

    for traj in trajs:
        states, moves = traj
        tensorized_symbolic_states = [tensorize_symbolic_state(abstractify(state)) for state in states]

        world_data += list(zip(tensorized_symbolic_states[:-1], moves, tensorized_symbolic_states[1:]))
        # world_data += list(zip(states[:-1], moves, states[1:]))

    def test_world_model_data(world_data):
        for (state, move, next_state) in world_data:
            # embed it, run through world model, unembed, and compare to next_state
            # print(f'{state=}, {move=}, {abstract_state=}')
            # move[0] is the new held key, move[1] is the key to be unlocked by current key
            move_ix = bw.COLORS.index(move[0])
            move_tensor = torch.tensor(move_ix)
            # state = torch.stack([state, 1 - state], dim=-1).log()
            assert STATE_EMBED_TRUE_IX == 0
            state = torch.stack([state, 1 - state], dim=-1)

            next_state_pred = world_model_step_prob(state.unsqueeze(0),
                                                    move_tensor.unsqueeze(0),
                                                    world_model_program=BW_WORLD_MODEL_PROGRAM)
            assert torch.allclose(next_state, next_state_pred[0, :, :, :, STATE_EMBED_TRUE_IX])
            # next_state_pred = world_model_step(state.unsqueeze(0),
                                                #  move_tensor.unsqueeze(0),
                                                #  world_model_program=BW_WORLD_MODEL_PROGRAM)

            # assert torch.allclose(next_state, torch.exp(next_state_pred[0, :, :, :, STATE_EMBED_TRUE_IX]))

    test_world_model_data(world_data)

    # linear combinations of states
    states_with_move = {}
    for (state, move, next_state) in world_data:
        if move in states_with_move:
            states_with_move[move].append((state, next_state))
        else:
            states_with_move[move] = [(state, next_state)]

    mixed_states = []
    for move, states in states_with_move.items():
        if len(states) > 1:
            print(f'{move=}, {len(states)=}')
            for i in range(len(states)):
                # random pair of (state, next_state)'s
                (state1, next_state1), (state2, next_state2) = random.sample(states, 2)
                frac = random.choice([0.1, 0.2, 0.3, 0.4, 0.5])
                state = frac * state1 + (1 - frac) * state2
                next_state = frac * next_state1 + (1 - frac) * next_state2
                mixed_states.append((state, move, next_state))

    test_world_model_data(mixed_states)

    # also test all identity programs
    identity_programs = [
        {'precondition': [('held_key', 'X', False)], 'effect': [('held_key', 'X', False)]},
        {'precondition': [('held_key', 'Y', False)], 'effect': [('held_key', 'Y', False)]},
        {'precondition': [('domino', 'XY', False)], 'effect': [('domino', 'XY', False)]},
        {'precondition': [('action', 'X', False)], 'effect': []},
        {'precondition': [('action', 'Y', False)], 'effect': []},
    ]

    # for any state, identity program should give the same state
    for (state, move, next_state) in world_data:
        move_ix = bw.COLORS.index(move[0])
        move_tensor = torch.tensor(move_ix)
        assert STATE_EMBED_TRUE_IX == 0
        state = torch.stack([state, 1 - state], dim=-1)
        for program in identity_programs:
            next_state_pred = world_model_step_prob(state.unsqueeze(0),
                                                    move_tensor.unsqueeze(0),
                                                    world_model_program=program)
            torch.testing.assert_allclose(state[:, :, :, STATE_EMBED_TRUE_IX], next_state_pred[0, :, :, :, STATE_EMBED_TRUE_IX])


# test_world_model()

# supervised_symbolic_state_abstraction_data(bw.BoxWorldEnv(), n=100)
