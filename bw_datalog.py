import box_world as bw
import data
import utils
from pyDatalog import pyDatalog as pyd
from modules import RelationalDRLNet
from utils import assert_equal
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import main


def abstractify_datalog(obs):
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
                facts: List[Tuple] = predicate(*args).ask()
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
    abs_states = [abstractify_datalog(state) for state in states]
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
            A[0, colors.index(args[0]), 0] = 1
    if 'domino' in abs_state:
        for args in abs_state['domino']:
            A[1, colors.index(args[0]), colors.index(args[1])] = 1
    return A


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
    if state[0, :, 0].sum() > 0:
        abs_state['held_key'] = [(colors[i],) for i in range(len(colors)) if state[0, i, 0]]
    if state[1, :, :].sum() > 0:
        abs_state['domino'] = []
        for i in range(len(colors)):
            for j in range(len(colors)):
                if state[1, i, j] == 1:
                    abs_state['domino'].append((colors[i], colors[j]))

    # sort domino args
    abs_state['domino'] = sorted(abs_state['domino'])
    return abs_state


def supervised_symbolic_state_abstraction_data(env, n) -> list[tuple]:
    '''
    Creates symbolic state abstraction data for n episodes of the environment.
    Returns a list of tuples (state, tensorized_symbolic_state)
    '''
    trajs: list[list] = [bw.generate_abstract_traj(env) for _ in range(n)]

    # plural so we don't overwrite the data import
    datas = []
    for traj in trajs:
        states, moves = traj
        check_datalog_consistency(states, moves)
        state_tensors = [data.obs_to_tensor(state) for state in states]
        symbolic_states = [abstractify_datalog(state) for state in states]
        tensor_states = [tensorize_symbolic_state(symbolic_state) for symbolic_state in symbolic_states]
        datas.append((symbolic_states, tensor_states))
        parsed_states = [parse_symbolic_tensor(state) for state in symbolic_states]
        for symbolic_state, parsed_state in zip(symbolic_states, parsed_states):
            assert_equal(symbolic_state, parsed_state)

        # take as input the raw state, try to generate the tensor encoding of the symbolic state
        datas += list(zip(state_tensors, tensor_states))

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
        # reshape to (batch_size, 2, C, C)
        C = bw.NUM_COLORS
        out = out.reshape(out.shape[0], 2, C, C)
        out = torch.sigmoid(out)
        return out


# seems like I have to do this outside of the function to get it to work?
pyd.create_terms('X', 'Y', 'held_key', 'domino', 'action', 'neg_held_key', 'neg_domino')

n = 1
env = bw.BoxWorldEnv(solution_length=(4, ), num_forward=(4, ))
abs_data = ListDataset(abstract_sv_data(env, n=n))

dataloader = DataLoader(abs_data, batch_size=16, shuffle=True)

net = AbstractEmbedNet(RelationalDRLNet(input_channels=bw.NUM_ASCII, out_dim=2 * bw.NUM_COLORS * bw.NUM_COLORS))
print(utils.num_params(net), 'parameters')
main.sv_train2(dataloader, net=net, epochs=10, lr=3e-4, save_every=None, print_every=1)
