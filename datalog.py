from typing import Dict, List, Tuple
import pyDatalog as pyd

# seems like I have to do this outside of the function to get it to work?
pyd.create_terms('X', 'Y', 'held_key', 'domino', 'action', 'neg_held_key', 'neg_domino')


def transition_datalog(abs_obs: Dict[str, List[Tuple[str, str]]], abs_action: Tuple[str]):
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


def transition_datalog(abs_obs: Dict[str, List[Tuple[str, str]]], abs_action: Tuple[str]):
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

