from environment import State

import random
import numpy as np

import torch



class RolloutDataset(torch.utils.data.IterableDataset):
    """
    Data set of rollouts. 
    Each rollout includes a list of states, actions, and per each state, whether that state is the goal.
    """
    def __init__(self, start=0, end=1280):
        super(TransitionDataset).__init__()
        assert end > start, "this example code only works with end >= start"
        self.size = end-start
        # precompute the set of all legal state transitions
        legal_states = [ State([[], [], []])]
        for new_disk in [3,2,1]:
            legal_states = [ State([ r if j != i else r+[new_disk]
                                     for j, r in enumerate(s.rings) ])
                             for s in legal_states for i in range(3) ]
        self.legal_states = legal_states
        self.transitions = {s: {a: (s.step(a),
                                    (s.render(), s.strips_action(a), s.step(a).render()))
                                for a in s.legal_actions() }
                            for s in legal_states }
        self.length = 4

    def rollout(self):
        s = random.choice(self.legal_states)
        states = [s.render()]
        goals = [s.is_goal]
        actions = []

        for _ in range(self.length):
            possible_actions = list(self.transitions[s].keys())
            a = random.choice(possible_actions)
            s, (_, action_vector, new_state_vector) = self.transitions[s][a]
            actions.append(action_vector)
            states.append(new_state_vector)
            goals.append(s.is_goal)

        return torch.tensor(np.stack(states)).float(), \
            torch.tensor(np.stack(actions)).float(), \
            torch.tensor(np.stack(goals)).float()
        
    def __iter__(self):
        for t in range(self.size):
            yield self.rollout()


class TransitionDataset(torch.utils.data.IterableDataset):
    """Data set of all possible mdp transitions. DEPRECATED, not used"""
    def __init__(self, start=0, end=1280):
        super(TransitionDataset).__init__()
        assert end > start, "this example code only works with end >= start"
        # precompute the set of all legal state transitions
        legal_states = [ State([[], [], []])]
        for new_disk in [3,2,1]:
            legal_states = [ State([ r if j != i else r+[new_disk]
                                     for j, r in enumerate(s.rings) ])
                             for s in legal_states for i in range(3) ]
        self.legal_transitions = [(s, a, s.step(a)) for s in legal_states for a in s.legal_actions() ] #[:3]

        states, actions, ons, clears, strips_actions = [], [], [], [], []
        next_states = []
        for s, a, sp in self.legal_transitions:
            states.append(s.render())
            next_states.append(sp.render())
            action_matrix = np.zeros((3,3))
            action_matrix[a[0],a[1]] = 1
            actions.append(action_matrix)

            predicates = s.predicates()
            ons.append(predicates[0])
            clears.append(predicates[1])

            strips_actions.append(s.strips_action(a))

            

        self.states, self.next_states, self.actions, self.ons, self.clears, self.strips = \
            torch.tensor(np.stack(states)).float(), \
            torch.tensor(np.stack(next_states)).float(), \
            torch.tensor(np.stack(actions)).float(), \
            torch.tensor(np.stack(ons)).float(), \
            torch.tensor(np.stack(clears)).float(), \
            torch.tensor(np.stack(strips_actions)).float()
        
        self.start = 0
        self.end = len(self.legal_transitions)

    def __iter__(self):
        for t in range(len(self.legal_transitions)):
            yield self.states[t], self.actions[t], self.next_states[t], self.ons[t], self.clears[t], self.strips[t]
