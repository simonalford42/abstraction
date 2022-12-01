import random
import numpy as np

class State():
    "towers of Hanoi state"
    def __init__(self, rings):
        """
        rings:
        list of list of numbers, each list corresponds to a peg, an each number is the size of a ring on that peg
        for example, [[3,2,1],[],[]] is a reasonable initial state
        see .valid()
        """
        self.rings = rings

    def valid(self):
        return all( tuple(r) == tuple(sorted(r, reverse=True)) for r in self.rings)

    def __eq__(self, other):
        return self.rings == other.rings

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return hash(str(self.rings))

    @property
    def is_goal(self): return 3 == len(self.rings[0])

    def render(self):
        "convert state to image w/ one color for each different peg size"
        h = sum(len(r) for r in self.rings)
        w = len(self.rings)

        i = np.zeros((w,h,h))
        for ri, r in enumerate(self.rings):
            for h, c in enumerate(r):
                i[ri, h, c-1] = 1.
        assert self.rings ==  State.invert_render(i).rings
        return i

    def predicates(self):
        """
        returns STRIPS representation of the state in terms of predicates ON, CLEAR
        """
        on = np.zeros((len(self.rings)*2, len(self.rings) *2))
        clear = np.zeros((len(self.rings)*2))
        for r in range(len(self.rings)):
            if len(self.rings[r]) == 0:
                clear[r+3]=1
            else:
                on[self.rings[r][0]-1, r+3]=1
                for i in range(len(self.rings[r])-1):
                    on[self.rings[r][i+1]-1, self.rings[r][i]-1]=1
                clear[self.rings[r][-1]-1]=1
        return on, clear

    def strips_action(self, a):
        """
        returns STRIPS representation of action a in terms of predicates MOVE(x,y,z)
        a=(i,j) means that we are moving the top of peg i to the top of peg j
        """
        strips_action = np.zeros([len(self.rings)*2]*3)
        i,j = a
        x = self.rings[i][-1]-1
        if len(self.rings[j]) > 0:
            z = self.rings[j][-1]-1
        else:
            z = j+3
        if len(self.rings[i]) > 1:
            y = self.rings[i][-2]-1
        else:
            y = i+3
        assert np.sum(strips_action) < 1
        strips_action[x,y,z]=1

        assert np.sum(strips_action) == 1

        return strips_action

    @staticmethod
    def invert_render(i):
        rings = [[] for _ in range(i.shape[0]) ]
        for ri in range(i.shape[0]):
            for hi in range(i.shape[1]):
                for wi in range(i.shape[2]):
                    if i[ri,hi,wi] > 0.5:
                        rings[ri].append(wi+1)
        return State(rings)


    def step(self, action):
        """environment step"""
        assert self.valid()
        source_peg, destination_peg = action

        if len(self.rings[source_peg]) > 0:
            if len(self.rings[destination_peg]) == 0 or self.rings[source_peg][-1] < self.rings[destination_peg][-1]:
                new_rings = list(self.rings)
                new_rings[source_peg] = self.rings[source_peg][:-1]
                new_rings[destination_peg] = self.rings[destination_peg] + [self.rings[source_peg][-1]]
                s = State(new_rings)
                assert s.valid()
                return s

    def random_legal_action(self):
        return random.choice(self.legal_actions())

    def legal_actions(self):
        return [(i,j)
                for i in range(len(self.rings))
                for j in range(len(self.rings))
                if i != j and self.step((i,j)) is not None
        ]


            
def pretty_print_on_predicate(on):
    lines = []
    for x in range(on.shape[0]):
        for y in range(on.shape[1]):
            if on[x,y] > 0.5:
                a = f"disk_{x+1}" if x < 3 else f"peg_{x-2}"
                b = f"disk_{y+1}" if y < 3 else f"peg_{y-2}"
                lines.append(f"on({a}, {b})")
    return "\n".join(lines)
