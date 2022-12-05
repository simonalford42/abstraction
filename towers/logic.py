from data_generator import TransitionDataset

from pysat.formula import IDPool
from pysat.solvers import Solver
from pysat.formula import CNF

class HanoiLogic():
    def __init__(self):
        self.reset()

    def reset(self):
        self._variables = IDPool()
        self._formula = CNF()
        self._model = None

    def constraint(self, c):
        self._formula.append(c)

    def exactly_one(self, vs):
        for u in vs:
            for v in vs:
                if u > v:
                    self.constraint([ -u, -v])
        self.constraint(vs)

    def implies(self, u, v):
        self.constraint([-u, v])

    def clear(self, k, x):
        i = self._variables.id(("clear",k,x))
        if self._model:
            return self._model[i]
        return i
    
    def on(self, k, x, y):
        i = self._variables.id(("on",k,x,y))
        if self._model:
            return self._model[i]
        return i

    def move(self, k, x, y, z):
        i = self._variables.id(("move",k,x,y,z))
        if self._model:
            return self._model[i]
        return i
    
    def valid_state(self, k):
        # every disk is on exactly one thing, and that thing must be bigger
        for x in range(3):
            self.exactly_one([ self.on(k, x, y) for y in range(x+1,6) ])
            for y in range(0,x):
                self.constraint([-self.on(k, x,y)])
            
        # not 2 disks on same thing
        for x1 in range(3):
            for x2 in range(3):
                if x1 < x2:
                    for y in range(6):                    
                        self.constraint([ - self.on(k, x1, y),
                                          - self.on(k, x2, y)])

        # If someone is on you, you are not clear
        for x in range(6):
            for y in range(3):
                self.constraint([-self.clear(k, x)] + [ -self.on(k, y, x) ])

    def valid_action(self, action, state):
        # one action is chosen
        self.exactly_one([ self.move(action, x, y, z)
                           for x in range(3)
                           for y in range(6)
                           for z in range(6)])
        
        for x in range(3):
            for y in range(6):
                for z in range(6):
                    if x == z or x == y or y == z:
                        self.constraint([-self.move(action, x, y, z)])
                    
                    # move(x,y,z) => on(x,y)
                    self.constraint([ - self.move(action, x, y, z),
                                      self.on(state, x, y) ])
                    
                    # move(x,y,z) => clear(x)
                    self.constraint([ - self.move(action, x, y, z),
                                      self.clear(state, x) ])
                    
                
                    # move(x,y,z) => clear(z)
                    self.constraint([ - self.move(action, x, y, z),
                                      self.clear(state, z) ])

    def valid_transition(self, k1, k2, action=None):
        if action is None: action=k1
        
        for x in range(3):
            for y in range(6):
                for u in range(6):
                    # move(x,u,y) -> on2(x,y)
                    self.implies(self.move(action, x,u,y), self.on(k2, x,y))
                    # move(x,u,y) -> ~on2(x,u)
                    self.implies(self.move(action, x,u,y), -self.on(k2, x,u))

                # on1(x,y) and not moving x -> on2(x,y)
                # [on1(x,y) and (~move(x,u,v), for all u, v)] -> on2(x,y)
                # ~[on1(x,y) and (~move(x,u,v), for all u, v)] or on2(x,y)
                # [~on1(x,y) or (move(x,u,v), for all u, v)] or on2(x,y)
                self.constraint([-self.on(k1, x,y), self.on(k2, x,y)] + \
                                [self.move(action, x,u,v) for v in range(6) for u in range(6)])

                                    
                                    
    def set_goal(self, k, is_goal, peg=3):
        atoms = [self.on(k, 0, 1), self.on(k, 1, 2), self.on(k, 2, peg)]
        if is_goal:
            for a in atoms: self.constraint([a])
        else:
            self.constraint([-a for a in atoms])

    def print_state(self, k):
        assert self._model
        
        for x in range(3):
            for y in range(6):
                if self.on(k, x, y):
                    print(f"on({x}, {y})")
                    
    def print_action(self, k):
        assert self._model
        for x in range(3):
            for y in range(6):
                for z in range(6):
                    if self.move(k, x, y, z):
                        print(f"move({x}, {y}, {z})")

    def solve(self):
        with Solver(bootstrap_with=self._formula) as solver:
            if solver.solve():
                self._model = {abs(v): v > 0 for v in solver.get_model()}
            else:
                print("impossible")



s = HanoiLogic()
for s1, a, s2 in TransitionDataset().legal_transitions:
    s1_name, s2_name = str(s1.rings), str(s2.rings)
    action_name = (str(a), s1_name)
    
    s.valid_state(s1_name)
    s.valid_state(s2_name)
    s.valid_action(action_name, s1_name)
    
    s.valid_transition(s1_name, s2_name,
                       action=action_name)
    for p in range(3):
        s.set_goal(s1_name, len(s1.rings[p]) == 3, peg=p+3)
        s.set_goal(s2_name, len(s2.rings[p]) == 3, peg=p+3)
    
s.solve()

for s1, a, s2 in TransitionDataset().legal_transitions:
    s1_name, s2_name = str(s1.rings), str(s2.rings)
    action_name = (str(a), s1_name)

    print("Actual state-action transition:")
    print(s1.rings)
    print(a)
    print(s2.rings)
    print("Solved as:")
    s.print_state(s1_name)
    s.print_action(action_name)
    s.print_state(s2_name)
    print()
# print()
# s.print_action(1)
# print()
# s.print_state(2)
# print()
# s.print_action(2)
# print()
# s.print_state(3)

