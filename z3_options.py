import itertools
import random
from z3 import *
import os
import matplotlib.pyplot as plt
import numpy as np

import argparse
parser = argparse.ArgumentParser(description="")
parser.add_argument("--states", "-t", default=1, type=int, help="maximum number of abstract states")
parser.add_argument("--options", "-o", default=3, type=int, help="maximum number of options considered")
parser.add_argument("--size", "-s", default=4, type=int, help="dimensions of grid world")
parser.add_argument("--landmarks", "-l", default=3, type=int, help="number of landmarks")
parser.add_argument("--unique", "-u", default=False, action="store_true") # how to get landmarks:
parser.add_argument("--consistent", "-c", default=False, action="store_true", help="force causal consistency") # enable causal consistency
parser.add_argument("--predict_stop", "-p", default=False, action="store_true", help="stop condition is controlled by causal consistency, stop iff causal consistency is attained") # stop condition is controlled by causal consistency
parser.add_argument("--change", "-x", default=False, action="store_true", help="force the abstract state to change between options") # abstract actions have to change the abstract state
parser.add_argument("--condition_initial", "-i", default=False, action="store_true", help="policy is conditioned on initial abstract state") # policy is conditioned on initial abstract state""arguments = parser.parse_args()

arguments = parser.parse_args()

if arguments.predict_stop:
    assert arguments.consistent
if arguments.change:
    assert arguments.consistent
if arguments.condition_initial:
    assert arguments.consistent

N=arguments.size # size of grid world
ABSTRACTSTATES=arguments.states # number of abstract states
ABSTRACTACTIONS=arguments.options # number of abstract actions

argument_names = [a for a in dir(arguments) if not a.startswith("_") ]
argument_names.sort()



filename_prefix="_".join(f"{name}={getattr(arguments, name)}" for name in argument_names)+"/"
os.system(f"mkdir {filename_prefix}")

global_plot_title="\n".join(f"{name}={int(getattr(arguments, name))}"
                           for name in argument_names if name not in ["size", "landmarks"])

solver=Solver()
def constrain(x):
    solver.add(x)



def extract_real(m,r):
    if m[r] == None:
        print(r)
    return float(m[r].as_decimal(3).replace('?',''))

def maximum(a, b):
    return If(a>b, a, b)

def minimize(c):
    global_cost_function = Real("global_cost_function")
    constrain(global_cost_function == c)
    m = None
    up = None
    while True:
        if str(solver.check()) == 'sat':
            m = solver.model()
        else:
            break
        up = extract_real(m,global_cost_function)
        print("best cost so far", up)

        constrain(global_cost_function<up)

    print("Minimum cost found", up)
    return m, up


def exactly_one(v):
    for i, x in enumerate(v):
        for j, y in enumerate(v):
            if i>j:
                constrain(Not(And(x, y)))
    constrain(Or(*v))

def exactly_k(variables, k):

    return PbEq([(v, 1) for v in variables], k)


def mutually_exclusive(m):
    v=[z3.FreshBool() for _ in range(m) ]
    exactly_one(v)
    return v

# def make_state(n):
#     global solver
#     new_state = [[z3.FreshBool() for _ in range(n) ]
#                  for _ in range(n) ]
#     for i in range(n):
#         for j in range(n):
#             for ii in range(n):
#                 for jj in range(n):
#                     if (i, j)>(ii, jj):
#                         constrain(Not(And(new_state[ii][jj], new_state[i][j])))
#     constrain(Or(*[new_state[i][j]
#                  for i in range(n)
#                 for j in range(n)]))
#     return new_state

# def make_action():
#     global solver
#     a = [z3.FreshBool() for _ in range(4) ]
#     exactly_one(a)
#     return a



def make_model():
    n=N

    states=[(i, j)
            for i in range(n) for j in range(n) ]
    dx={0: 0,
        1: 0,
        2: 1,
        3: -1}
    dy={0: 1,
        1: -1,
        2: 0,
        3: 0}

    def clamp(z):
        return max(0, min(z, n-1))

    t={((i, j), a):
       (clamp(i+dx[a]), clamp(j+dy[a]))
       for i in range(n) for j in range(n)
       for a in range(4) }

    def transition(state, action):
        nonlocal t
        return t[(state, action)]

    return states, transition

def print_state(s):
    return "\n".join("".join( "X" if s==(x, y) else "."
                              for x in range(N) )
                     for y in range(N) )
def print_action(a):
    if a is None: return str(a)
    return ["+y", "-y", "+x", "-x"][a]


def make_landmark_trajectory(s0, landmarks):
    states=[]
    actions=[]
    while len(landmarks):
        x, y = s0
        xp, yp=landmarks[0]

        if s0==landmarks[0]:
            landmarks=landmarks[1:]
            continue

        if x==xp:
            if y<yp:
                a=0
            else:
                a=1
        elif y == yp:
            if x<xp:
                a=2
            else:
                a=3
        else:
            if (x+y)%2==0:
                if x<xp:
                    a=2
                else:
                    a=3
            else:
                if y<yp:
                    a=0
                else:
                    a=1

        actions.append(a)
        states.append(s0)
        s0=make_model()[-1](s0, a)
    states.append(s0)
    actions.append(None)
    return states, actions

def visualize_trajectory(states, actions, folder, options=None):
    for i, (s, a) in enumerate(zip(states, actions)):
        data=np.zeros((N, N))
        data[s[1], s[0]]=1
        plt.figure()
        plt.imshow(data)
        if options:
            plt.title(print_action(a)+f"  option={options[i]}")
        else:
            plt.title(print_action(a))

        plt.savefig(f"{folder}/{i}.png")
        plt.close()

def visualize_trajectories(trajectories, option_trajectories, folder):
    plt.figure(figsize=(30, 30))
    nr=len(trajectories)
    nc=max(len(tr[0]) for tr in trajectories )


    for j, ((states, actions), option_trajectory) in enumerate(zip(trajectories,
                                                                   option_trajectories)):
        options, ending = option_trajectory

        for i, (s, a) in enumerate(zip(states, actions)):
            plt.subplot(nr, nc, 1+j*nc+i)
            plt.gca().set_yticklabels([])
            plt.gca().set_xticklabels([])
            data=np.zeros((N, N))
            data[s[1], s[0]]=1



            plt.imshow(data)

            if ending[i]:
                plt.title(print_action(a)+f", START option={options[i]}")
            else:
                plt.title(print_action(a)+f", option={options[i]}")

    global best_cost, global_plot_title
    plt.suptitle(f"{global_plot_title}\nglobal cost={best_cost}", fontsize=14)
    plt.savefig(f"{folder}.png")
    #plt.show()
    plt.close()




_state_abstraction={}
def state_abstraction(s):
    global _state_abstraction
    if s not in _state_abstraction:
        _state_abstraction[s] = mutually_exclusive(ABSTRACTSTATES)

    return _state_abstraction[s]
def visualize_states(m, filename):
    global _state_abstraction
    data=np.zeros((N, N))
    for x in range(N):
        for y in range(N):
            if (x, y) not in _state_abstraction:
                data[y, x]=-1
            else:
                data[y, x]= [m[i] for i in _state_abstraction[(x, y)] ].index(True)
    plt.figure()
    plt.imshow(data)
    plt.colorbar()
    global best_cost, global_plot_title
    plt.title(f"state abstractions\n{global_plot_title}\nglobal cost={best_cost}", fontsize=14)
    plt.savefig(f"{filename}.png")
    plt.close()

# Simon: given (b, t) maps to a new t, I believe
_abstract_model=[ [ mutually_exclusive(ABSTRACTSTATES)
                    for b0 in range(ABSTRACTACTIONS) ]
                  for t0 in range(ABSTRACTSTATES) ]

def abstract_model(abstract_indicators, option_indicators):
    final_abstract_state=mutually_exclusive(ABSTRACTSTATES)
    for t0 in range(ABSTRACTSTATES):
        for b0 in range(ABSTRACTACTIONS):
                for t1 in range(ABSTRACTSTATES):
                    constrain(Implies(And(_abstract_model[t0][b0][t1],
                                          abstract_indicators[t0],
                                          option_indicators[b0]),
                                      final_abstract_state[t1]))
    return final_abstract_state


def abstract_equal(t1, t2):
    return And(*[x==y for x, y in zip(t1, t2) ])

# make_controllers

low_level_policy = {}
stop_policy = {}
initiation_policy = {}

def pi(s, a, b, t):
    if (s, a, t) not in low_level_policy:
        low_level_policy[(s, a, t)] = [z3.FreshBool() for _ in range(ABSTRACTACTIONS) ]

    return low_level_policy[(s, a, t)][b]

def beta(s, b):
    if (s, b) not in stop_policy:
        stop_policy[((s, b))] = z3.FreshBool()

    return stop_policy[((s, b))]

def unique_stopping_state():
    # sufficient for landmarks
    # loss gets worse
    for b in range(ABSTRACTACTIONS):
        exactly_one([stop_policy[(((x, y), b))]
                     for x in range(N) for y in range(N)
                     if (((x, y), b)) in stop_policy])

def print_beta(m):
    returned_value=""
    for x in range(N):
        for y in range(N):
            s=(x, y)
            for b in range(ABSTRACTACTIONS):
                if ((s, b)) in stop_policy:
                    returned_value+=f"beta(s={(x, y)}, b={b}) = {m[stop_policy[((s, b))]]}\n"
    return returned_value


def visualize_options(m, folder):

    plt.figure()

    nr=arguments.states if arguments.condition_initial else 1
    nc=extract_number_of_options

    for t in range(nr):
        for b in range(nc):
            plt.subplot(nr, nc, b+1+t*nc)

            data = np.zeros((N*4+1, N*4+1))
            for x in range(N+1):
                data[:, 4*x]=0.5
                for y in range(N+1):
                    data[4*y, :]=0.5
            for x in range(N):

                for y in range(N):

                    s=(x, y)
                    for a in range(4):
                        if (s, a, t) in low_level_policy:
                            if m[pi(s, a, b, t)]:
                                dx={0: 0,
                                        1: 0,
                                        2: 1,
                                        3: -1}
                                dy={0: 1,
                                        1: -1,
                                        2: 0,
                                        3: 0}

                                data[2+y*4+dy[a], 2+x*4+dx[a]]=1

                    if ((s, b)) in stop_policy and m[stop_policy[((s, b))]]:
                        data[2+y*4, 2+x*4]=-1



            plt.imshow(data)
            if arguments.condition_initial:
                plt.title(f"tau0={t}, option={b}")
            else:
                plt.title(f"option={b}")

    global best_cost, global_plot_title
    #plt.suptitle(f"{global_plot_title}\nglobal cost={best_cost}", fontsize=14)

    plt.savefig(f"{folder}.png")
    #plt.show()
    plt.close()



def make_option_trajectory(T):
    return [mutually_exclusive(ABSTRACTACTIONS) for _ in range(T)]

def print_option_trajectory(m, trajectory):
    return str([ [m[b] for b in bs].index(True) for bs in trajectory ])

action_cost_table={}
def action_cost(s, a, option_indicators, abstract_indicators):
    if a is None: return 0.

    total_cost = 0.

    for t, abstract_indicator in enumerate(abstract_indicators):
        for option_index in range(ABSTRACTACTIONS):
            #exactly_one([pi(s, other_action, option_index, t) for other_action in range(4)])
            constrain(Implies(And(option_indicators[option_index], abstract_indicator),
                              pi(s, a, option_index, t)))

            if (s, option_index, t) in action_cost_table:
                this_cost = action_cost_table[(s, option_index, t)]
            else:
                other_possibilities = [pi(s, other_action, option_index, t) for other_action in range(4)]

                this_cost = z3.FreshReal()
                for k in range(1, 4+1):
                    constrain(Implies(exactly_k(other_possibilities, k),
                                      this_cost==math.log(k)))

                action_cost_table[(s, option_index, t)] = this_cost



            if total_cost is None:
                total_cost = this_cost
            else:
                # a bunch of chained if elses to match the option with its cost
                total_cost = If(And(option_indicators[option_index], abstract_indicator),
                                this_cost, total_cost)

    return total_cost



def print_abstract_state(m, t):
    return str([ m[x] for x in t ].index(True))

def extract_option_trajectory(m, options_ending):
    options, ending = options_ending
    return [[ m[x] for x in z ].index(True) for z in options ], [m[z] for z in ending ]


def observe_trajectory(states, actions):
    option_at_time = make_option_trajectory(len(states))

    ending = [z3.FreshBool() for _ in range(len(states)) ]



    #ending[t] = are we beginning a new option at this time step (previous time step ended)
    constrain(ending[0])
    #stop at the end of the trajectory
    constrain(ending[-1])

    if not arguments.predict_stop:
        for time in range(1, len(states)):
            constrain(ending[time] == Or(*[And(option_at_time[time-1][option_index],
                                               beta(states[time], option_index))
                                           for option_index in range(ABSTRACTACTIONS)]))


    # if you are not stopping then the option has to be the same at the next time step
    for time in range(len(states)-1):
        constrain(
            Implies(Not(ending[time+1]),
                    abstract_equal(option_at_time[time], option_at_time[time+1])))


    # causal consistency
    if arguments.consistent:
        # abstraction of state when we started
        # initial_abstract[t] should equal abstract[t'] for some t'<t
        initial_abstract = [mutually_exclusive(ABSTRACTSTATES)
                            for _ in range(len(states))]
        # abstraction of each time point
        abstract = [state_abstraction(states[time])
                    for time in range(len(states)) ]

        triggering_abstract = [mutually_exclusive(ABSTRACTSTATES)
                               for _ in range(len(states))]

        # don't care what happens at the last time step??
        for time in range(len(states)-1):
            for trigger_time in range(time+1):
                constrain(Implies(And(ending[trigger_time],
                                      Not(Or(*[ending[it] for it in range(trigger_time+1,time+1) ]))),
                                  abstract_equal(triggering_abstract[time],
                                                 abstract[trigger_time])))
        # constrain(abstract_equal(triggering_abstract[0],triggering_abstract[1]))
        # constrain(abstract_equal(triggering_abstract[0],abstract[0]))
        # constrain(Not(abstract_equal(triggering_abstract[0],triggering_abstract[-1])))

        for time in range(1, len(states)):
            t1 = abstract[time] # abstraction of current state
            t0 = initial_abstract[time]
            for initial_time in range(0, time): # initiation time
                check = And(ending[initial_time], Not(Or(*[ending[intermediate_time]
                                                           for intermediate_time in range(initial_time+1, time)])))
                constrain(Implies(check, abstract_equal(t0, abstract[initial_time])))

            t1p = abstract_model(t0, option_at_time[time-1]) # predicted abstract state
            constrain(Implies(ending[time],
                              abstract_equal(t1, t1p)))
            if arguments.predict_stop:
                constrain(Implies(abstract_equal(t1, t1p),
                                  ending[time]))

            # abstract actions change the abstract state
            if arguments.change:
                constrain(Implies(ending[time],
                                  Not(abstract_equal(t1, t0))))

    # reproduce data, hard+soft constraint
    low_level_action_cost=[]
    for time, (s, a, option_indicators) in enumerate(zip(states, actions, option_at_time)):

        this_cost = z3.FreshReal()
        if a is None:
            constrain(this_cost == 0.)
            low_level_action_cost.append(this_cost)
            continue

        if not arguments.condition_initial:
            constrain(action_cost(s, a, option_indicators, [True])==this_cost)
        else:
            constrain(action_cost(s, a, option_indicators, triggering_abstract[time])==this_cost)

        low_level_action_cost.append(this_cost)


    # compute how many times we switch options
    high_level_plan_cost=[]
    for time in range(len(states)):
        for no, number_indicator in enumerate(number_of_options):
            if no==0: continue
            constrain(Implies(number_indicator,
                              Not(Or(*option_at_time[time][no:]))))

        if time==0:
            # initial option is a freebie
            _this_time_cost = z3.FreshReal()
            constrain(0. ==_this_time_cost)
            high_level_plan_cost.append(_this_time_cost)
            continue

        this_time_cost=If(ending[time], single_option_cost, 0.)
        _this_time_cost = z3.FreshReal()
        constrain(this_time_cost ==_this_time_cost)
        high_level_plan_cost.append(_this_time_cost)

    return high_level_plan_cost, \
                 low_level_action_cost, option_at_time, ending, triggering_abstract


transition=make_model()[-1]
total_cost=0
option_at_time=[]


levels = [stuff
          for length in range(2, arguments.landmarks+1)
          for stuff in itertools.permutations([(0, 0), (0, N-1), (N-1, N-1), (N-1, 0)][:arguments.landmarks], length)
          ]

number_of_options = mutually_exclusive(arguments.options+1)
constrain(Not(number_of_options[0]))
single_option_cost=z3.FreshReal()
for no, number_indicator in enumerate(number_of_options):
    if no==0: continue
    constrain(Implies(number_indicator, single_option_cost==math.log(no)))

def extract_number_of_options(m):
    return [m[i] for i in number_of_options ].index(True)

hlc=[]
llc=[]
triggers=[]
for i, landmarks in enumerate(levels):
    states, actions = make_landmark_trajectory(landmarks[0], landmarks[1:])
    if i == 0 and arguments.consistent:
        constrain(state_abstraction(states[0])[0])

    hl, ll, options, starting, triggering = observe_trajectory(states, actions)

    hlc.append(hl)
    llc.append(ll)
    triggers.append(triggering)
    option_at_time.append((options, starting))
    total_cost += sum(hl+ll)
math.log(ABSTRACTACTIONS)

if arguments.unique:
    unique_stopping_state()

m, best_cost=minimize(total_cost)

for hl, ll, triggering in zip(hlc, llc, triggers):
    print("high level cost", [extract_real(m, x) for x in hl ])
    print("lo level cost", [extract_real(m, x) for x in ll ])
    print("abstract state triggering option", [[m[x] for x in t ].index(True) for t in triggering ])


# for i, landmarks in enumerate(levels):
#     states, actions = make_landmark_trajectory(landmarks[0], landmarks[1:])
#     visualize_trajectory(states, actions, str(i),
#                          extract_option_trajectory(m, option_at_time[i]))

visualize_trajectories([make_landmark_trajectory(landmarks[0], landmarks[1:])
                            for landmarks in levels ],
                       [extract_option_trajectory(m, option_at_time[i])
                        for i in range(len(levels)) ],
                       filename_prefix+"trajectories")

visualize_options(m, filename_prefix+"options")
visualize_states(m, filename_prefix+"states")
#print(print_pi(m))
print(filename_prefix)
print(f"RUN:\nthunar {filename_prefix}&")
