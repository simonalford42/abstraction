import math

from rdistribution import *
from autoencoder import Autoencoder

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader


        
        


class Simulator(nn.Module):
    """abstract class for abstract world simulators"""
    
    def sim(self, abstract_state, actions, temperature):
        """simulates multiple time steps
        returns the distribution over abstracts state at each time step, as well as a sample from that distribution
        
        abstract_state: [b, *]
        actions: [b, t, *]
        temperature: float

        returned lists are of length t, and tell you the abstract state after taking a prefix of the actions
        """

        distributions = []
        samples = []
        
        T = actions.shape[1]
        for t in range(T):
            abstract_state = self.sim1(abstract_state, actions[:,t], temperature)
            distributions.append(abstract_state)
            samples.append(abstract_state.sample(temperature))
            # we have to simulate another time step, so make sure that will we feed into the simulator is a concrete sample
            abstract_state = samples[-1]
                
        return distributions, samples

    def sim1(self, abstract_state, action, temperature):
        """simulates a single action
        returns RDistribution"""
        assert False, 'not implemented'
        
    def predict_goal(self, abstract_states):
        assert False, 'not implemented'

    def encode(self, concrete_states):
        """returns RDistribution"""
        assert False, 'not implemented'

    def encode_action(self, action):
        """override if needed"""
        return action

    def precondition_log_prob(self, state, action):
        """override if needed"""
        return torch.zeros((1,), device=state.device)

    def planning_loss(self, initial_concrete_state, actions, is_goal, temperature):
        """
        initial_concrete_state: [b, *]
        actions: [b, t, *]
        is_goal: [b, t, *], should be binary. last dimension ranges over possible goals

        simulates actions using the world model starting from the provided initial concrete states
        tries to predict whether or not the final state is a goal state
        loss is binary classification based on this goal state prediction
        """

        assert is_goal.shape[1] == actions.shape[1]
        assert initial_concrete_state.shape[0] == actions.shape[0]
        assert is_goal.shape[0] == actions.shape[0]
        
        initial_abstract_state = self.encode(initial_concrete_state).sample(temperature)
        actions = self.encode_action(actions)
        _, subsequent_abstract_states = self.sim(initial_abstract_state, actions, temperature)
        
        assert len(subsequent_abstract_states) == actions.shape[1]
        subsequent_abstract_states = torch.stack(subsequent_abstract_states).transpose(0,1).contiguous()        
        
        predicted_is_goal = self.predict_goal(subsequent_abstract_states)
        loss = nn.BCEWithLogitsLoss(reduction="none")(predicted_is_goal, is_goal)
        mistakes = ((((predicted_is_goal>0.)*1 - (is_goal>0.5)*1).abs() > 0.5)*1).sum()
                
        return loss.sum(-1).sum(-1).mean(), mistakes

    def causal_consistency_loss(self, s0, a, s1, temperature):
        """
        s0, a, s1: (b, *)
        single step causal consistency
        """
        B = s0.shape[0]
        
        initial_abstract_state = self.encode(s0).sample(temperature)
        final_abstract_state_distribution = self.encode(s1)
        
        a = self.encode_action(a)
        predicted_final_abstract_state = self.sim1(initial_abstract_state, a, temperature)

        pre = -self.precondition_log_prob(initial_abstract_state, a)

        return *predicted_final_abstract_state.loss(final_abstract_state_distribution), pre

    def goal_prediction_loss(self, concrete_state, is_goal, temperature):
        """can we predict the goal from the abstraction of the state?"""
        abstract_state = self.encode(concrete_state).sample(temperature)
        predicted_is_goal = self.predict_goal(abstract_state)
        loss = nn.BCEWithLogitsLoss(reduction="none")(predicted_is_goal, is_goal)
        mistakes = ((((predicted_is_goal>0.)*1 - (is_goal>0.5)*1).abs() > 0.5)*1).sum()
                
        return loss.sum(-1).mean(), mistakes

class NeuralSimulator(Simulator):
    def __init__(self, dimension=64, layers=1,
                 gsm=False, vanilla=False):
        super().__init__()

        if vanilla: cls = nn.RNN
        else: cls = nn.GRU

        if gsm: hidden_size = dimension//2
        else: hidden_size = dimension
            
        self.rnn = cls(input_size=dimension,
                       hidden_size=hidden_size,
                       num_layers=layers,
                       batch_first=True)
        
        self.goal_predictor = nn.Linear(dimension*layers, self.n_goals)

        self.gsm = gsm
        self.steps = 1
        self.vanilla = vanilla

    def sim1(self, abstract_state, action, temperature):
        if not self.gsm:
            o, h = self.rnn(action.unsqueeze(1), abstract_state.unsqueeze(0))
            return DeltaDistribution(h.squeeze(0))
        else:
            B = abstract_state.shape[0]
            
            assert self.n_categories == 2
            
            h = probability_to_gru_hidden(abstract_state)
                        
            o, h = self.rnn(action.unsqueeze(1), h.unsqueeze(0))
            h = gru_hidden_to_probability(h).squeeze(0)
            assert h.shape[0] == B
            assert h.shape[-1] == 2
            return CategoricalGumbel(h)

    def predict_goal(self, abstract_states):
        if self.gsm:
            abstract_states = abstract_states.view(*abstract_states.shape[:-2],
                                                   self.n_variables*self.n_categories)            
        return self.goal_predictor(abstract_states)

class NeuralHanoiSimulator(NeuralSimulator):
    def __init__(self, dimension=64, layers=1, gsm=False, vanilla=False):
        super().__init__(dimension, layers, gsm=gsm, vanilla=vanilla)
        
        self.state_representation = Autoencoder([27],
                                                dimension=dimension*layers,
                                                layers=layers, hidden=dimension,
                                                loss="bce", stride=1, discrete=gsm)
        self.action_representation = nn.Linear(6*6*6, dimension)

    @property
    def n_variables(self):
        return self.state_representation.n_variables

    @property
    def n_categories(self):
        return self.state_representation.n_categories

    @property
    def n_goals(self): return 3

    def encode_action(self, action):
        # flatten last three dimensions
        action = action.view(*action.shape[:-3], -1)
        return self.action_representation(action)        

    def encode(self, concrete_states):
        B = concrete_states.shape[0]
        
        sampled, distribution = self.state_representation.encode(concrete_states.view(B, -1))

        if self.gsm:
            distribution = distribution.view(B, self.n_variables, self.n_categories)
            assert torch.all(((distribution.sum(-1)-1).abs() < 1e-5))
            return CategoricalGumbel(distribution.log())
        else:
            return DeltaDistribution(distribution)
    
    def autoencoder_loss(self, concrete_states):
        return -self.model.state_representation(concrete_states)[-1]

class ProgramHanoiSimulator(Simulator):
    def __init__(self, dimension, infer_options=False):
        super().__init__()
        
        self.state_representation = nn.Sequential(nn.Linear(27, dimension),
                                                  nn.ReLU(),
                                                  nn.Linear(dimension, 6*3))

        self.entities, self.pegs = 6, 3
        
        smaller = np.zeros((self.entities, self.entities))
        for i in range(self.pegs):
            for j in range(i+1,self.entities):
                smaller[i,j] = 1
        self.smaller = torch.tensor(smaller, requires_grad=False).float().cuda()
        self.is_disk = torch.tensor(np.array([1]*self.pegs+[0]*self.pegs), requires_grad=False).float().cuda()

        self.infer_options = infer_options
        if self.infer_options:
            self.predict_option = nn.Linear(self.entities**3, self.entities**3)
        
    @property
    def n_goals(self): return 3

    def encode_action(self, action):
        if self.infer_options:
            option = self.predict_option(action.view(*action.shape[:-3], self.entities**3))

            option = option.view(*action.shape)
            # action(x,y,z)
            # x is disk
            option = option + self.is_disk.unsqueeze(-1).unsqueeze(-1).log()
            # x smaller than y
            option = option + self.smaller.unsqueeze(-1).log()
            # x smaller than z
            option = option + self.smaller.unsqueeze(-2).log()
            # y!=z
            option = option + (1-torch.eye(self.entities)).unsqueeze(0).log().to(option.device)

            # normalize, which requires reshaping
            option = torch.softmax(option.view(*action.shape[:-3], self.entities**3), -1).\
                     view(*action.shape[:-3], self.entities, self.entities, self.entities)
            
            return option
        else:
            return action
    
    def encode(self, concrete_states):
        B = concrete_states.shape[0]
        
        distribution = self.state_representation(concrete_states.view(B, -1))
        distribution = distribution.view(B, 3, 6)
        distribution = self.smaller[:3,:].log()+distribution
        distribution = torch.log_softmax(distribution, -1)

        return CategoricalGumbel(distribution)
        
    def predict_goal(self, abstract_states):
        smallest_disks_on_each_other = abstract_states[..., 0, 1]*abstract_states[..., 1, 2]
        biggest_disk_on_peg = abstract_states[..., 2, 3:]
        
        goal_probability = smallest_disks_on_each_other.unsqueeze(-1)*biggest_disk_on_peg
        
        e = 1e-5
        goal_probability = (1-2*e)*goal_probability + e
        # inverse sigmoid
        return -(1./goal_probability-1).log()

    def state_to_on_and_clear_predicates(self, state):
        #introduce variables for pegs
        on = torch.concat([state, torch.zeros_like(state)], -2)
        #clear[x] = ~on[y,x], for all y
        clear = torch.prod(1-on, -2)
        return on, clear

    def precondition_log_prob(self, state, action):
        on, clear = self.state_to_on_and_clear_predicates(state)

        action_precondition = torch.einsum("bxyz,by,bx,bxy,bz,xz,xy,x->b",
                                           action, 1-clear, clear, on, clear,
                                           self.smaller, self.smaller, self.is_disk)

        # no two disks are on the same thing
        # for all x, by construction, \sum_y on(x,y) = 1
        # but we don't have that for all y, \sum_x on(x,y) <= 1
        # can enforce this by saying for all y, for all x!=x', ~(on(x,y) and on(x,y))
        x_pairs = [(0,1), (0,2), (1,2)]
        invariant = [ 1 - state[:,x,:]*state[:,xp,:] for x,xp in x_pairs ]

        # sum, log, smoothing for numerical stability, average of across batch
        action_precondition = (action_precondition+1e-6).log().mean()
        invariant = sum([ (i+1e-6).log() for i in invariant ]).mean()
                        
        return action_precondition + invariant
        
    

    def sim1(self, state, action, temperature):
        on, clear = self.state_to_on_and_clear_predicates(state)

        # move(x,y,z) -> clear(y)
        add_clear = action.sum(-1).sum(-2)
        # move(x,y,z) -> ~clear(z)
        del_clear = action.sum(-2).sum(-2)

        # move(x,y,z) -> on(x,z)
        add_on = action.sum(-2)
        # move(x,y,z) -> ~on(x,y)
        del_on = action.sum(-1)

        def update(old, add, delete):
            return (1-delete)*(old+add-old*add)

        on = update(on, add_on, del_on)[:, :3, :]

        return CategoricalGumbel((on+1e-5).log())


"""discrete rnn states utility functions"""
def gru_hidden_to_probability(h):
    """
    h: [B, D], elements between [-1, +1]
    returns: [B, D, 2], elements should be interpreted as log probabilities
    """
    h = (h+1)/2 # now we can think of these as probabilities
    h = torch.stack([1-h,h], dim=-1)
    return (h+1e-9).log()

def probability_to_gru_hidden(h):
    """
    h: [..., D, 2]
    returns: [B, D], elements between [-1,+1]
    """
    assert torch.all((h.sum(-1)-1).abs() < 1e-5)
    return h[..., 1]*2 - 1
    

