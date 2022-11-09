from autoencoder import Autoencoder

import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from torch import autograd


def probability_same(P,Q, epsilon=0.):
    """
    log probability that random samples from distributions P and Q come out exactly the same
    epsilon is smoothing that can help with numerical stability, keeps it from diverging
    """
    P = epsilon+(1-2*epsilon)*P
    Q = epsilon+(1-2*epsilon)*Q
    return (P*Q+(1-P)*(1-Q)).log()


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

    def render(self):
        "convert state to image w/ one color for each different peg size"
        h = sum(len(r) for r in self.rings)
        w = len(self.rings)

        # i[a, b, c] says that there is a ring of type c at height b on peg asum
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
        # on[a, b] says whether ring/peg is on another ring/peg
        on = np.zeros((len(self.rings)*2, len(self.rings) *2))
        # clear[a] says whether ring/peg is clear. first three are rings, second three are pegs
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
        # size of ring on top of peg i
        x = self.rings[i][-1]-1
        # size of ring on top of peg j, or the peg itself
        if len(self.rings[j]) > 0:
            z = self.rings[j][-1]-1
        else:
            z = j+3

        # new top of peg i, or the peg itself
        if len(self.rings[i]) > 1:
            y = self.rings[i][-2]-1
        else:
            y = i+3
        assert np.sum(strips_action) < 1
        # move size x on top of size y or peg y. new top of peg x was on is z
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

#assert False


def rollout(l=10):
    states, actions, ons, clears, strips_actions = [], [], [], [], []
    s = State([[3,2,1],[],[]])
    for _ in range(l):
        a = s.random_legal_action()
        states.append(s.render())
        action_matrix = np.zeros((3,3))
        action_matrix[a[0],a[1]] = 1
        actions.append(action_matrix)

        predicates = s.predicates()
        ons.append(predicates[0])
        clears.append(predicates[1])

        strips_actions.append(s.strips_action(a))
        s = s.step(a)
        #assert any(l.rings == s.rings for l in legal_states )

    return torch.tensor(np.stack(states)).float(), \
        torch.tensor(np.stack(actions)).float(), \
        torch.tensor(np.stack(ons)).float(), \
        torch.tensor(np.stack(clears)).float(), \
        torch.tensor(np.stack(strips_actions)).float()


class TransitionDataset(torch.utils.data.IterableDataset):
    def __init__(self, start=0, end=1280):
        super(TransitionDataset).__init__()
        assert end > start, "this example code only works with end >= start"
        # precompute the set of all legal state transitions
        legal_states = [ State([[], [], []])]
        for new_disk in [3,2,1]:
            legal_states = [ State([ r if j != i else r+[new_disk]
                                     for j, r in enumerate(s.rings) ])
                             for s in legal_states for i in range(3) ]
        self.legal_transitions = [(s, a, s.step(a)) for s in legal_states for a in s.legal_actions() ]

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

class RepresentationLearner(pl.LightningModule):
    def __init__(self, cnn=False, layers=2, hidden=64, dimension=16, random_features=False, gsm=False):
        super().__init__()
        self.save_hyperparameters()

        self.cnn = cnn

        self.dimension = dimension

        self.entities = 6

        self.random_features = random_features
        
        self.gsm = gsm

        self.steps=0

        # we are going to try and decode the state from this representation
        self.state_representation = Autoencoder([3,3,3] if cnn else [27],
                                                dimension=dimension, layers=layers, hidden=hidden,
                                                loss="bce", stride=1, discrete=gsm)
        self.state_predictor = nn.Linear(dimension, self.entities*self.entities)

        # and decode the action from this representation
        self.action_representation = Autoencoder([5,3,3] if cnn else [36],
                                                 dimension=dimension, layers=layers, hidden=hidden,
                                                 loss="bce", stride=1, discrete=gsm)
        self.action_predictor = nn.Linear(dimension, self.entities*self.entities*self.entities)

        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx, logger=None):
        if logger is None: logger = self
            
        s0, a0, s1, on_gt, clear_gt, a_gt = train_batch
        B = s0.shape[0]

        if self.cnn:
            zs, ls = self.state_representation(s0)
            assert False, "not implemented for cnn because I think that that is overkill"
            za, la = self.action_representation(s0)
        else:
            zs, ls = self.state_representation(s0.view(B, -1))
            za, la = self.action_representation(torch.concat([a0.view(B, -1), s0.view(B, -1)],1))
            
        on_h = torch.sigmoid(self.state_predictor(zs.detach()).view(B, self.entities, self.entities))
        state_loss = nn.BCELoss(reduction="none")(on_h, on_gt).sum(-1).sum(-1).mean()
        state_mistakes=(((on_h>0.5)!=(on_gt>0.5))*1.).sum(-1).sum(-1).mean()

        logger.log("state_predict_loss", state_loss)
        logger.log("state_mistakes", state_mistakes)
        logger.log("state_ae_likelihood", ls)

        action_h = torch.sigmoid(self.action_predictor(za.detach()).view(B, self.entities, self.entities, self.entities))
        action_loss = nn.BCELoss(reduction="none")(action_h, a_gt).sum(-1).sum(-1).sum(-1).mean()
        action_mistakes=(((action_h>0.5)!=(a_gt>0.5))*1.).sum(-1).sum(-1).sum(-1).mean()

        logger.log("action_predict_loss", action_loss)
        logger.log("action_mistakes", action_mistakes)
        logger.log("action_ae_likelihood", la)
        
        if self.gsm: logger.log("temperature", self.state_representation.temperature)
        
        self.steps+=1
        if self.steps%100==0:
            print("STEP", self.steps,
                  "\nground truth:\n", on_gt[0], "\npredicted:", (on_h[0]>0.5)*1.,
                  "\n\n")

        if self.random_features:
            return state_loss
        
        return action_loss + state_loss - ls - la
        
        
        
        
            
            
        
class Abstraction(pl.LightningModule):

    def __init__(self, hardcode=False, function=False, on_constraints=False, cnn=False, supervise=[], contrastive=False, reward=False, marginal=False, gsm=False, over=False, autoencoder=None):
        super().__init__()

        self.gsm, self.marginal, self.hardcode, self.function, self.on_constraints, self.cnn, self.supervise, self.contrastive, self.over = gsm, marginal, hardcode, function, on_constraints, cnn, supervise, contrastive, over

        
        self.entities = 6
        self.pegs = 3
        self.step = 0
        
        self.autoencoder = autoencoder

        if self.over:
            # overparameterize by not hard coding the sizes of the disks
            # compute SMALLER predicate my comparing sizes on-the-fly as we learn the sizes
            self._sizes =  torch.nn.Parameter(torch.randn(self.entities, requires_grad=True).float().cuda())
        else:
            smaller = np.zeros((self.entities, self.entities))
            for i in range(3):
                for j in range(i+1,self.entities):
                    smaller[i,j] = 1
            self._smaller = torch.tensor(smaller, requires_grad=False).float().cuda()
        # ...but we still hard code what is a disk
        self.is_disk = torch.tensor(np.array([1,1,1,0,0,0]), requires_grad=False).float().cuda()

        self.dummy = nn.Linear(1,1)

        # create state and action encoders, if we need them because we do not have the encoder/decoder
        if self.autoencoder:
            h=autoencoder.dimension
            # head which outputs abstract action
            self.action = nn.Sequential(
                nn.Linear(h, self.entities*self.entities*self.entities),
                nn.Softmax(-1))
        elif cnn:
            h=32
            self.state_abstraction = nn.Sequential(
                nn.Conv2d(3, h, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(h, h, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(h, 32, kernel_size=3, stride=1, padding=1),
                nn.Flatten(),
                nn.ReLU(),
                nn.Linear(32*self.pegs**2, h),
                nn.ReLU(),
            )
            self.action = nn.Sequential(
                nn.Conv2d(3+2, h, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(h, h, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(h, 32, kernel_size=3, stride=1, padding=1),
                nn.Flatten(),
                nn.ReLU(),
                nn.Linear(32*self.pegs**2, self.entities*self.entities*self.entities),
                nn.Softmax(-1)
            )
        else:
            h = 256
            self.state_abstraction = nn.Sequential(nn.Linear(self.pegs**3, h),
                                               nn.ReLU(),
                                               nn.Linear(h, h),
                                               nn.ReLU(),
                                               nn.Linear(h, h),
                                               nn.ReLU(),
            )
            self.action = nn.Sequential(nn.Linear(self.pegs**3+self.pegs*self.pegs, h),
                                    nn.ReLU(),
                                    nn.Linear(h, h),
                                    nn.ReLU(),
                                    nn.Linear(h, h),
                                    nn.ReLU(),
                                    nn.Linear(h, self.entities*self.entities*self.entities),
                                    nn.Softmax(-1)
            )

        # heads which predicts CLEAR/ON predicate
        self.clear = nn.Linear(h, self.entities)
        if on_constraints:
            self.on = nn.Linear(h, self.pegs*self.entities)
        else:
            self.on = nn.Linear(h, self.entities*self.entities)

        # head which predicts reward
        # we always train this, but only pass gradients backward if reward=True
        self.reward = reward
        self._reward = nn.Sequential(nn.Linear(self.entities**2, h),
                                    nn.ReLU(),
                                    nn.Linear(h,1))

    @property
    def smaller(self):
        if self.over:
            return torch.sigmoid(self._sizes.unsqueeze(0)-self._sizes.unsqueeze(1))
        return self._smaller

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):

        self.step+=1
        if self.autoencoder:
            autoencoder_loss = self.autoencoder.training_step(train_batch, batch_idx,
                                                              logger=self)
            if arguments.pretrained:
                autoencoder_loss = autoencoder_loss.detach()
            #if self.step < 2000: return autoencoder_loss                
        else:
            autoencoder_loss = 0
            
        s0, a0, s1, on_gt, clear_gt, a_gt = train_batch

        B = s0.shape[0] # batch size

        # predicted abstract representation at time step 0
        clear_h_0, on_h_0, a_h_0 = self(s0, a0, gsm=self.gsm)

        # predicted abstract representation at time step 1 (not from simulation)
        clear_1, on_1 = self(s1, actions=None)

        # reward prediction
        reward = (s0[:,0].sum(-1).sum(-1) == 3)*1.
        if not self.reward: reward = reward.detach()
        rh = self._reward(on_h_0.view(B, -1)).squeeze(-1)
        reward_loss = ((reward-rh)**2).mean()
        self.log("reward_loss", reward_loss)

        if self.marginal: # EXPERIMENTAL, unsure of it's working
            on_h_1, clear_h_1, precondition = self.simulate_all_actions(clear_h_0, on_h_0)
            # on is correct, clear is correct, precondition holds
            probability_all_correct = probability_same(on_1.unsqueeze(1).unsqueeze(1).unsqueeze(1), on_h_1, epsilon=1e-3).sum(-1).sum(-1)+precondition
            if precondition.isnan().any():
                import pdb; pdb.set_trace()
            if precondition.isinf().any():
                import pdb; pdb.set_trace()

            probability_all_correct = probability_all_correct.view(B, -1)
            marginal = probability_all_correct.max(-1).values.mean()
            self.log('marginal_likelihood', marginal)
            print("marginal_likelihood", marginal)
            if "nan" in str(marginal):
                import pdb; pdb.set_trace()

            consistency_likelihood = marginal
        else: # pay attention to this

            # predicted abstract representation at time step 1 from simulation
            on_h_1, clear_h_1, precondition = self.simulate(clear_h_0, on_h_0, a_h_0)

            # what is a probability that we satisfy the precondition?
            self.log('precondition_likelihood', precondition.mean())
            self.log('precondition_accuracy', (1.*(precondition.exp()>0.5)).mean())

            # what is he probability that we correctly predict CLEAR/ON?
            clear_likelihood = probability_same(clear_1, clear_h_1).mean()
            on_likelihood = probability_same(on_1, on_h_1).mean()

            self.log('clear_likelihood', clear_likelihood)
            self.log('on_likelihood', on_likelihood)

            self.log('clear_accuracy', (((clear_gt > 0.5) == (clear_h_0 > 0.5))*1.).mean())
            self.log('on_accuracy', (1.*((on_gt > 0.5) == (on_h_0 > 0.5))).mean())

            # collectively these define the causal consistency loss
            if self.function: # clear is a function of on, no need to include clear in the loss
                consistency_likelihood = precondition.mean()+on_likelihood
            else:
                consistency_likelihood = precondition.mean()+clear_likelihood+on_likelihood
            self.log("causal_consistency_loss", -consistency_likelihood)

        # Also we can supervise on the state-action
        supervised_state = probability_same(clear_gt, clear_h_0).mean() +\
                                probability_same(on_gt, on_h_0).mean()
        supervised_action = probability_same(a_gt, a_h_0).sum(-1).sum(-1).mean()
        self.log("supervised_state", supervised_state)
        self.log("supervised_action", supervised_action)

        if self.contrastive:
            #on_1 should look like on_h_1
            #and should be contrasted with on_h_0
            should_be_big = probability_same(on_1, on_h_1, epsilon=1e-3).sum(-1).sum(-1)
            should_be_small = probability_same(on_1, on_h_0, epsilon=1e-3).sum(-1).sum(-1)
            margin = 2
            contrast_loss_same = (should_be_small-should_be_big+margin).clamp(min=0).mean()
            self.log('contrast_loss_same', contrast_loss_same)

            # contrast with random other points
            should_be_small = probability_same(on_h_0[torch.randperm(B)],
                                               on_h_0, epsilon=1e-3).sum(-1).sum(-1)
            contrast_loss_random = (should_be_small-should_be_big+margin).clamp(min=0).mean()
            self.log('contrast_loss_random', contrast_loss_random)
            contrast_loss = contrast_loss_random+contrast_loss_same


        loss = -consistency_likelihood + reward_loss

        if "action" in self.supervise: loss = -10*supervised_action +loss
        if "state" in self.supervise: loss = loss + -10*supervised_state
        if self.contrastive: loss = loss + 10*contrast_loss

        self.log('train_loss', loss)

        if self.step%10 == 0:
            print()
            print()
            print("step", self.step)
            if self.over: print("sizes", self._sizes)
            print("actual initial state", s0[0])
            print("actual action", a0[0])
            print("predicted initial abstract state, clear", (clear_h_0[0] > 0.5)*1)
            print("predicted initial abstract state, on", (on_h_0[0] > 0.5)*1)
            print("predicted abstract action", (a_h_0[0] > 0.5)*1)
            print()
            print()
        
        # ignore this hack, hardcode=False for all the experiments we care about
        if self.hardcode:
            return self.dummy(loss.unsqueeze(0))-self.dummy(loss.unsqueeze(0))+loss+autoencoder_loss
        return loss+autoencoder_loss

    def forward(self, states, actions, gsm=False):
        """
        given microscopic states and actions, predict abstract state/action
        if actions is none, then just abstract the states

        gsm: gumbel softmax abstract state-action
        """
        B = states.shape[0]
        
        if self.autoencoder:
            tau = self.autoencoder.state_representation.encode(states.view(B, -1))[1]
            if arguments.pretrained: tau = tau.detach()
        elif self.cnn:
            tau = self.state_abstraction(states.transpose(-1,-3))
        else:
            tau = self.state_abstraction(states.view(*states.shape[:-3], self.pegs**3))

        on = self.on(tau)
        if self.on_constraints:
            on = on.view(*on.shape[:-1], self.pegs, self.entities)
            if gsm:
                temperature = 2**(-self.step/1000)
                on = F.gumbel_softmax(on, tau=temperature)
                self.log("temperature", temperature)
            else:
                # each disk is on exactly one entity
                on = torch.softmax(on, -1)
            # there are other entities corresponding to the pegs, and they are on nothing
            peg_on = torch.zeros_like(on)
            on = torch.concat([on, peg_on], 1)
        else:
            assert not gsm
            on = on.view(*on.shape[:-1], self.entities, self.entities)
            on = torch.sigmoid(on)

        if self.function:
            #clear[x] = ~on[y,x], for all y
            clear = torch.prod(1-on, -2)
        else:
            assert not gsm
            clear = torch.sigmoid(self.clear(tau))

        if actions is None: return clear, on

        if self.autoencoder:
            action = self.autoencoder.action_representation.encode(torch.concat([actions.view(B, -1),
                                                                                 states.view(B, -1)],1))[1]
            if arguments.pretrained: action = action.detach()
            action = self.action(action)
        elif arguments.cnn:
            B = states.shape[0]
            states = states.transpose(-1,-3)

            # stuff the action into the channel dimension
            # this is a pain
            action_index = actions.view(actions.shape[0], -1).argmax(-1).cpu()
            source_index = action_index // self.pegs
            destination_index = action_index % self.pegs
            action_info = torch.zeros(B, 2, self.pegs, self.pegs)
            for b in range(B):
                action_info[b,0,:,source_index[b]] = 1
                action_info[b,1,:,destination_index[b]] = 1
            input = torch.concat([states, action_info.to(states.device)],-3)
            action = self.action(input)
        else:
            actions = actions.view(*actions.shape[:-2], self.pegs**2)
            action = self.action(torch.concat([ states.view(*states.shape[:-3], self.pegs**3)
                                                , actions ], -1))

        
        
        action = action.view(*action.shape[:-1], self.entities, self.entities, self.entities)
        return clear, on, action

    def simulate(self, clear, on, actions):
        """
        simulate abstract model, given abstract predicates CLEAR/ON to predict those predicates at the next time step
        """

        # move(x,y,z) -> clear(y)
        add_clear = actions.sum(-1).sum(-2)
        # move(x,y,z) -> ~clear(z)
        del_clear = actions.sum(-2).sum(-2)

        # move(x,y,z) -> on(x,z)
        add_on = actions.sum(-2)
        # move(x,y,z) -> ~on(x,y)
        del_on = actions.sum(-1)

        precondition = torch.einsum("bxyz,by,bx,bxy,bz,xz,x->b",
                                    actions, 1-clear, clear, on, clear, self.smaller, self.is_disk).log()

        def update(old, add, delete):
            return (1-delete)*(old+add-old*add)

        on = update(on, add_on, del_on)
        clear = update(clear, add_clear, del_clear)

        return on, clear, precondition

    def simulate2(self, clear, on, actions):
        """
        simulate abstract model, given abstract predicates CLEAR/ON to predict those predicates at the next time step
        """

        
        # move(x,z) -> on(x,z)
        add_on = actions
        # move(x,z) & on() -> ~on(x,y)
        del_on = actions.sum(-1)

        precondition = torch.einsum("bxyz,by,bx,bxy,bz,xz,x->b",
                                    actions, 1-clear, clear, on, clear, self.smaller, self.is_disk).log()

        def update(old, add, delete):
            return (1-delete)*(old+add-old*add)

        on = update(on, add_on, del_on)
        clear = update(clear, add_clear, del_clear)

        return on, clear, precondition

    def simulate_all_actions(self, clear, on):
        """
        experimental code to simulate every possible action. used by marginal but I'm not sure about if this is good idea
        """
        B = clear.shape[0]

        precondition = torch.einsum("by,bx,bxy,bz,xz->bxyz",
                                    1-clear, clear, on, clear, self.smaller).clamp(min=1e-8).log()
        # new on:
        # (1-delete_xyzuv) * (old_buv+add_bxyzuv-old_buv*add_xyzuv)
        # add_xyzuv = 1[u=x] * 1[v=z]
        # del_xyzuv = 1[u=x] * 1[v=y]
        same = torch.eye(self.entities).float().cuda()
        add = torch.einsum("xu,zv->xzuv",
                           same, same)
        # for x in range(self.entities):
        #     for z in range(self.entities):
        #         for u in range(self.entities):
        #             for v in range(self.entities):
        #                 assert add[x,z,u,v] == 1.*(u == x and v == z)
        delete = add.unsqueeze(2).unsqueeze(0)
        add = add.unsqueeze(1).unsqueeze(0)

        _on = on.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        on_prediction = (1-delete) * (_on+add-_on*add)
        clear_prediction = torch.prod(1-on_prediction, -2)
        # for b in range(B):
        #     for x in range(self.entities):
        #         for y in range(self.entities):
        #             for z in range(self.entities):
        #                 for u in range(self.entities):
        #                     for v in range(self.entities):
        #                         a = 1.*(u == x and v == z)
        #                         d = 1.*(u == x and v == y)
        #                         o = on[b, u, v]
        #                         assert prediction[b,x,y,z,u,v] == (1-d)*(a+o-a*o)

        return on_prediction, clear_prediction, precondition


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description = "")

    parser.add_argument("--autoencode", "-a", default=False, action="store_true",
                        help="pretrain an autoencoder")
    parser.add_argument("--layers", "-l", default=2, type=int,
                        help="pretraining layers")
    parser.add_argument("--hidden", default=64, type=int,
                        help="pretraining internal dimension")
    parser.add_argument("--dimension", default=16, type=int,
                        help="pretraining representation dimension")
    parser.add_argument("--random", default=False, action="store_true",
                        help="pretraining with frozen random features, as a control experiment")

    parser.add_argument("--pretrained", default=None, 
                        help="load pretraining checkpoint")

    parser.add_argument("--contrastive", "-c", default=False, action="store_true",
                        help="various contrastive objectives designed to make the model not predict the exact same abstracts date for everything")
    parser.add_argument("--function", "-f", default=False, action="store_true",
                        help="constraint that CLEAR predicate is a deterministic function of ON predicate (good idea)")
    parser.add_argument("--on_constraints", "-o", default=False, action="store_true",
                        help="constraint that pegs cannot be ON anything (good idea)")

    parser.add_argument("--gsm", "-g", default=False, action="store_true",
                        help="gumbel softmax before running abstract model")

    parser.add_argument("--cnn", "-v", default=False, action="store_true",
                        help="convolutional encoders instead of mlp")

    parser.add_argument("--reward", "-r", default=False, action="store_true",
                        help="learn to predict reward from abstract state")

    parser.add_argument("--over", default=False, action="store_true",
                        help="overparameterized problem (in progress, could be significantly more over parametrized)")

    parser.add_argument("--supervise", "-s", default="", nargs="+",
                        choices=["state", "action"],
                        help="directly supervise the state/action prediction networks using extra terms in the loss")

    parser.add_argument("--hardcode", "-H", default=False, action="store_true",
                        help="hardcoded solution. deprecated in favor of supervise")

    parser.add_argument("--marginal", "-m", default=False, action="store_true",
                        help="marginalize over action instead of learning to predict it (experimental, seems not to work)")

    arguments = parser.parse_args()

    experiment_name = "+".join(
      [n if True == getattr(arguments,n) else \
       (f"{n}={getattr(arguments,n)}" if isinstance(getattr(arguments,n), int) else f"{n}={''.join(getattr(arguments,n))}")
       for n in sorted(arguments.__dir__())
       if not n.startswith("_") and getattr(arguments,n) and n != "pretrained" ])
    logger = TensorBoardLogger("lightning_logs", name=experiment_name)
    print("LOGGING DIRECTORY", logger.root_dir)
    if False and  os.path.exists(logger.root_dir):
        assert False, f"already did {logger.root_dir}"
    
    if arguments.pretrained:
        ae = RepresentationLearner.load_from_checkpoint(arguments.pretrained)
    else:
        ae = RepresentationLearner(cnn=arguments.cnn,
                                  layers=arguments.layers,
                                  hidden=arguments.hidden,
                                  dimension=arguments.dimension,
                                  gsm=arguments.gsm,
                                  random_features=arguments.random)
    assert arguments.function
    if arguments.autoencode:
        m = ae
    else:
        m = Abstraction(hardcode=arguments.hardcode,
                        function=arguments.function,
                        on_constraints=arguments.on_constraints,
                        supervise=arguments.supervise,
                        contrastive=arguments.contrastive,
                        cnn=arguments.cnn,
                        gsm=False, #arguments.gsm,
                        marginal=arguments.marginal,
                        over=arguments.over,
                        autoencoder=ae)
        
    trainer = pl.Trainer(detect_anomaly=True,
                         gpus=1,
                         logger=logger,
                         max_epochs=2000)
    dataset = TransitionDataset()
    train_loader = DataLoader(dataset, batch_size=dataset.end)
    trainer.fit(m, train_loader)
    




