from autoencoder import Autoencoder
from data_generator import  *
from models import *
from environment import pretty_print_on_predicate

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


class WorldModelTrainer(pl.LightningModule):
    def __init__(self, dimension, autoencode=False, gsm=False, cc=False, reward=False, plan_loss=False, vanilla=False, symbolic=False, infer_options=False):
        super().__init__()

        dimension = dimension
        if symbolic:
            self.model = ProgramHanoiSimulator(dimension=dimension, infer_options=infer_options)
        else:
            self.model = NeuralHanoiSimulator(dimension=dimension,
                                              gsm=gsm, vanilla=vanilla)
        self.autoencode = autoencode
        self.cc = cc
        self.reward = reward
        self.plan_loss = plan_loss
        self.steps = 1
        self.symbolic = symbolic
        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    @property
    def temperature(self):
        r = 2*1e-5
        return max(0.01, math.exp(-r*self.steps))

    def training_step(self, train_batch, batch_idx):
        concrete_states, actions, is_goal = train_batch
        B, T = concrete_states.shape[:2]

        self.steps+=1
        self.log("temperature", self.temperature)

        # take the first time step
        initial_concrete_state = concrete_states[:,0]
                
        planning_loss, planning_mistakes = self.model.planning_loss(initial_concrete_state, actions, is_goal[:,1:], self.temperature)
        

        self.log("planning_loss", planning_loss)
        self.log("planning_mistakes", planning_mistakes)

        if not self.plan_loss:
            loss = 0
        else:
            loss = planning_loss

        # Pick random action time steps on which to evaluate causal consistency
        n_actions = actions.shape[1]
        action_time_steps = torch.randint(n_actions, (B,))
        selected_actions = actions[torch.arange(B),action_time_steps]
        selected_initial_states = concrete_states[torch.arange(B),action_time_steps]
        selected_final_states = concrete_states[torch.arange(B),action_time_steps+1]

        if False:
            #all of these pass
            assert torch.tensor([(selected_actions[b] == actions[b, action_time_steps[b]]).all()
                                 for b in range(B) ]).all()
            assert torch.tensor([(selected_initial_states[b] == concrete_states[b, action_time_steps[b]]).all()
                                 for b in range(B) ]).all()
            assert torch.tensor([(selected_final_states[b] == concrete_states[b, action_time_steps[b]+1]).all()
                                 for b in range(B) ]).all()

        cc_loss, cc_mistakes, precondition = \
                    self.model.causal_consistency_loss(selected_initial_states,
                                                       selected_actions,
                                                       selected_final_states,
                                                       self.temperature)

        self.log("cc_loss", cc_loss)
        self.log("cc_mistakes", cc_mistakes)
        self.log("cc_precondition", precondition)
        
        if self.cc:
            loss = loss + cc_loss + precondition
                
        reward_loss, reward_mistakes = self.model.goal_prediction_loss(concrete_states[:,0], is_goal[:,0], self.temperature)
        self.log("reward_loss", reward_loss)
        self.log("reward_mistakes", reward_mistakes)
        if self.reward:
            loss = loss + reward_loss            

        if self.autoencode:
            concrete_states = torch.reshape(concrete_states,
                                            (B, T, 3**3))
            concrete_states = torch.reshape(concrete_states,
                                            (B * T, 3**3))            
            ae_loss = self.model.autoencoder_loss(concrete_states)
            self.log("ae_loss", ae_loss)
            loss = loss + ae_loss

        if self.steps%1000 == 0 and self.symbolic:
            print("step", self.steps)
            print("initial state")
            print(selected_initial_states[0])
            s0 = selected_initial_states[0]
            s0 = State.invert_render(s0)
            print(pretty_print_on_predicate(s0.predicates()[0]))
            print()
            print("abstracts to")
            initial_abstract = self.model.encode(selected_initial_states).sample(self.temperature)
            initial_abstract = 1*(initial_abstract[0] > 0.5)
            
            print(initial_abstract)
            print(pretty_print_on_predicate(initial_abstract))
            print()
        
        return loss

        


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description = "")

    parser.add_argument("--cc", default=False, action="store_true",
                    help="train to be causally consistent")
    parser.add_argument("--vanilla", default=False, action="store_true",
                    help="vanilla rnn instead of gru")
    parser.add_argument("--symbolic", default=False, action="store_true",
                        help="use symbolic model, instead of learning neural network")
    parser.add_argument("--infer_options", default=False, action="store_true",
                        help="when using the symbolic model, do not take the option as given")

    parser.add_argument("--plan_loss", default=False, action="store_true",
                        help="train to predict reward after taking actions")
    
    parser.add_argument("--autoencode", "-a", default=False, action="store_true",
                        help="train an autoencoder")
    parser.add_argument("--layers", "-l", default=2, type=int,
                        help="pretraining layers")
    parser.add_argument("--hidden", default=64, type=int,
                        help="pretraining internal dimension")
    parser.add_argument("--dimension", default=128, type=int,
                        help="pretraining representation dimension")

    parser.add_argument("--gsm", "-g", default=False, action="store_true",
                        help="gumbel softmax")

    parser.add_argument("--reward", "-r", default=False, action="store_true",
                        help="learn to predict reward from abstract state")

    arguments = parser.parse_args()

    experiment_name = "+".join(
      [n if True == getattr(arguments,n) else \
       (f"{n}={getattr(arguments,n)}" if isinstance(getattr(arguments,n), int) else f"{n}={''.join(getattr(arguments,n))}")
       for n in sorted(arguments.__dir__())
       if not n.startswith("_") and getattr(arguments,n) and n != "pretrained" ])
    logger = TensorBoardLogger("tower_logs", name=experiment_name)
    print("LOGGING DIRECTORY", logger.root_dir)
    if False and  os.path.exists(logger.root_dir):
        assert False, f"already did {logger.root_dir}"

    assert arguments.reward or arguments.plan_loss or arguments.cc, "let me know at least one thing you want to have in the loss function!"

    m = WorldModelTrainer(dimension=arguments.dimension,
                          autoencode=arguments.autoencode,
                          symbolic=arguments.symbolic,
                          plan_loss=arguments.plan_loss,
                          gsm=arguments.gsm,
                          vanilla=arguments.vanilla, 
                          cc=arguments.cc,
                          infer_options=arguments.infer_options,
                          reward=arguments.reward)
    trainer = pl.Trainer(#detect_anomaly=True,
        gpus=1,
        logger=logger,
        max_epochs=50000)

    dataset = RolloutDataset()
    train_loader = DataLoader(dataset, batch_size=128)
    trainer.fit(m, train_loader)
