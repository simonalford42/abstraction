from typing import List, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from utils import assert_equal, DEVICE, assert_shape
import warnings
import boxworld
from vta_modules import *


def forward(self, input_data_list):
    input_data = input_data_list.permute(0, 2, 1)
    return self.network(input_data).permute(0, 2, 1)

def boxworld_vta():
    belief_size = 128
    return VTA(state_embed=StateEmbed(input_shape=boxworld.GRID_SIZE, out_dim=belief_size),
               action_embed=nn.Embedding(4, belief_size),
               belief_size=belief_size,
    )


class VTA(nn.Module):
    def __init__(self, state_embed, action_embed, belief_size):
        super().__init__()
        self.state_embed = state_embed
        self.action_embed = action_embed

        self.combine_action_obs = nn.Linear(
            self.action_encoder.embedding_size + self.enc_obs.embedding_size,
            belief_size,
        )

        self.prior_boundary = PriorBoundaryDetector(input_size=self.obs_feat_size)
        self.post_boundary = PostBoundaryDetector(input_size=self.feat_size,
                                                  num_layers=self.num_layers,
                                                  causal=True)

    def forward(self, states, actions, lengths, masks=None):
        return self.loss(states, actions, lengths, masks)


    def loss(self, states, actions, lengths, masks):
        """
        states: (B, max_T+1, s) tensor
        actions: (B, max_T,) tensor of ints
        lengths: T for each traj in the batch
        masks: (B, max_T) tensor of 1s and 0s. if there are T actions and T+1 states, then masks[:T] = 1 and masks[T:] = 0.

        returns: loss summed over B and T, no averaging (averaging is done in main.learn_options())
        """
        B, max_T = actions.shape
        assert_equal((B, max_T+1), states.shape[0:2])

        # ignore the last state
        states = states[:, :-1, :]

        x, a = states, actions
        x, a = self.state_embed(x), self.action_embed(a)
        a_concat_x = torch.cat([a, x], dim=-1)
        assert_shape(a_concat_x, (B, max_T, self.state_embed.out_dim + self.action_embed.out_dim))

        m_dist = self.boundary_posterior_decoder(a_concat_x)
        m = m_dist.sample()
        c = self.option_gru(a_concat_x)
        z_dist = ...
        z = self.generate_options(z_dist, m)
        s = torch.cat([c, z], dim=-1)
        assert_shape(s, (B, max_T, self.option_gru.hidden_dim + self.b))
        a_pred = self.action_decoder(s)

        reconstruction_loss = F.cross_entropy(a_pred, a, reduction='none')
        assert_shape(reconstruction_loss, (B, max_T))
        # if there are T actions and T+1 states, then masks[:T] = 1 and masks[T:] = 0.
        # reconstruction_loss = torch.sum(reconstruction_loss * masks, dim=1)



