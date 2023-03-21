from math import log
import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Bernoulli, kl_divergence
import warnings
import einops
from bw_utils import assert_equal, DEVICE, assert_shape


class Flatten(nn.Module):
    def forward(self, input_data):
        if len(input_data.size()) == 4:
            return input_data.view(input_data.size(0), -1)
        else:
            return input_data.view(input_data.size(0), input_data.size(1), -1)


class LinearLayer(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 nonlinear=nn.ELU(inplace=True)):
        super(LinearLayer, self).__init__()
        # linear
        self.linear = nn.Linear(in_features=input_size,
                                out_features=output_size)

        # nonlinear
        self.nonlinear = nonlinear

    def forward(self, input_data):
        return self.nonlinear(self.linear(input_data))


class ConvLayer1D(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 causal=False,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 normalize=True,
                 nonlinear=nn.ELU(inplace=True)):
        super(ConvLayer1D, self).__init__()
        # linear
        Conv = CausalConv1d if causal else nn.Conv1d
        self.linear = Conv(in_channels=input_size,
                           out_channels=output_size,
                           kernel_size=kernel_size,
                           stride=stride,
                           padding=padding,
                           bias=False if normalize else True)
        if normalize:
            self.normalize = nn.BatchNorm1d(num_features=output_size)
        else:
            self.normalize = nn.Identity()

        # nonlinear
        self.nonlinear = nonlinear

    def forward(self, input_data):
        return self.nonlinear(self.normalize(self.linear(input_data)))


class ConvLayer2D(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 normalize=True,
                 nonlinear=nn.ELU(inplace=True)):
        super(ConvLayer2D, self).__init__()
        # linear
        self.linear = nn.Conv2d(in_channels=input_size,
                                out_channels=output_size,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                bias=False if normalize else True)
        if normalize:
            self.normalize = nn.BatchNorm2d(num_features=output_size)
        else:
            self.normalize = nn.Identity()

        # nonlinear
        self.nonlinear = nonlinear

    def forward(self, input_data):
        return self.nonlinear(self.normalize(self.linear(input_data)))


class ConvTransLayer2D(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 kernel_size=4,
                 stride=2,
                 padding=1,
                 normalize=True,
                 nonlinear=nn.ELU(inplace=True)):
        super(ConvTransLayer2D, self).__init__()
        # linear
        self.linear = nn.ConvTranspose2d(in_channels=input_size,
                                         out_channels=output_size,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         bias=False if normalize else True)
        if normalize:
            self.normalize = nn.BatchNorm2d(num_features=output_size)
        else:
            self.normalize = nn.Identity()

        # nonlinear
        self.nonlinear = nonlinear

    def forward(self, input_data):
        return self.nonlinear(self.normalize(self.linear(input_data)))


class RecurrentLayer(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size):
        super(RecurrentLayer, self).__init__()
        # rnn cell
        self.rnn_cell = nn.GRUCell(input_size=input_size,
                                   hidden_size=hidden_size)

    def forward(self, input_data, prev_state):
        return self.rnn_cell(input_data, prev_state)


class LatentDistribution(nn.Module):
    def __init__(self,
                 input_size,
                 latent_size,
                 feat_size=None):
        super(LatentDistribution, self).__init__()
        if feat_size is None:
            self.feat = nn.Identity()
            feat_size = input_size
        else:
            self.feat = LinearLayer(input_size=input_size,
                                    output_size=feat_size)

        self.mean = LinearLayer(input_size=feat_size,
                                output_size=latent_size,
                                nonlinear=nn.Identity())

        self.std = LinearLayer(input_size=feat_size,
                               output_size=latent_size,
                               nonlinear=nn.Sigmoid())

    def forward(self, input_data):
        feat = self.feat(input_data)
        return Normal(loc=self.mean(feat), scale=self.std(feat))

#################################################################
#################################################################

class DiscreteLatentDistributionVQ(nn.Module):
    def __init__(self,
                 input_size,
                 latent_n=10,
                 commitment_cost=1.0,
                 feat_size=None):
        super(DiscreteLatentDistributionVQ, self).__init__()
        self.latent_n = latent_n
        if feat_size is None:
            self.feat = nn.Identity()
            feat_size = input_size
        self.feat = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, feat_size)
        )
        self.beta = 1.0
        self._commitment_cost = commitment_cost
        self.code_book = nn.parameter.Parameter(torch.zeros(latent_n, feat_size))
        self.code_book.data.uniform_(-1/latent_n, 1/latent_n)

    def forward(self, input_data):
        z_embedding = self.feat(input_data)
        # Calculate distances
        distances = (torch.sum(z_embedding**2, dim=1, keepdim=True)
                    + torch.sum(self.code_book**2, dim=1)
                    - 2 * torch.matmul(z_embedding, self.code_book.t()))
        # Encoding
        probs = torch.softmax(-distances/0.1, dim=-1)
        multi_dist = torch.distributions.Categorical(probs)
        encoding_indices = multi_dist.sample().unsqueeze(1)

        encodings = torch.zeros(encoding_indices.shape[0], self.latent_n, device=input_data.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.code_book)  # (B, feat_size)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), z_embedding)
        q_latent_loss = F.mse_loss(quantized, z_embedding.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = z_embedding + (quantized - z_embedding).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, quantized, perplexity, encodings, -distances

    def z_embedding(self, z_index):
        """Returns e(z) (shape (num_options,)) given z (int)."""
        one_hot_z = torch.zeros(self.latent_n, device=self.code_book.device)
        one_hot_z[z_index] = 1
        return torch.matmul(one_hot_z, self.code_book)


class DiscreteLatentDistribution(nn.Module):
    def __init__(self,
                 input_size,
                 latent_size,
                 latent_n,
                 feat_size=None):
        super(DiscreteLatentDistribution, self).__init__()
        self.latent_n = latent_n
        if feat_size is None:
            self.feat = nn.Identity()
            feat_size = input_size
        else:
            self.feat = LinearLayer(input_size=input_size,
                                    output_size=feat_size)
        self.beta = 1.0
        self.log_p = LinearLayer(input_size=feat_size,
                                 output_size=latent_n,
                                 nonlinear=nn.Identity())

    def forward(self, input_data, is_training=True):
        feat = self.feat(input_data)
        log_p = self.log_p(feat)
        return STCategorical(log_p, self.beta, is_training)


class STCategorical:

    def __init__(self, log_p, beta, is_training):
        self.log_p = log_p
        self.n = log_p.shape[1]
        self.beta = beta
        self.is_training = is_training

    def rsample(self):
        if self.is_training:
            log_sample_p = utils.gumbel_sampling(
                    log_alpha=self.log_p, temp=self.beta)
        else:
            log_sample_p = self.log_p / self.beta
        # probability
        log_sample_p = log_sample_p - torch.logsumexp(log_sample_p, dim=-1, keepdim=True)
        sample_prob = log_sample_p.exp()
        sample_data = torch.eye(self.n, dtype=self.log_p.dtype, device=self.log_p.device)[torch.max(sample_prob, dim=-1)[1]]
        # sample with rounding and st-estimator
        sample_data = sample_data.detach() + (sample_prob - sample_prob.detach())
        # return sample data and logit
        return sample_data


def kl_categorical(a, b, mask_a=False, mask_b=False):
    """Compute the KL-divergence between two STCategorical Distributions."""
    log_a, log_b = a.log_p, b.log_p
    if mask_a:
        log_a = log_a.detach()
    if mask_b:
        log_b = log_b.detach()
    softmax_a = F.softmax(log_a, dim=-1)
    logsumexp_a, logsumexp_b = torch.logsumexp(log_a, -1), torch.logsumexp(log_b, -1)
    product = (softmax_a * (log_a - log_b)).sum(axis=-1)
    return product + logsumexp_b - logsumexp_a


#################################################################
#################################################################


class Encoder(nn.Module):
    def __init__(self,
                 output_size=None,
                 feat_size=64):
        super(Encoder, self).__init__()
        network_list = [ConvLayer2D(input_size=3,
                                    output_size=feat_size,
                                    kernel_size=4,
                                    stride=2,
                                    padding=1),  # 16 x 16
                        ConvLayer2D(input_size=feat_size,
                                    output_size=feat_size,
                                    kernel_size=4,
                                    stride=2,
                                    padding=1),  # 8 x 8
                        ConvLayer2D(input_size=feat_size,
                                    output_size=feat_size,
                                    kernel_size=4,
                                    stride=2,
                                    padding=1),  # 4 x 4
                        ConvLayer2D(input_size=feat_size,
                                    output_size=feat_size,
                                    kernel_size=4,
                                    stride=1,
                                    padding=0),  # 1 x 1
                        Flatten()]
        if output_size is not None:
            network_list.append(LinearLayer(input_size=feat_size,
                                            output_size=output_size))
            self.output_size = output_size
        else:
            self.output_size = feat_size
        self.embedding_size = self.output_size
        self.network = nn.Sequential(*network_list)

    def forward(self, input_data):
        return self.network(input_data)


class StateIndependentEncoder(Encoder):

    def forward(self, input_data):
        shape = list(input_data.shape)
        return input_data.new_zeros((shape[0], shape[1], self.output_size))


class StateDependentEncoder(Encoder):

    def forward(self, input_data):
        shape = list(input_data.shape) # (B, S, H, W, C)
        input_data = input_data.reshape([-1] + shape[2:])  # (B x S, H, W, C)
        input_data = input_data.permute(0, 3, 1, 2)  # (B x S, C, H, W)
        out = super().forward(input_data)
        return out.reshape(shape[:2] + [-1])


class CompILEGridEncoder(nn.Module):
    """Embedder for states in CompILEEnv."""

    def __init__(self, input_dim=12, input_shape=(10,10), output_dim=128, feat_size=64):
        super().__init__()
        h, w = input_shape
        network_list = [ConvLayer2D(input_size=input_dim,
                                    output_size=feat_size,
                                    kernel_size=3,
                                    stride=1,
                                    padding=0,
                                    nonlinear=nn.ReLU()),  # 6x6
                        ConvLayer2D(input_size=feat_size,
                                    output_size=feat_size,
                                    kernel_size=3,
                                    stride=1,
                                    padding=0,
                                    nonlinear=nn.ReLU()),  # 4x4
                        Flatten(),
                        nn.Linear((h-4) * (w-4) * feat_size, output_dim),
                        nn.ReLU(),
                        nn.Linear(output_dim, output_dim),
                       ]
        self.output_size = output_dim
        self.embedding_size = self.output_size
        self.network = nn.Sequential(*network_list)

    def forward(self, input_data):
        # (B, S, W, H, C) input_data
        shape = list(input_data.shape)
        input_data = input_data.reshape([-1] + shape[2:])  # (B x S, W, H, C)
        input_data = input_data.permute(0, 3, 2, 1)  # (B x S, C, H, W)
        out = self.network(input_data)  # (B x S, embed_dim)
        return out.reshape(shape[:2] + [-1])

        # (B, W, H, C) input_data
        input_data = input_data.permute(0, 3, 2, 1)  # (B, C, H ,W)
        return self.network(input_data)  # (B, embed_dim)


class MiniWorldEncoder(nn.Module):
    """Embedder for states in CompILEEnv."""

    def __init__(self, input_dim=3, output_dim=128):
        super().__init__()
        self._network = nn.Sequential(
                nn.Conv2d(input_dim, 32, kernel_size=5, stride=2),
                nn.ReLU(),

                nn.Conv2d(32, 32, kernel_size=5, stride=2),
                nn.ReLU(),

                nn.Conv2d(32, 32, kernel_size=4, stride=2),
                nn.ReLU(),

                Flatten(),

                nn.Linear(32 * 7 * 5, output_dim),
        )
        self.output_size = output_dim
        self.embedding_size = self.output_size

    def forward(self, input_data):
        # (B, S, W, H, C) input_data
        shape = list(input_data.shape)
        input_data = input_data.reshape([-1] + shape[2:])  # (B x S, W, H, C)
        input_data = input_data.permute(0, 3, 2, 1)  # (B x S, C, H, W)
        out = self._network(input_data)  # (B x S, embed_dim)
        return out.reshape(shape[:2] + [-1])


class GridActionEncoder(nn.Module):
    """Embedder for actions in SimpleGridEnv."""

    def __init__(self, action_size, embedding_size):
        super().__init__()
        self.action_size = action_size
        self.embedding_size = embedding_size
        self._embedder = nn.Embedding(action_size, embedding_size)

    def forward(self, x):
        return self._embedder(x)


class GridEncoder(nn.Module):
    """Embedder for SimpleGridEnv states.
    Concretely, embeds (x, y) separately with different embeddings for each cell.
    """

    def __init__(self, embed_dim=64):
        """Constructs for SimpleGridEnv.
        Args:
        observation_space (spaces.Box): limits for the observations to embed.
        """
        super().__init__()

        hidden_size = 32
        self.embedding_size = embed_dim
        self._embedders = nn.ModuleList(
            [nn.Embedding(dim, hidden_size) for dim in [9, 6, 4]]
        )
        self._fc_layer = nn.Linear(hidden_size * 3, 256)
        self._final_fc_layer = nn.Linear(256, embed_dim)

    def forward(self, obs):
        tensor = obs
        embeds = []
        for i in range(tensor.shape[2]):
            embeds.append(self._embedders[i](tensor[:, :, i]))
        return self._final_fc_layer(F.relu(self._fc_layer(torch.cat(embeds, -1))))


class Decoder(nn.Module):
    def __init__(self,
                 input_size,
                 feat_size=64):
        super(Decoder, self).__init__()
        if input_size == feat_size:
            self.linear = nn.Identity()
        else:
            self.linear = LinearLayer(input_size=input_size,
                                      output_size=feat_size,
                                      nonlinear=nn.Identity())
        self.network = nn.Sequential(ConvTransLayer2D(input_size=feat_size,
                                                      output_size=feat_size,
                                                      kernel_size=4,
                                                      stride=1,
                                                      padding=0),
                                     ConvTransLayer2D(input_size=feat_size,
                                                      output_size=feat_size),
                                     ConvTransLayer2D(input_size=feat_size,
                                                      output_size=feat_size),
                                     ConvTransLayer2D(input_size=feat_size,
                                                      output_size=3,
                                                      normalize=False,
                                                      nonlinear=nn.Tanh()))

    def forward(self, input_data):
        return self.network(self.linear(input_data).unsqueeze(-1).unsqueeze(-1))


class GridDecoder(nn.Module):
    def __init__(self, input_size, action_size, feat_size=64):
        super().__init__()
        if input_size == feat_size:
            self.linear = nn.Identity()
        else:
            self.linear = LinearLayer(input_size=input_size,
                                      output_size=feat_size,
                                      nonlinear=nn.Identity())
        self.network = nn.Sequential(
            nn.Linear(feat_size, feat_size),
            nn.ReLU(),
            nn.Linear(feat_size, feat_size),
            nn.ReLU(),
            nn.Linear(feat_size, feat_size),
            nn.ReLU(),
            nn.Linear(feat_size, action_size),
        )

    def forward(self, input_data):
        return self.network(self.linear(input_data))


class PriorBoundaryDetector(nn.Module):
    def __init__(self,
                 input_size,
                 output_size=2):
        super(PriorBoundaryDetector, self).__init__()
        self.network = LinearLayer(input_size=input_size,
                                   output_size=output_size,
                                   nonlinear=nn.Identity())

    def forward(self, input_data):
        logit_data = self.network(input_data)
        return logit_data


class PostBoundaryDetector(nn.Module):
    def __init__(self,
                 input_size,
                 output_size=2,
                 num_layers=1,
                 causal=False):
        super(PostBoundaryDetector, self).__init__()
        network = list()
        for l in range(num_layers):
            network.append(ConvLayer1D(input_size=input_size,
                                       output_size=input_size,
                                       kernel_size=5,
                                       causal=causal,
                                       padding=2))
        network.append(ConvLayer1D(input_size=input_size,
                                   output_size=output_size,
                                   causal=causal,
                                   normalize=False,
                                   nonlinear=nn.Identity()))
        self.network = nn.Sequential(*network)

    def forward(self, input_data_list):
        input_data = input_data_list.permute(0, 2, 1)
        return self.network(input_data).permute(0, 2, 1)


class CausalConv1d(torch.nn.Conv1d):
    """
    https://github.com/pytorch/pytorch/issues/1333
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 padding=1,
                 bias=True):
        self.__padding = (kernel_size - 1) * dilation
        assert stride == 1
        assert padding == (kernel_size - 1) / 2
        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias)

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)
        if self.__padding != 0:
            return result[:, :, :-self.__padding]
        return result


class AttentionEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.relational_net = RelationalDRLNet()

    def forward(self, obs):
        obs = obs.permute(0, 3, 1, 2)
        return self.relational_net(obs)


class RelationalDRLNet(nn.Module):
    def __init__(self, input_channels=3, d=64, num_attn_blocks=2, num_heads=4, out_dim=4):
        super().__init__()
        self.input_channels = input_channels
        self.d = d
        self.out_dim = out_dim
        self.num_attn_blocks = num_attn_blocks
        self.conv1 = nn.Conv2d(input_channels, 12, 2, padding='same')
        self.conv2 = nn.Conv2d(12, 24, 2, padding='same')

        # 2 exra dims for positional encoding
        self.pre_attn_linear = nn.Linear(24 + 2, self.d)

        # shared weights, so just one network
        self.attn_block = nn.MultiheadAttention(embed_dim=self.d,
                                                num_heads=num_heads,
                                                batch_first=True)

        self.fc = nn.Sequential(nn.Linear(self.d, self.d),
                                nn.ReLU(),
                                nn.Linear(self.d, self.d),
                                nn.ReLU(),
                                nn.Linear(self.d, self.d),
                                nn.ReLU(),
                                nn.Linear(self.d, self.d),
                                nn.ReLU(),
                                nn.Linear(self.d, self.out_dim),)

    def forward(self, x):
        # filter warnings about padding copy for conv2d
        with warnings.catch_warnings():
            # input: (N, C, H, W)
            (N, C, H, W) = x.shape
            # assert_equal(C, self.input_channels)

            x = self.conv1(x)
            x = self.conv2(x)
            x = F.relu(x)
            # assert_equal(x.shape[-2:], (H, W))

            x = self.add_positions(x)
            x = einops.rearrange(x, 'n c h w -> n (h w) c')
            x = self.pre_attn_linear(x)
            # assert_equal(x.shape, (N, H*W, self.d))

            for _ in range(self.num_attn_blocks):
                x = x + self.attn_block(x, x, x, need_weights=False)[0]
                x = F.layer_norm(x, (self.d,))
                # assert_equal(x.shape, (N, H*W, self.d))

            x = einops.reduce(x, 'n l d -> n d', 'max')
            x = self.fc(x)

            return x

    def add_positions(self, inp):
        # input shape: (N, C, H, W)
        # output: (N, C+2, H, W)
        N, C, H, W = inp.shape
        # ranges between -1 and 1
        y_map = -1 + torch.arange(0, H + 0.01, H / (H - 1))/(H/2)
        x_map = -1 + torch.arange(0, W + 0.01, W / (W - 1))/(W/2)
        y_map, x_map = y_map.to(DEVICE), x_map.to(DEVICE)
        x_map = einops.repeat(x_map, 'w -> n 1 h w', n=N, h=H)
        y_map = einops.repeat(y_map, 'h -> n 1 h w', n=N, w=W)

        inp = torch.cat((inp, x_map, y_map), dim=1)
        assert_equal(inp.shape, (N, C+2, H, W))
        return inp

