from typing import List, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from utils import assert_equal, DEVICE


class Print(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        print(x)
        for p in self.parameters():
            print(p)
        return self.layer(x)


def abstract_out_dim(a, b):
    # a * b for action probs, 2 * b for stop probs, b for start probs
    return a * b + 2 * b + b


class MicroNet(nn.Module):
    def __init__(self, input_shape, input_channels=3, d=32, out_dim=4):
        super().__init__()
        self.input_channels = input_channels
        self.input_shape = input_shape
        self.d = 64
        self.out_dim = out_dim
        self.conv1 = nn.Conv2d(input_channels, 12, 3, padding='same')
        self.conv2 = nn.Conv2d(12, 12, 3, padding='same')
        self.fc = nn.Sequential(nn.Linear(12 * np.prod(input_shape), self.d),
                                nn.ReLU(),
                                # nn.BatchNorm1d(self.d),
                                nn.Linear(self.d, self.d),
                                nn.ReLU(),
                                # nn.BatchNorm1d(self.d),
                                nn.Linear(self.d, self.out_dim)
                                )

    def forward(self, x):
        # input: (N, C, H, W)
        (N, C, H, W) = x.shape
        assert_equal(C, self.input_channels)
        assert_equal((H, W), self.input_shape)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        assert_equal(x.shape, (N, 12, H, W))

        x = einops.rearrange(x, 'n c h w -> n (c h w)')
        x = self.fc(x)
        return x


store_debug = {}

def store(N, x, s):
    if N == 2:
        store_debug[s + '2'] = x
    else:
        store_debug[s] = x


def debug():
    for key in store_debug:
        if key[-1] != '2':
            print(key, torch.sum(store_debug[key][0] - store_debug[key + '2'][0]))


class DebugNet(nn.Module):
    def __init__(self, input_channels=3, d=64, num_attn_blocks=2, num_heads=4, out_dim=4):
        super().__init__()
        self.input_channels = input_channels
        self.d = 32
        self.out_dim = out_dim
        self.num_attn_blocks = num_attn_blocks

        # 2 exra dims for positional encoding
        self.pre_attn_linear = nn.Linear(24 + 2, self.d)

        # shared weights, so just one network
        self.attn_block = nn.MultiheadAttention(embed_dim=self.d,
                                                num_heads=num_heads,
                                                batch_first=True)

        self.fc = nn.Sequential(nn.Linear(self.d, self.d),
                                nn.ReLU(),
                                # nn.BatchNorm1d(self.d),
                                nn.Linear(self.d, self.d),
                                nn.ReLU(),
                                # nn.BatchNorm1d(self.d),
                                nn.Linear(self.d, self.d),
                                nn.ReLU(),
                                # nn.BatchNorm1d(self.d),
                                nn.Linear(self.d, self.d),
                                nn.ReLU(),
                                nn.Linear(self.d, self.out_dim),
                                )

    def add_positions(self, inp):
        # input shape: (N, C, H, W)
        # output: (N, C+2, H, W)
        N, C, H, W = inp.shape
        # ranges between -1 and 1
        y_map = -1 + torch.arange(0, H + 0.01, H / (H - 1))/(H/2)
        x_map = -1 + torch.arange(0, W + 0.01, W / (W - 1))/(W/2)
        y_map, x_map = y_map.to(DEVICE), x_map.to(DEVICE)
        assert_equal((x_map[-1], y_map[-1]), (1., 1.,))
        assert_equal((x_map[0], y_map[0]), (-1., -1.,))
        assert_equal(y_map.shape[0], H)
        assert_equal(x_map.shape[0], W)
        x_map = einops.repeat(x_map, 'w -> n 1 h w', n=N, h=H)
        y_map = einops.repeat(y_map, 'h -> n 1 h w', n=N, w=W)
        # wonder if there could be a good way to do with einops
        inp = torch.cat((inp, x_map, y_map), dim=1)
        assert_equal(inp.shape, (N, C+2, H, W))
        return inp

    def forward(self, x):
        # input: (N, C, H, W)
        (N, C, H, W) = x.shape

        x = self.add_positions(x)
        x = einops.rearrange(x, 'n c h w -> n (h w) c')
        print(f'x1: {x.shape}')
        store(N, x, 'pos')
        x = self.pre_attn_linear(x)
        print(f'x2: {x.shape}')
        store(N, x, 'pre_attn')
        assert_equal(x.shape, (N, H*W, self.d))

        for _ in range(self.num_attn_blocks):
            x = x + self.attn_block(x, x, x, need_weights=False)[0]
            x = F.layer_norm(x, (self.d,))
            assert_equal(x.shape, (N, H*W, self.d))

        x = einops.reduce(x, 'n l d -> n d', 'max')
        x = self.fc(x)

        return x


class RelationalDRLNet(nn.Module):
    def __init__(self, input_channels=3, d=64, num_attn_blocks=2, num_heads=4, out_dim=4):
        super().__init__()
        self.input_channels = input_channels
        self.d = 64
        self.out_dim = out_dim
        self.num_attn_blocks = num_attn_blocks
        self.conv1 = nn.Conv2d(input_channels, 12, 2, padding='same')
        # self.conv1_batchnorm = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(12, 24, 2, padding='same')
        # self.conv2_batchnorm = nn.BatchNorm2d(24)

        # 2 exra dims for positional encoding
        self.pre_attn_linear = nn.Linear(24 + 2, self.d)

        # shared weights, so just one network
        self.attn_block = nn.MultiheadAttention(embed_dim=self.d,
                                                num_heads=num_heads,
                                                batch_first=True)

        self.fc = nn.Sequential(nn.Linear(self.d, self.d),
                                nn.ReLU(),
                                # nn.BatchNorm1d(self.d),
                                nn.Linear(self.d, self.d),
                                nn.ReLU(),
                                # nn.BatchNorm1d(self.d),
                                nn.Linear(self.d, self.d),
                                nn.ReLU(),
                                # nn.BatchNorm1d(self.d),
                                nn.Linear(self.d, self.d),
                                nn.ReLU(),
                                nn.Linear(self.d, self.out_dim),
                                )

    def forward(self, x):
        # input: (N, C, H, W)
        (N, C, H, W) = x.shape
        assert_equal(C, self.input_channels)

        x = self.conv1(x)
        # x = self.conv1_batchnorm(x)
        x = self.conv2(x)
        # x = self.conv2_batchnorm(x)
        x = F.relu(x)
        assert_equal(x.shape[-2:], (H, W))

        x = self.add_positions(x)
        x = einops.rearrange(x, 'n c h w -> n (h w) c')
        print(f'x1: {x.shape}')
        store(N, x, 'pos')
        x = self.pre_attn_linear(x)
        print(f'x2: {x.shape}')
        store(N, x, 'pre_attn')
        assert_equal(x.shape, (N, H*W, self.d))

        for _ in range(self.num_attn_blocks):
            x = x + self.attn_block(x, x, x, need_weights=False)[0]
            x = F.layer_norm(x, (self.d,))
            assert_equal(x.shape, (N, H*W, self.d))

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
        assert_equal((x_map[-1], y_map[-1]), (1., 1.,))
        assert_equal((x_map[0], y_map[0]), (-1., -1.,))
        assert_equal(y_map.shape[0], H)
        assert_equal(x_map.shape[0], W)
        x_map = einops.repeat(x_map, 'w -> n 1 h w', n=N, h=H)
        y_map = einops.repeat(y_map, 'h -> n 1 h w', n=N, w=W)
        # wonder if there could be a good way to do with einops
        inp = torch.cat((inp, x_map, y_map), dim=1)
        assert_equal(inp.shape, (N, C+2, H, W))
        return inp


class ResBlock(nn.Module):
    def __init__(self, conv1_filters, conv2_filters):
        super().__init__()
        self.conv1 = nn.Conv2d(conv1_filters, conv2_filters, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(conv1_filters)
        self.conv2 = nn.Conv2d(conv2_filters, conv2_filters, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(conv2_filters)
        # self.finalize()

    def forward(self, x):
        # x = self.tensor(x)
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = F.relu(out)
        return out


class OneByOneBlock(nn.Module):
    def __init__(self, in_filters, out_filters):
        super().__init__()
        self.conv = nn.Conv2d(in_filters, out_filters, 1)
        self.bn = nn.BatchNorm2d(out_filters)
        # self.finalize()

    def forward(self, x):
        # x = self.tensor(x)
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x


class FC(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 num_hidden=1,
                 hidden_dim=512,
                 batch_norm=False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        layers: List[Any]
        if num_hidden == 0:
            layers = [nn.Linear(input_dim, output_dim)]
        else:
            layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            for i in range(num_hidden - 1):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
                if batch_norm:
                    layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)
        # self.finalize()

    def forward(self, x):
        # x = self.tensor(x)
        return self.net(x)


class AllConv(nn.Module):
    def __init__(self,
                 residual_blocks=1,
                 input_filters=10,
                 residual_filters=10,
                 conv_1x1s=2,
                 output_dim=128,
                 conv_1x1_filters=128,
                 pooling='max'):
        super().__init__()

        self.conv1 = nn.Conv2d(input_filters, residual_filters, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(residual_filters)

        self.res_blocks = nn.ModuleList([
            ResBlock(residual_filters, residual_filters)
            for i in range(residual_blocks)
        ])

        filters_1x1 = [residual_filters] + [conv_1x1_filters] * (conv_1x1s - 1) + [output_dim]

        self.conv_1x1s = nn.ModuleList([
            OneByOneBlock(filters_1x1[i], filters_1x1[i + 1])
            for i in range(conv_1x1s)
        ])

        self.pooling = pooling
        # self.finalize()

    def forward(self, x):
        """
        Input: float tensor of shape (batch_size, n_filters, w, h)
        Output: float tensor of shape (batch_size, output_dim)
        """
        # x = self.tensor(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        for block in self.res_blocks:
            x = block(x)

        for c in self.conv_1x1s:
            x = c(x)

        if self.pooling == 'max':
            x = F.max_pool2d(x, kernel_size=x.size()[2:])
        else:
            x = F.avg_pool2d(x, kernel_size=x.size()[2:])

        x = x.squeeze(3).squeeze(2)
        return x


class ImageFC(nn.Module):
    def __init__(self, inp_shape, fc_net: FC):
        super().__init__()
        self.fc_net = fc_net
        self.first_layer = nn.Linear(in_features=np.prod(inp_shape),
                                     out_features=fc_net.input_dim)

    def forward(self, x):
        x = einops.rearrange(x, 'n c h w -> n (c h w)')
        x = self.first_layer(x)
        x = F.relu(x)
        x = self.fc_net(x)
        return x
