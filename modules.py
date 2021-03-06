from typing import List, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from utils import assert_equal, DEVICE, assert_shape
import warnings


class Print(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        print(x)
        for p in self.parameters():
            print(p)
        return self.layer(x)


class MicroNet(nn.Module):
    def __init__(self, input_shape, input_channels=3, d=32, out_dim=4):
        super().__init__()
        self.input_channels = input_channels
        self.input_shape = input_shape
        self.d = d
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
        with warnings.catch_warnings():
            # UserWarning: Using padding='same' with even kernel lengths and odd
            # dilation may require a zero-padded copy of the input be created
            # ^^ those are annoying
            warnings.filterwarnings("ignore",category=UserWarning)

            # input: (N, C, H, W)
            (N, C, H, W) = x.shape
            # assert_equal(C, self.input_channels)
            # assert_equal((H, W), self.input_shape)

            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            # assert_equal(x.shape, (N, 12, H, W))

            x = einops.rearrange(x, 'n c h w -> n (c h w)')
            x = self.fc(x)
            return x


class RelationalMacroNet(nn.Module):
    def __init__(self, input_dim=32, d=32, num_attn_blocks=2, num_heads=4, out_dim=4, l=16):
        # pre attn linear has params input_dim * d * l, so be careful not to make too big
        super().__init__()
        self.d = d
        self.l = l
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.num_attn_blocks = num_attn_blocks
        self.pre_attn_linear = nn.Linear(input_dim, self.d * self.l)

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
                                nn.Linear(self.d, self.out_dim),)

    def forward(self, x):
        with warnings.catch_warnings():
            assert_equal(x.shape[1], self.input_dim)

            x = self.pre_attn_linear(x)
            x = einops.rearrange(x, 'b (l d) -> b l d', l=self.l, d=self.d)

            for _ in range(self.num_attn_blocks):
                x = x + self.attn_block(x, x, x, need_weights=False)[0]
                x = F.layer_norm(x, (self.d,))

            x = einops.reduce(x, 'n l d -> n d', 'max')
            x = self.fc(x)

            return x


class ShrinkingRelationalDRLNet(nn.Module):
    def __init__(self, input_channels=3, d=64, num_attn_blocks=2, num_heads=4, out_dim=4, shrink_loss_scale=1):
        super().__init__()
        self.input_channels = input_channels
        self.d = d
        self.out_dim = out_dim
        assert num_attn_blocks == 2
        self.conv1 = nn.Conv2d(input_channels, 24, 2, padding='same')
        # self.conv1_batchnorm = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(24, 24, 2, padding='same')
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
                                nn.Linear(self.d, self.out_dim),)

        self.num_layer_outs = 4
        self.ensemble = nn.Parameter(torch.ones(self.num_layer_outs, dtype=float))
        self.shrink_loss_scale = shrink_loss_scale

    def forward(self, x):
        with warnings.catch_warnings():
            # UserWarning: Using padding='same' with even kernel lengths and odd
            # dilation may require a zero-padded copy of the input be created
            # ^^ those are annoying
            warnings.filterwarnings("ignore",category=UserWarning)

            # input: (N, C, H, W)
            (N, C, H, W) = x.shape
            # assert_equal(C, self.input_channels)

            x = self.conv1(x)

            out1 = self.add_positions(x)
            out1 = einops.rearrange(out1, 'n c h w -> n (h w) c')
            out1 = self.pre_attn_linear(out1)
            out1 = einops.reduce(out1, 'n l d -> n d', 'max')
            out1 = self.fc(out1)

            x = self.conv2(x)
            x = F.relu(x)

            out2 = self.add_positions(x)
            out2 = einops.rearrange(out2, 'n c h w -> n (h w) c')
            out2 = self.pre_attn_linear(out2)
            out2 = einops.reduce(out2, 'n l d -> n d', 'max')
            out2 = self.fc(out2)

            x = self.add_positions(x)
            x = einops.rearrange(x, 'n c h w -> n (h w) c')
            x = self.pre_attn_linear(x)
            # assert_equal(x.shape, (N, H*W, self.d))

            # for _ in range(self.num_attn_blocks):
            x = x + self.attn_block(x, x, x, need_weights=False)[0]
            x = F.layer_norm(x, (self.d,))

            out3 = einops.reduce(x, 'n l d -> n d', 'max')
            out3 = self.fc(out3)

            x = x + self.attn_block(x, x, x, need_weights=False)[0]
            x = F.layer_norm(x, (self.d,))

            x = einops.reduce(x, 'n l d -> n d', 'max')
            x = self.fc(x)
            out4 = x

            weighting = F.softmax(self.ensemble, dim=0)
            out = out1 * weighting[0] + out2 * weighting[1] + out3 * weighting[2] + out4 * weighting[3]

            return out

    def shrink_loss(self):
        weighting = F.softmax(self.ensemble, dim=0)
        out = 0 * weighting[0] + 1 * weighting[1] + 2 * weighting[2] + 3 * weighting[3]
        return self.shrink_loss_scale * out

    def add_positions(self, inp):
        # input shape: (N, C, H, W)
        # output: (N, C+2, H, W)
        N, C, H, W = inp.shape
        # ranges between -1 and 1
        y_map = -1 + torch.arange(0, H + 0.01, H / (H - 1))/(H/2)
        x_map = -1 + torch.arange(0, W + 0.01, W / (W - 1))/(W/2)
        y_map, x_map = y_map.to(DEVICE), x_map.to(DEVICE)
        # assert_equal((x_map[-1], y_map[-1]), (1., 1.,))
        # assert_equal((x_map[0], y_map[0]), (-1., -1.,))
        # assert_equal(y_map.shape[0], H)
        # assert_equal(x_map.shape[0], W)
        x_map = einops.repeat(x_map, 'w -> n 1 h w', n=N, h=H)
        y_map = einops.repeat(y_map, 'h -> n 1 h w', n=N, w=W)
        # wonder if there could be a good way to do with einops
        inp = torch.cat((inp, x_map, y_map), dim=1)
        assert_equal(inp.shape, (N, C+2, H, W))
        return inp


class RelationalDRLNet(nn.Module):
    def __init__(self, input_channels=3, d=64, num_attn_blocks=2, num_heads=4, out_dim=4):
        super().__init__()
        self.input_channels = input_channels
        self.d = d
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
                                nn.Linear(self.d, self.out_dim),)

    def forward(self, x):
        with warnings.catch_warnings():
            # UserWarning: Using padding='same' with even kernel lengths and odd
            # dilation may require a zero-padded copy of the input be created
            # ^^ those are annoying
            warnings.filterwarnings("ignore",category=UserWarning)

            # input: (N, C, H, W)
            (N, C, H, W) = x.shape
            # assert_equal(C, self.input_channels)

            x = self.conv1(x)
            # x = self.conv1_batchnorm(x)
            x = self.conv2(x)
            # x = self.conv2_batchnorm(x)
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
        # assert_equal((x_map[-1], y_map[-1]), (1., 1.,))
        # assert_equal((x_map[0], y_map[0]), (-1., -1.,))
        # assert_equal(y_map.shape[0], H)
        # assert_equal(x_map.shape[0], W)
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
