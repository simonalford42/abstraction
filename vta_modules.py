import torch
import torch.nn as nn
import torch.nn.functional as F
import einops


class StateEmbed(nn.Module):
    def __init__(self, input_shape, input_channels=10, d=128, out_dim=128, inter_channels=64):
        super().__init__()
        self.input_channels = input_channels
        self.input_shape = input_shape
        self.d = d
        self.out_dim = out_dim
        self.conv1 = nn.Conv2d(input_channels, inter_channels, 3)
        self.conv2 = nn.Conv2d(inter_channels, inter_channels, 3)

        h, w = input_shape
        self.fc = nn.Sequential(nn.Linear(inter_channels * (h-4) * (w-4), self.d),
                                nn.ReLU(),
                                nn.Linear(self.d, self.out_dim),
                                )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = einops.rearrange(x, 'n c h w -> n (c h w)')
        x = self.fc(x)
        return x


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
