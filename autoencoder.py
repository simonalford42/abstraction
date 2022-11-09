import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self, shape, dimension, loss="bce", stride=1, hidden=32, layers=2, discrete=False):
        """
        loss: bce, categorical, normal
        """
        super().__init__()
        # figure out what the shape means for us
        if len(shape) == 3:
            self.convolutional=True
        else:
            if loss == "categorical":
                assert len(shape) == 2
            else:
                assert len(shape) == 1

            self.convolutional=False

        self.hidden, self.loss, self.discrete = hidden, loss, discrete

        if discrete:
            # try to split up the dimension roughly equally between variables and categories
            self.n_variables, self.n_categories = dimension//2, 2 #int(dimension**0.5+0.5), int(dimension**0.5)
            dimension = self.n_variables * self.n_categories
            self.steps = 1

        if self.convolutional:
            image_channels = shape[0]
            e_layers = []
            for layer in range(layers):
                e_layers.append(nn.Conv2d(image_channels if layer == 0 else hidden,
                                          hidden, kernel_size=3, padding=1, stride=stride))
                e_layers.append(nn.ReLU())
            self.encoder = nn.Sequential(*e_layers)

            test = torch.zeros(1, *shape)
            output_shape = self.encoder(test).shape
            output_dimension = output_shape[1]*output_shape[2]*output_shape[3]
            self.fc1 = nn.Linear(output_dimension, dimension)

            decoder_layers = 2
            up_sampling_factor = stride**decoder_layers
            output_resolution = shape[1]
            self.input_resolution = output_resolution//up_sampling_factor

            self.fc2 = nn.Linear(dimension, hidden*self.input_resolution*self.input_resolution)
            d_layers = []
            for layer in range(layers):
                d_layers.append(nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, stride=stride))
                d_layers.append(nn.ReLU())
            self.decoder = nn.Sequential(nn.ReLU(), *d_layers)
        else:
            e_layers = []
            for layer in range(layers):
                e_layers.extend([nn.Linear(np.prod(shape) if layer == 0 else hidden, hidden),
                                 nn.ReLU()])
            self.encoder = nn.Sequential(*e_layers)
            self.fc = nn.Linear(hidden, dimension)
            d_layers = []
            for layer in range(layers):
                d_layers.extend([nn.Linear(dimension if layer == 0 else hidden,
                                           np.prod(shape) if layer == layers-1 else hidden),
                                 nn.ReLU()])
            self.decoder = nn.Sequential(*d_layers[: -1])

    @property
    def temperature(self):
        r = 2*1e-4
        return max(0.5, math.exp(-r*self.steps))
    
    def encode(self, x):
        B = x.shape[0]
        if self.convolutional:
            z = self.fc1(self.encoder(x).view(B,-1))
        else:
            z = self.fc(self.encoder(x))
        z_original = z # before adding noise
        if self.discrete:
            self.steps+=1
            z = z.view(B, self.n_variables, self.n_categories)
            z_original = torch.softmax(z,-1).view(B, -1)
            z = F.gumbel_softmax(z,
                                 tau=self.temperature).view(B, -1)
        else:
            z = z.clamp(min=0)

        return z, z_original

    def decode(self, z):
        if self.convolutional:
            xh = self.decoder(self.fc2(z).view(z.shape[0], self.hidden, self.input_resolution, self.input_resolution))
        else:
            xh = self.decoder(z)
        if self.loss == "bce":
            xh = torch.sigmoid(xh)
        return xh

    def log_likelihood(self, x, xh):
        if self.loss == "normal":
            return -((x-xh)**2).sum()/x.shape[0]
        if self.loss == "bce":
            return -nn.BCELoss(reduction="none")(xh, x).sum()/x.shape[0]
        assert False

    def forward(self, x):
        """returns (z, log likelihood)"""
        z, z_original = self.encode(x)
        xh = self.decode(z)
        ll = self.log_likelihood(x, xh)
        return z_original, ll
        
        


if __name__ == '__main__':
    m = Autoencoder([64], dimension=128, loss="normal", stride=1)
    print(m.log_likelihood(torch.zeros(2,64)))
    print(m.encode(torch.zeros(2,64)).shape)
    print(m.decode(m.encode(torch.zeros(2,64))).shape)
            
                
            
