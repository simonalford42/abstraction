import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import time

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1_channels = 32
        self.layer2_channels = 32
        self.dense_hidden = 128

        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=self.layer1_channels,
                               kernel_size=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=self.layer1_channels,
                               out_channels=self.layer2_channels,
                               kernel_size=3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.dense1 = nn.Linear(5 * 5 * self.layer2_channels, self.dense_hidden)
        self.dense2 = nn.Linear(self.dense_hidden, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool2(x)

        x = torch.flatten(x, start_dim=1)
        x = self.dense1(x)
        x = F.relu(x)
        x = self.dense2(x)
        return x

def pred(model, x):
    with torch.no_grad():
        model.eval()
        x = torch.unsqueeze(x, 0)
        x = torch.unsqueeze(x, 0)
        logits = model(x)
        preds = torch.argmax(logits, dim=1)
        return preds[0]

def load_model(path):
    model = LeNet()
    model.load_state_dict(torch.load(path))
    return model

