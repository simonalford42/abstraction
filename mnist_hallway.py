import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader
import random
from typing import List, Any
import gym
import lenet


def get_mnist_sampler():
    mnist_path = '/Users/alfordsimon/data/MNIST'
    mnist_train = torchvision.datasets.MNIST(
        root=mnist_path,
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor())

    label_to_images = {i: [] for i in range(10)}
    for (image, label) in mnist_train:
        label_to_images[label].append(image)

    # return lambda d: (d, random.choice(label_to_images[d]))
    return lambda d: random.choice(label_to_images[d])
    # return lambda d: d


class DigitHall(gym.Env):
    def __init__(self, digits, init_pos, max_steps):
        super().__init__()
        self.digits = digits
        self.init_pos = init_pos
        self.max_steps = max_steps
        self.mnist_sampler = get_mnist_sampler()
        self.reset()

    def reset(self):
        self.digit_pics = [self.mnist_sampler(d) for d in self.digits]
        self.pos = self.init_pos
        self.i = 0
        return self.digit_pics[self.pos]

    def step(self, action):
        """
        returns (obs, rew, done, info)
        """
        assert action in [-1, +1]  # forward or backwards

        self.i += 1

        self.pos += action
        if self.pos < 0:
            self.pos = 0
        if self.pos == len(self.digits):
            self.pos = len(self.digits) - 1

        rew = self.digits[self.pos]
        obs = self.digit_pics[self.pos]
        done = self.i == self.max_steps
        info = {}

        return rew, obs, done, info


class CNN(nn.Module):
    """
    LeNet-5, achieves 99%+ on mnist
    """
    def __init__(self, out_dim):
        super().__init__()
        self.layer1_channels = 32
        self.layer2_channels = 32
        self.dense_hidden = 128
        self.out_dim = out_dim

        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=self.layer1_channels,
                               kernel_size=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=self.layer1_channels,
                               out_channels=self.layer2_channels,
                               kernel_size=3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.dense1 = nn.Linear(5 * 5 * self.layer2_channels, self.dense_hidden)
        self.dense2 = nn.Linear(self.dense_hidden, out_dim)

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


class TransitionNet(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = FC(dim + 1, dim, num_hidden=1, hidden_dim=64)

    def forward(self, x, actions):
        """
        x: batch of states (batch, abstract_dim)
        actions: batch of -1, 1 actions (batch, )
        """
        (B, d) = x.shape
        inp = torch.cat((x, torch.unsqueeze(actions, 1)), dim=1)
        assert inp.shape == (B, d + 1)

        return self.fc(inp)


class FC(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 num_hidden=1,
                 hidden_dim=512,
                 batch_norm=False):
        super().__init__()
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

    def forward(self, x):
        return self.net(x)


class MuZeroNet(nn.Module):
    def __init__(self, abstract_dim):
        super().__init__()
        self.abstract_dim = abstract_dim
        self.cnn = CNN(abstract_dim)
        self.omega_net = TransitionNet(abstract_dim)
        self.reward_net = FC(abstract_dim, 1, hidden_dim=64)
        self.double()

    def forward(self, state_batch, actions_batch):
        """
            state_batch: (B, 28, 28) batch of images
            action_batch: (B, ep_len) tensor of actions
            returns:
                rewards: (batch, seq,) list of predicted reward for each step
        """
        B, ep_len = actions_batch.shape

        # forget optimization for now
        state_batch = torch.unsqueeze(state_batch, 1)
        assert state_batch.shape == (B, 1, 28, 28)
        # abstract representation
        state_batch = self.cnn(state_batch)
        assert state_batch.shape == (B, self.abstract_dim)

        states = [state_batch]
        rews = []
        for action_i_batch in torch.transpose(actions_batch, 0, 1):
            state_batch = self.omega_net(state_batch, action_i_batch)
            rew_batch = self.reward_net(state_batch)
            states.append(state_batch)
            rew_batch = rew_batch.squeeze()
            rews.append(rew_batch)

        # go from list of length ep_len with (B, ) tensors to (B, ep_len)
        # tensor
        rews = torch.transpose(torch.stack(rews), 0, 1)
        assert rews.shape == (B, ep_len)
        return rews


def random_rollout(env):
    states = [env.reset()]
    rews = []
    actions = []
    done = False
    while not done:
        action = random.choice([-1, 1])
        rew, obs, done, _ = env.step(action)

        actions.append(action)
        rews.append(rew)
        states.append(obs)

    states = torch.cat(states)
    states = states.double()
    actions = torch.tensor(actions)
    actions = actions.double()
    rews = torch.tensor(rews)
    rews = rews.double()

    return (states, actions), rews


class RandomDataset(Dataset):
    def __init__(self, env, size=1000):
        self.env = env
        self.size = size

    def __len__(self):
        # arbitrary
        return self.size

    def __getitem__(self, idx):
        return random_rollout(self.env)


def train():
    net = MuZeroNet(abstract_dim=10)
    # doesn't seem to help.. surprising?
    # net.cnn.load_state_dict(torch.load('lenet.pt'))

    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    criterion = torch.nn.MSELoss()

    env = DigitHall([1, 2, 3, 4], init_pos=0, max_steps=8)

    dataset = RandomDataset(env, size=10000)
    dataloader = DataLoader(dataset, batch_size=32)

    for epoch in range(25):
        train_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()

            (state_batch, actions_batch), rews_batch = batch
            first_state_batch = state_batch[:, 0, :, :]
            pred_rews = net(first_state_batch, actions_batch)
            # pred_rews: (batch, ep_len)
            # rews_batch: (batch, ep_len)
            assert pred_rews.shape == rews_batch.shape

            loss = criterion(pred_rews, rews_batch)
            train_loss += loss
            loss.backward()
            optimizer.step()

        print(f"train_loss: {train_loss}")


def inspect_inputs():
    # test and make sure inputs look good
    env = DigitHall([1, 2, 3, 4], init_pos=0, max_steps=3)

    model = lenet.load_model('lenet.pt')

    dataset = RandomDataset(env)
    dataloader = DataLoader(dataset, batch_size=2)
    for batch in dataloader:
        (state_batch, action_batch), rews_batch = batch
        print(state_batch.shape)
        first_state_batch = state_batch[:, 0, :, :]
        print(first_state_batch.shape)
        print(action_batch.shape)
        print(rews_batch.shape)
        for batch in state_batch:
            digits = [lenet.pred(model, img) for img in batch]
            print(f"digits: {digits}")

        for a_batch, r_batch in zip(action_batch, rews_batch):
            for a, r in zip(a_batch, r_batch):
                print(f'a: {a}, r: {r}')

        assert False


def play_with_env():
    env = DigitHall([1, 2, 3, 4], init_pos=0, max_steps=3)
    done = False
    img = env.reset()

    model = lenet.load_model('lenet.pt')

    while not done:
        # action = random.choice([-1, 1])
        action = 1
        rew, img, done, _ = env.step(action)
        print(lenet.pred(model, img))


if __name__ == '__main__':
    train()
