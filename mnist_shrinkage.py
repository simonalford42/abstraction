from typing import Any, List
from utils import assert_equal, assert_shape
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import time


def mnist_data():
    transform = torchvision.transforms.ToTensor()
    # make a new dir just in case it's somehow different from last time..
    mnist_path = '/Users/alfordsimon/data/MNIST_cs6787'
    mnist_train = torchvision.datasets.MNIST(root=mnist_path, train=True, transform=transform, download=True)
    mnist_test = torchvision.datasets.MNIST(root=mnist_path, train=False, transform=transform)
    return mnist_train, mnist_test


class ShrinkFC(nn.Module):
    def __init__(self, hidden_dim=10, out_dim=10, in_dim=28*28*1, shrink_loss_scale=1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, out_dim)
        self.ensemble = nn.Parameter(torch.ones(4, dtype=float))
        self.shrink_loss_scale = shrink_loss_scale

    def forward(self, x, out):
        N, *s = x.shape
        x = x.reshape(N, -1)
        assert_shape(x, (N, self.in_dim))

        x = F.relu(self.fc1(x))
        out1 = self.fc5(x)
        x = F.relu(self.fc2(x))
        out2 = self.fc5(x)
        x = F.relu(self.fc3(x))
        out3 = self.fc5(x)
        x = F.relu(self.fc4(x))
        out4 = self.fc5(x)

        if out == 1:
            return out1
        if out == 2:
            return out2
        if out == 3:
            return out3
        if out == 4:
            return out4

        outs = torch.stack([out1, out2, out3, out4], dim=2)
        assert_shape(outs, (N, self.out_dim, 4))
        weighting = F.softmax(self.ensemble, dim=0)
        out = out1 * weighting[0] + out2 * weighting[1] + out3 * weighting[2] + out4 * weighting[3]

        return out

    def shrink_loss(self):
        weighting = F.softmax(self.ensemble, dim=0)
        out = 0 * weighting[0] + 1 * weighting[1] + 2 * weighting[2] + 3 * weighting[3]
        return self.shrink_loss_scale * out


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
        # self.requires_grad_(False)

    def forward(self, x):
        # print(x.shape)
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


def split_train_val(mnist_train):
    val_fraction = 0.1
    val_size = int(val_fraction * len(mnist_train))
    train_size = len(mnist_train) - val_size
    mnist_train, mnist_val = torch.utils.data.random_split(
        mnist_train,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42))

    return mnist_train, mnist_val


def train_mnist(out):
    start = time.time()
    batch_size = 32
    epochs = 3

    mnist_train, mnist_test = mnist_data()
    mnist_train, mnist_val = split_train_val(mnist_train)

    model = ShrinkFC(shrink_loss_scale=1)
    print(f"{num_params(model)} parameters")

    criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(),
    #                             lr=0.001,
    #                             momentum=0.99)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=0.001,
                                 betas=(0.99, 0.999))

    train_dataloader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(mnist_val, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True)

    train_losses = []
    train_errors = []
    val_errors = []
    for epoch in range(epochs):
        # print(F.softmax(model.ensemble))
        train_loss = 0
        train_shrink_loss = 0

        for examples, labels in train_dataloader:
            optimizer.zero_grad()
            # examples: [batch_size, 1, 28, 28]
            # labels: [batch_size]
            logits = model(examples, out=out)
            loss = criterion(logits, labels)
            shrink_loss = model.shrink_loss()
            # loss += shrink_loss
            train_loss += loss
            train_shrink_loss += shrink_loss
            loss.backward()
            optimizer.step()

        train_error = model_error(train_dataloader, model, out)
        val_error = model_error(val_dataloader, model, out)

        print(f"epoch: {epoch}\t"
              + f"train loss: {train_loss}\t"
              + f"train shrink loss: {train_shrink_loss}\t"
              + f"train error: {train_error}\t"
              + f"val error: {val_error}\t")
        train_losses.append(train_loss)
        train_errors.append(train_error)
        val_errors.append(val_error)

    test_error = model_error(test_dataloader, model, out)
    print(f"test_error: {test_error}")
    print(f'time taken: {time.time() - start}')
    # torch.save(model.state_dict(), 'model.pt')
    # plot_results(train_losses, train_errors, val_errors)


def model_error(dataloader, model, out):
    total_wrong = 0
    n = 0

    with torch.no_grad():
        model.eval()

        for examples, labels in dataloader:
            n += len(labels)
            logits = model(examples, out=out)
            preds = torch.argmax(logits, dim=1)
            num_wrong = (preds != labels).sum()
            total_wrong += num_wrong

        model.train()
    return total_wrong / n


def plot_results(train_losses, train_errors, val_errors):
    plt.plot(train_losses)
    plt.title('Training loss')
    plt.xlabel('Epoch')
    plt.show()

    plt.plot(train_errors)
    plt.title('Training error')
    plt.xlabel('Epoch')
    plt.show()

    plt.plot(val_errors)
    plt.title('Validation error')
    plt.xlabel('Epoch')
    plt.show()


def num_params(model):
    return sum([torch.prod(torch.tensor(p.shape))
                for p in list(model.parameters())])


if __name__ == '__main__':
    for out in [1,2,3,4,5]:
        train_mnist(out)
