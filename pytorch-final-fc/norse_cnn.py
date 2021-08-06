import torch
import torch.nn as nn
from norse.torch.module.lif import LIFFeedForwardCell            # Leaky integrate-and-fire
from norse.torch.module.leaky_integrator import LICell            # Leaky integrate-and-fire
import torchvision
from fun_utils import *

import matplotlib.pyplot as plt
import numpy as np

class NorseSNN(torch.nn.Module):
    def __init__(self):
        super(NorseSNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.lif1 = LIFFeedForwardCell((6, 18, 18))
        self.mpool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 12, 5)
        self.lif2 = LIFFeedForwardCell((12, 5, 5))
        self.fl = nn.Flatten()
        self.fc1 = nn.Linear(1200, 120)
        self.fc2 = nn.Linear(120, 10)
        self.output = LICell(10,10)

    def forward(self, x):
        seq_length = 10
        seq_batch_size = 4

        # Initialize state variables
        s1 = None
        s2 = None
        so = None

        voltages = torch.zeros(seq_length, seq_batch_size, 10)

        for ts in range(seq_length):
            print('ts is ' + str(ts))
            print('.........')
            print('input is:')
            print(x[ts, :])
            z = self.conv1(x[ts, :])
            z = self.lif1(z)
            z = self.mpool(z)
            z = 10 * self.conv2(z)
            z = self.lif2(z)
            z = self.fl(z)
            z = self.fc1(z)
            z = self.fc2(z)
            v = self.output(z)
            voltages[ts, :, :] = v
        return voltages


def train(model):
    
    model.train()
    losses = []
    epoch = 10
    optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.01,
            momentum=0.9,
            weight_decay=5e-4 * 4,
            nesterov=True,
        )

    
    transform = get_basic_transformation()

    trainset = torchvision.datasets.CIFAR10(root = './data', train = True, transform = transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size = 4, shuffle = True, num_workers = 0)

    train_batches = len(train_loader)
    step = train_batches * epoch

    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.nll_loss(output, target)
        loss.backward()

        optimizer.step()
        step += 1

        ts = np.arange(0, 200)
        _, axs = plt.subplots(4, 4, figsize=(15, 10), sharex=True, sharey=True)
        axs = axs.reshape(-1)  # flatten
        for nrn in range(10):
            one_trace = model.voltages.detach().cpu().numpy()[:, 0, nrn]
            plt.sca(axs[nrn])
            plt.plot(ts, one_trace)
        plt.xlabel("Time [s]")
        plt.ylabel("Membrane Potential")
        plt.show()
        losses.append(loss.item())

    mean_loss = np.mean(losses)
    return losses, mean_loss

snn_model = NorseSNN()
train(snn_model)