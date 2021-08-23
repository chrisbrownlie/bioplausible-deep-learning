import torch, torch.nn as nn
from norse.torch import LICell             # Leaky integrator
from norse.torch import LIFCell            # Leaky integrate-and-fire
from norse.torch import LIFParameters            # Leaky integrate-and-fire
from norse.torch import SequentialState    # Stateful sequential layers
from norse.torch import PoissonEncoder
import matplotlib.pyplot as plt
import torch


class simpleSNN(nn.Module):
    def __init__(self):
        super(simpleSNN, self).__init__()

        self.encoder = PoissonEncoder(seq_length = 100, f_max = 20) # simulate 100 timesteps

        self.features = SequentialState(
            nn.Linear(3072, 1536),
            LIFCell(LIFParameters(v_th=torch.as_tensor(0.4))),                   # Spiking activation layer
            nn.Linear(1536, 256),
            LIFCell(LIFParameters(v_th=torch.as_tensor(0.4)))
            )

        self.classification = SequentialState(
            # Classification
            nn.Linear(256, 10),
            LICell()                    # Non-spiking integrator layer
        )

    def forward(self, x):

        # Encode image as vector of binary spikes across 100 timesteps
        x = self.encoder(x).reshape(100,4,3*32*32)

        # Integrate inputs across the 100 timesteps
        voltages = torch.empty(
            100, 4, 10, dtype=x.dtype
        )
        sf = None
        sc = None
        for ts in range(100):
            out_f, sf = self.features(x[ts, :], sf)
            out_c, sc = self.classification(out_f, sc)
            voltages[ts, :, :] = out_c
            
        y_hat, _ = torch.max(voltages, 0)
        return y_hat # as with other models, softmax applied in CE loss