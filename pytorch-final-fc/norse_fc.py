import torch, torch.nn as nn
from norse.torch import LICell             # Leaky integrator
from norse.torch import LIFCell            # Leaky integrate-and-fire
from norse.torch import LIFParameters            # Leaky integrate-and-fire
from norse.torch import SequentialState    # Stateful sequential layers

class simpleSNN(nn.Module):
    def __init__(self):
        super(simpleSNN, self).__init__()

        self.seq_length = 128 # number of timesteps

        self.features = SequentialState(
            nn.Flatten(),
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
        voltages = torch.empty(
            self.seq_length, x.shape[0], 10, device=x.device, dtype=x.dtype
        )
        sf = None
        sc = None
        for ts in range(self.seq_length):
            out_f, sf = self.features(x, sf)
            # print(out_f.shape)
            out_c, sc = self.classification(out_f, sc)
            voltages[ts, :, :] = out_c + 0.001 * torch.randn(
                x.shape[0], 10, device=x.device
            )

        y_hat, _ = torch.max(voltages, 0)
        return y_hat
    
        return x

norse_snn = simpleSNN()
data = torch.randn(4, 3, 32, 32) # 8 batches, 3 channels, 28x28 pixels
output = norse_snn(data)      # Provides a tuple (tensor (8, 10), neuron state)
#print("snn output is:")
#print(output)