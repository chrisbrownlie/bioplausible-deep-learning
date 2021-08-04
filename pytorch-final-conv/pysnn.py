import torch
import pysnn

class Network(SNNNetwork):
    def __init__(self):
        super(Network, self).__init__()

        # Input
        self.input = Input((batch_size, 1, n_in), *input_dynamics)

        # Layer 1
        self.mlp1_c = Linear(n_in, n_hidden, *connection_dynamics)
        self.neuron1 = LIFNeuron((batch_size, 1, n_hidden), *neuron_dynamics)
        self.add_layer("fc1", self.mlp1_c, self.neuron1)

        # Layer 2
        self.mlp2_c = Linear(n_hidden, n_out, *connection_dynamics)
        self.neuron2 = LIFNeuron((batch_size, 1, n_out), *neuron_dynamics)
        self.add_layer("fc2", self.mlp2_c, self.neuron2)

        # Feedback connection from neuron 2 to neuron 1
        self.mlp2_prev = Linear(n_out, n_hidden, *c_dynamics)
        self.add_layer("fc2_back", self.mlp2_prev, self.neuron1)

    def forward(self, input):
        spikes, trace = self.input(input)

        # Layer 1
        x_prev, _ = self.mlp2_prev(self.neuron2.spikes, self.neuron2.trace)
        x_forw, _ = self.mlp1_c(x, t)
        x, t = self.neuron1([x_forw, x_rec, x_prev])

        # Layer out
        spikes, trace = self.mlp2_c(spikes, trace)
        spikes, trace = self.neuron2(spikes, trace)

        return x