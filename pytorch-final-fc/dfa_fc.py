# Implementation from https://github.com/lightonai/dfa-scales-to-modern-deep-learning

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from enum import Enum

def remove_indices(array, indices):
    # From: https://stackoverflow.com/questions/11303225/how-to-remove-multiple-indexes-from-a-list-at-the-same-time
    return [e for i, e in enumerate(array) if i not in set(indices)]

def keep_indices(array, indices):
    return [e for i, e in enumerate(array) if i in set(indices)]


class FeedbackPointsHandling(Enum):
    LAST = 'LAST'  # Store only the last candidate feedback point going through DFALayer
    MINIBATCH = 'MINIBATCH'  # Store all candidate feedback points going through DFALayer (experimental!)
    REDUCE = 'REDUCE'  # Sum feedback points across network and backward on a centralized reduced one


class DFABackend(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dfa_context):

        ctx.dfa_context = dfa_context  # Access to global informations in the backward

        return input

    @staticmethod
    def backward(ctx, grad_output):
        dfa_context = ctx.dfa_context

        # If training, perform the random projection and send it to the feedback points:
        if not dfa_context.no_training:
            grad_size = np.prod(remove_indices(grad_output.shape, dfa_context.batch_dims))
            random_projection = torch.mm(grad_output.reshape(-1, grad_size).to(dfa_context.rp_device),
                                         dfa_context.feedback_matrix)  # Global random projection
            if dfa_context.normalization:
                random_projection /= np.sqrt(np.prod(random_projection.shape[1:]))

            # Go through the feedback points and backward with the RP on each of them:
            for layer in dfa_context.dfa_layers:
                # Select the feedback point based on how they are to be handled:
                feedback_point = layer.feedback_points

                feedback_shape = feedback_point.shape[:]
                feedback_size = np.prod(remove_indices(feedback_point.shape, layer.batch_dims))
                shared_size = np.prod(keep_indices(feedback_point.shape, layer.batch_dims))
                if shared_size != random_projection.shape[0]:
                    random_projection_expanded = random_projection.unsqueeze(1)
                    random_projection_expanded = random_projection_expanded.repeat(1, shared_size // random_projection.shape[0], 1)
                    #print(random_projection_expanded.shape)
                    random_projection_expanded /= np.sqrt(np.prod(shared_size // random_projection.shape[0]))
                    random_projection_expanded = random_projection_expanded[:, :, :feedback_size]
                    feedback = random_projection_expanded.view(*feedback_shape).to(feedback_point.device)
                    feedback_point.backward(feedback)
                else:
                    feedback = random_projection[:, :feedback_size].view(*feedback_shape).to(feedback_point.device)
                    feedback_point.backward(feedback)

        return grad_output, None  # Gradients for output and dfa_context (None)


class DFA(nn.Module):
    def __init__(self, dfa_layers, normalization=True, rp_device=None, no_training=False,
                 feedback_points_handling=FeedbackPointsHandling.LAST, batch_dims=(0,)):
        super(DFA, self).__init__()
        self.dfa_layers = dfa_layers
        self.normalization = normalization
        self.rp_device = rp_device
        self.no_training = no_training
        self.batch_dims = batch_dims

        # Set the feedback points handling mode of all DFALayers
        self.feedback_points_handling = feedback_points_handling
        for dfa_layer in self.dfa_layers:
            dfa_layer.feedback_registrar = self._register_feedback_point
            dfa_layer.feedback_points_handling = feedback_points_handling

            if dfa_layer.batch_dims is None:
                dfa_layer.batch_dims = self.batch_dims

        self.dfa = DFABackend.apply  # Custom DFA autograd function that actually handles the backward

        # Random feedback matrix and its dimensions
        self.feedback_matrix = None
        self.max_feedback_size = 0
        self.output_size = 0

        # Feedback points handling: minibatch
        self.forward_complete = False
        self.backward_batch = 0

        # Feedback points handling: reduce
        self.global_feedback_point = None

        self.initialized = False

    def forward(self, input):
        if not(self.initialized or self.no_training):
            # If we are training, but aren't initialized:
            # - Setup default rp device if none has been specified;
            # - Get the size of the output (output_size);
            # - Get the size of the largest feedback (max_feedback_size);
            # - Generate the backward random matrix (output_size * max_feedback_size).

            if self.rp_device is None:
                # Default to network output device.
                self.rp_device = input.device
                if self.global_feedback_point is not None:
                    self.global_feedback_point = self.global_feedback_point.to(self.rp_device)

            self.output_size = int(np.prod(remove_indices(input.shape, self.batch_dims)))

            for layer in self.dfa_layers:
                print(layer.feedback_shape)
                feedback_size = int(np.prod(remove_indices(layer.feedback_shape, layer.batch_dims)))
                if feedback_size > self.max_feedback_size:
                    self.max_feedback_size = feedback_size

            # The random feedback matrix is uniformly sampled between [-1, 1)
            self.feedback_matrix = torch.rand(self.output_size, self.max_feedback_size, device=self.rp_device) * 2 - 1

            self.initialized = True

        return self.dfa(input, self)

    def _register_feedback_point(self, feedback_point):
        feedback_point_size = np.prod(remove_indices(feedback_point.shape, feedback_point.batch_dims))
        feedback_point = feedback_point.view(-1, feedback_point_size)  # Handle in 1D, put all batch dims together.
        if self.global_feedback_point is None:
            self.global_feedback_point = feedback_point
            if self.rp_device is not None:
                self.global_feedback_point = feedback_point.to(self.rp_device)
        else:
            global_feedback_point_size = np.prod(self.global_feedback_point.shape[1:])
            if global_feedback_point_size > feedback_point_size:
                feedback_point = F.pad(feedback_point.to(self.global_feedback_point.device),
                                       [0, global_feedback_point_size - feedback_point_size])
            elif np.prod(feedback_point.shape) < np.prod(self.global_feedback_point.shape):
                self.global_feedback_point = F.pad(self.global_feedback_point,
                                                   [0, feedback_point_size - global_feedback_point_size])

            self.global_feedback_point = self.global_feedback_point + feedback_point.to(self.global_feedback_point.device)


class DFALayer(nn.Module):
    def __init__(self, name=None, batch_dims=None, passthrough=False):
        super(DFALayer, self).__init__()

        self.name = name
        self.batch_dims = batch_dims
        self.passthrough = passthrough

        self.feedback_registrar = None  # Will be specified by topmost DFA layer

        self.feedback_points_handling = None  # Will be specified by topmost DFA layer
        self.feedback_points = None

        self.feedback_shape = None

        self.initialized = False

    def forward(self, input):
        if not self.initialized:
            self.feedback_shape = input.shape
            if self.feedback_points_handling == FeedbackPointsHandling.LAST:
                self.feedback_points = None
            elif self.feedback_points_handling == FeedbackPointsHandling.MINIBATCH:
                self.feedback_points = []

            self.initialized = True

        # Feedback points are useful for backward calculations, only store them if we are calculating gradients:
        if input.requires_grad:  # TODO: input may be a tuple!
            if self.feedback_points_handling == FeedbackPointsHandling.MINIBATCH:
                self.feedback_points.append(input)
            elif self.feedback_points_handling == FeedbackPointsHandling.LAST:
                self.feedback_points = input
            elif self.feedback_points_handling == FeedbackPointsHandling.REDUCE:
                self.feedback_registrar(input)

        # Passthrough mode is used when reproducing the network but training with BP for alignment measurements.
        if self.passthrough:
            return input
        else:
            output = input.detach()  # Cut the computation graph so that gradients don't flow back beyond DFALayer
            output.requires_grad = True  # Gradients will still be required above
            return output

# Fully connected neural network
class DFA_CIFAR10(nn.Module):
    def __init__(self, training_method='DFA'):
        super(DFA_CIFAR10, self).__init__()
        self.fc1 = nn.Linear(3072, 1536)
        self.fc2 = nn.Linear(1536, 256)
        self.fc3 = nn.Linear(256, 10)

        self.training_method = training_method
        if self.training_method in ['DFA', 'SHALLOW']:
            self.dfa1, self.dfa2 = DFALayer(), DFALayer()
            self.dfa = DFA([self.dfa1, self.dfa2], feedback_points_handling=FeedbackPointsHandling.LAST,
                           no_training=(self.training_method == 'SHALLOW'))

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        if self.training_method in ['DFA', 'SHALLOW']:
            x = self.dfa1(torch.relu(self.fc1(x)))
            x = self.dfa2(torch.relu(self.fc2(x)))
            x = self.dfa(self.fc3(x))
        else:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
        return x

model = DFA_CIFAR10()
data = torch.randn(4, 3,32,32)
output = model(data)
print("output is " + str(output))