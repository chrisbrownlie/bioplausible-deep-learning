import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
from numpy import prod

class HookFunction(Function):
    @staticmethod
    def forward(ctx, input, labels, y, fixed_fb_weights):
        if fixed_fb_weights is None:
            ctx.output_layer = True
        else:
            ctx.save_for_backward(input, labels, y, fixed_fb_weights)
            ctx.output_layer = False
        return input

    @staticmethod
    def backward(ctx, grad_output):
        output_layer = ctx.output_layer
        if output_layer:
            return grad_output, None, None, None, None
        input, labels, y, fixed_fb_weights = ctx.saved_variables

        grad_output_est = (y-labels).mm(fixed_fb_weights.view(-1,prod(fixed_fb_weights.shape[1:]))).view(grad_output.shape)
        return torch.clamp(grad_output_est, -0.5, 0.5), None, None, None, None

trainingHook = HookFunction.apply

class TrainingHook(nn.Module):
    def __init__(self, dim_hook):
        super(TrainingHook, self).__init__()

        # Feedback weights definition
        if dim_hook is None:
            # This will be the case for the output layer
            self.fixed_fb_weights = None
        else:
            self.fixed_fb_weights = nn.Parameter(torch.Tensor(torch.Size(dim_hook)))
            self.reset_weights()

    def reset_weights(self):
        torch.nn.init.kaiming_uniform_(self.fixed_fb_weights)
        self.fixed_fb_weights.requires_grad = False

    def forward(self, input, labels, y):
        return trainingHook(input, labels, y, self.fixed_fb_weights)

    def __repr__(self):
        return self.__class__.__name__


class FC_block(nn.Module):
    def __init__(self, in_features, out_features, dim_hook):
        super(FC_block, self).__init__()
        
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        
        
        self.hook = TrainingHook(dim_hook=dim_hook)

    def forward(self, x, labels, y):
        x = self.fc(x)
        x = self.hook(x, labels, y)
        return x


class CNN_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dim_hook, second_block = False):
        super(CNN_block, self).__init__()
        
        # Second block has no pooling
        self.second_block = second_block

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.act = F.relu
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.hook = TrainingHook(dim_hook=dim_hook)

    def forward(self, x, labels, y):
        x = self.conv(x)
        x = self.act(x)
        x = self.hook(x, labels, y)
        if self.second_block:
            # Second block has no pooling
            return x
        x = self.pool(x)
        return x


class DFA_Network(nn.Module):
    def __init__(self):
        super(DFA_Network, self).__init__()
        self.y = torch.zeros(4, 10)
        self.y.requires_grad = False

        # First convolutional layer:
        # - input channels = 3 - as CIFAR10 contains RGB images)
        # - output channels = 6 - increasing channel depth to improve feature detection
        # - kernel size = 5 - standard kernel size
        self.conv1 = CNN_block(3, 6, 5, dim_hook = [10,6,28,28])
        # Activation and pooling done in cnn block

        # Second convolutional layer:
        # - input channels = 6 - as the pooling layer does not affect number of channels and this was the output from the first conv layer)
        # - output channels = 12 - again increasing channel depth
        # - kernel size = 5 - standard kernel size
        self.conv2 = CNN_block(6, 12, 5, dim_hook = [10,12,10,10], second_block = True)
        # Activation and pooling done in cnn block
        
        # First element comes from the batch size (4) multiplied by the output of each kernel (5*5, as kernel size in
        # previous layer is 5), multiplied by the number of channels output from the previous layer (12) (12*5*5*4 = 1200)
        # (This calculation is 'performed' in the flattening operation view() in the forward method)
        self.fully_connected1 = FC_block(1200, 120, dim_hook = [10,120])

        # 10 neurons in output layer to correspond to the 10 categories of object
        self.fully_connected2 = FC_block(120, 10, dim_hook = None)

    def forward(self, x, labels):
        x = self.conv1(x, labels, self.y)
        x = self.conv2(x, labels, self.y)
        x = x.view(x.size(0), -1) # flatten tensor from 12 channels to 1 for the final, linear layers
        x = self.fully_connected1(x, labels, self.y)
        x = self.fully_connected2(x, labels, self.y)

        if x.requires_grad and (self.y is not None):
            self.y.data.copy_(x.data) # in-place update, only happens with (s)DFA
        
        return x

#model = DFA_Network()
#data = torch.randn(4,3,32,32)
#targets = torch.zeros(4, 10)
#output = model(data, targets)
#print("output is " + str(output))