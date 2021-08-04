import torch
import torch.nn as nn
import torch.nn.functional as F

class FA_wrapper(nn.Module):
    def __init__(self, module, layer_type, dim):
        super(FA_wrapper, self).__init__()
        self.module = module
        self.layer_type = layer_type
        self.output_grad = None
        self.x_shape = None

        # FA feedback weights definition
        self.fixed_fb_weights = nn.Parameter(torch.Tensor(torch.Size(dim)))
        self.reset_weights()

    def forward(self, x):
        if x.requires_grad:
            x.register_hook(self.FA_hook_pre)
            self.x_shape = x.shape
            x = self.module(x)
            x.register_hook(self.FA_hook_post)
            return x
        else:
            return self.module(x)

    def reset_weights(self):
        torch.nn.init.kaiming_uniform_(self.fixed_fb_weights)
        self.fixed_fb_weights.requires_grad = False
    
    def FA_hook_pre(self, grad):
        if self.output_grad is not None:
            if (self.layer_type == "fc"):
                grad = self.output_grad.mm(self.fixed_fb_weights)
                return torch.clamp(grad, -0.5, 0.5)
            elif (self.layer_type == "conv"):
                grad = torch.nn.grad.conv2d_input(self.x_shape, self.fixed_fb_weights, self.output_grad)
                return torch.clamp(grad, -0.5, 0.5)
            else:
                raise NameError("=== ERROR: layer type " + str(self.layer_type) + " is not supported in FA wrapper")
        else:
            return grad

    def FA_hook_post(self, grad):
        self.output_grad = grad
        return grad


class LinearFA(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearFA, self).__init__()
        
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)

        self.fc = FA_wrapper(module=self.fc, layer_type='fc', dim=self.fc.weight.shape)

    def forward(self, x):

        x = self.fc(x)

        return x

class ConvFA(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvFA, self).__init__()
        
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)

        self.conv = FA_wrapper(module=self.conv, layer_type='conv', dim=self.conv.weight.shape)
        

    def forward(self, x):

        x = self.conv(x)

        return x


class feedbackAlignmentModel(torch.nn.Module):
    def __init__(self):
        super(feedbackAlignmentModel, self).__init__()

        # First convolutional layer:
        # - input channels = 3 - as CIFAR10 contains RGB images)
        # - output channels = 6 - increasing channel depth to improve feature detection
        # - kernel size = 5 - standard kernel size
        self.conv1 = ConvFA(3, 6, 5)
        self.relu1 = torch.nn.ReLU() # ReLU activation
        
        # Pooling layer:
        # Identify position and translation invariant features
        # kernel size and stride of 2 - result will be 1/4 of the original number of features
        self.pool = torch.nn.MaxPool2d(2, 2)

        # Second convolutional layer:
        # - input channels = 6 - as the pooling layer does not affect number of channels and this was the output from the first conv layer)
        # - output channels = 12 - again increasing channel depth
        # - kernel size = 5 - standard kernel size
        self.conv2 = ConvFA(6, 12, 5)
        self.relu2 = torch.nn.ReLU() # ReLU activation
        
        # First element comes from the batch size (4) multiplied by the output of each kernel (5*5, as kernel size in
        # previous layer is 5), multiplied by the number of channels output from the previous layer (12) (12*5*5*4 = 1200)
        # (This calculation is 'performed' in the flattening operation view() in the forward method)
        self.fully_connected1 = LinearFA(1200, 120)

        # 10 neurons in output layer to correspond to the 10 categories of object
        self.fully_connected2 = LinearFA(120, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = x.view(x.size(0), -1) # flatten tensor from 12 channels to 1 for the final, linear layers
        x = self.fully_connected1(x)
        x = self.fully_connected2(x)
        return x