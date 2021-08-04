# Implementation from https://github.com/ChFrenkel/DirectRandomTargetProjection

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


class feedbackAlignmentModel(torch.nn.Module):
    def __init__(self):
        super(feedbackAlignmentModel, self).__init__()

        self.fc1 = LinearFA(3072, 1536)
        self.fc2 = LinearFA(1536, 256)
        self.fc3 = LinearFA(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1) # flatten tensor from 12 channels to 1 for the final, linear layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x