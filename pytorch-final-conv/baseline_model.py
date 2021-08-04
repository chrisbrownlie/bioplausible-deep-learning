import numpy as np
import torch

# Simple CNN for CIFAR10 image classification
class simpleModel(torch.nn.Module):
    def __init__(self):
        super(simpleModel, self).__init__()

        # First convolutional layer:
        # - input channels = 3 - as CIFAR10 contains RGB images)
        # - output channels = 6 - increasing channel depth to improve feature detection
        # - kernel size = 5 - standard kernel size
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.relu1 = torch.nn.ReLU() # ReLU activation
        
        # Pooling layer:
        # Identify position and translation invariant features
        # kernel size and stride of 2 - result will be 1/4 of the original number of features
        # in this case, the input goes from 28*28 (after first conv layer) to 14*14
        self.pool = torch.nn.MaxPool2d(2, 2)

        # Second convolutional layer:
        # - input channels = 6 - as the pooling layer does not affect number of channels and this was the output from the first conv layer
        # - output channels = 12 - again increasing channel depth
        # - kernel size = 5 - standard kernel size
        self.conv2 = torch.nn.Conv2d(6, 12, 5)
        self.relu2 = torch.nn.ReLU() # ReLU activation
        
        # First element comes from the size of (no. of features in) each input which is currently 100 (10*10 so 100 features): 
        # from input of 32*32 -conv-> 28*28 -maxpool-> 14*14 -conv-> 10*10. This is multiplied by the number of channels output from 
        # the previous layer (12) (12*10*10 = 1200). This flattening of the input is done with x.view() in the forward method
        self.fully_connected1 = torch.nn.Linear(1200, 120)

        # 10 neurons in output layer to correspond to the 10 categories of object
        self.fully_connected2 = torch.nn.Linear(120, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = x.view(x.size(0), -1) # flatten tensor for the final, linear layers
        x = self.fully_connected1(x)
        x = self.fully_connected2(x)
    
        return x