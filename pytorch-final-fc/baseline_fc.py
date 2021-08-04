import torch.nn as nn
import torch.nn.functional as F

# Simple MLP for CIFAR10 image classification
class simpleModel(nn.Module):
    def __init__(self):
        super(simpleModel, self).__init__()

        self.fc1 = nn.Linear(3072, 1536)
        self.fc2 = nn.Linear(1536, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1) # flatten tensor for the final, linear layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) # softmax is used on this layer when calculating loss
    
        return x