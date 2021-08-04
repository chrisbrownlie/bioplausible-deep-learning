# Script for training and testing models
from baseline_model import *
from feedback_alignment import *
from direct_feedback_alignment import *
import bindsnet
from model_tools.activations.pytorch import PytorchWrapper
from model_tools.activations.pytorch import load_preprocess_images
import functools
from model_tools.brain_transformation import ModelCommitment
from brainscore import score_model

from fun_cifar10 import train_CIFAR10, test_CIFAR10
from test_scoring import test_score
from train_DFA import *

# Simple model
#train_CIFAR10(simpleModel(), 'simpleModel_testingw')
#test_score(model_name = 'simpleModel_testingw', model_def = simpleModel())


# FA model
#train_CIFAR10(feedbackAlignmentModel(), 'FAModel_testing2')
#test_score(model_name = 'FAModel_testing2', model_def = feedbackAlignmentModel(), layers_to_consider=['conv1', 'conv2', 'relu1', 'relu2'])

# DFA model
model = DFA_Network()
transform = get_basic_transformation()

trainset = torchvision.datasets.CIFAR10(root = './data', train = True, transform = transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size = 4, shuffle = True, num_workers = 0)

testset = torchvision.datasets.CIFAR10(root = './data', train = False, download = False, transform = transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size = 4, shuffle = False, num_workers = 0)
# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Loss function
criterion = (F.cross_entropy, (lambda l : torch.max(l, 1)[1]))
for p in model.parameters():
    if p.requires_grad:
        p.register_hook(lambda grad: torch.clamp(grad, -0.5, 0.5))
train_epoch(model, train_loader, optimizer, criterion)
#trainDFA(DFA_Network(), 2)

#train_CIFAR10(DFA_Network(), 'DFA_testing')
#test_score(model_name = 'DFA_testing2', model_def = DFA_Network(), layers_to_consider=['conv1', 'conv2', 'fully_connected1'], fun_trained=False)

# TP model
#train_CIFAR10()

# STDP SNN
#state_dict
#snn_model = bindsnet.network.Network()
#snn_model.load_state_dict(torch.load('bindsnet_snn_sd.pt'))

#snn_model = torch.load('pytorch-final/bindsnet_snn.pt')
#wrapped = PytorchWrapper(snn_model, preprocessing=functools.partial(load_preprocess_images, image_size=32))
#brain_model = ModelCommitment(
#        identifier = 'snn', 
#        activations_model = wrapped,
#        layers = ['X', 'Y']
#    )
#public_IT_score = score_model(
#        model_identifier = brain_model.identifier,
#        model = brain_model,
#        benchmark_identifier='dicarlo.MajajHong2015public.IT-pls'
 #   )
#test_score(model_name = 'bindsnet_snn', model_def = snn_model, layers_to_consider=['X', 'Y'], fun_trained = False)