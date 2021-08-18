# Script for training and testing models
from baseline_fc import *
from fa_fc import *
from dfa_fc import *
from tp_fc import *
from tp_fc_train import *
from norse_fc import *

from fun_cifar10 import train_CIFAR10, test_CIFAR10
from fun_brainscore import *
from fun_utils import get_basic_transformation

import torch
import torchvision

# Baseline BP model
#train_CIFAR10(simpleModel(), 'FC_Baseline_v2')
#test_CIFAR10(simpleModel(), 'FC_Baseline_v2')
#test_score(model_name = 'FC_Baseline_v2', model_def = simpleModel(), layers_to_consider=['fc1', 'fc2'])

# FA model
#train_CIFAR10(feedbackAlignmentModel(), 'FC_FA_v2')
#test_CIFAR10(feedbackAlignmentModel(), 'FC_FA_v2')
#test_score(model_name = 'FC_FA_v2', model_def = feedbackAlignmentModel(), layers_to_consider=['fc1', 'fc2'])

# DFA model
#train_CIFAR10(DFA_CIFAR10(), 'DFA_FC_v1')
#test_score(DFA_CIFAR10(), 'DFA_FC_v1')

# TP model
# - model trained differently as per Meulemans et al
#summary = setup_summary_dict(dtp_args)

# Same transformation and dataset as the other models
# TODO - put this into the train_CIFAR10 function with 'if DTP this, else normal'#
#transform = torchvision.transforms.Compose(
#            [torchvision.transforms.ToTensor(),
#             torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#trainset = torchvision.datasets.CIFAR10(root = './data', train = True, transform = transform)
#trainloader = torch.utils.data.DataLoader(trainset, batch_size = 4, shuffle = True, num_workers = 0)

#testset = torchvision.datasets.CIFAR10(root = './data', train = False, download = False, transform = transform)
#testloader = torch.utils.data.DataLoader(testset, batch_size = 4, shuffle = False, num_workers = 0)

#train(net=dtp_model, train_loader=trainloader, test_loader=testloader, summary=summary, writer=None, device='cpu', val_loader=None, args=dtp_args)

# STDP SNN
# try using norse for fc snn
#train_CIFAR10(norse_snn, 'snn_fc_v7', spiking=True, debug=False, epochs = 25)
#test_CIFAR10(norse_snn, model_name = 'snn_fc_epoch6')
test_scores(norse_snn, 'snn_fc_v7', layers_to_consider = ['features.1', 'features.3'], test=True)
