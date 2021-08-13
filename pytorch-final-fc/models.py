# Script for training and testing models
from baseline_fc import *
from fa_fc import *
from dfa_fc import *
from tp_fc import *
from tp_fc_train import *
#from norse_fc import *

# Disable caching
import os
os.environ["RESULTCACHING_DISABLE"] = "1"


from fun_cifar10 import train_CIFAR10, test_CIFAR10
from fun_brainscore import *
from fun_utils import get_basic_transformation

# Baseline BP model 
#train_CIFAR10(simpleModel(), 'FC_Baseline_v3')
#test_scores(model_name = 'FC_Baseline_v3', model_def = simpleModel(), layers_to_consider=['fc1', 'fc2'])

# FA model
#train_CIFAR10(feedbackAlignmentModel(), 'FC_FA_v3')
#test_scores(model_name = 'FC_FA_v3', model_def = feedbackAlignmentModel(), layers_to_consider=['fc1', 'fc2'])

# DFA model
#train_CIFAR10(DFA_CIFAR10(), 'DFA_FC_v3')
#test_scores(model_name = 'DFA_FC_v3', model_def = DFA_CIFAR10(), layers_to_consider=['fc1', 'fc2'])

# TP model
train_CIFAR10(dtp_model, 'DTP_FC_v5', meulemans=True)
test_scores(model_name = 'DTP_FC_v5', model_def = dtp_model, layers_to_consider=['fc1', 'fc2'], meulemans=True)

# STDP SNN
# try using norse for fc snn
#train_CIFAR10(norse_snn, 'snn_fc_v1', spiking=True, debug=False)
#test_score(norse_snn, 'snn_fc_v1')


os.environ["RESULTCACHING_DISABLE"] = ""