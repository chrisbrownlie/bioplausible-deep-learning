### Code from Meulemans et al (2020)


## UTILS functions
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tensorboardX import SummaryWriter
import os
import pandas as pd
import warnings
from argparse import Namespace

# from lib import networks, direct_feedback_networks
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
from matplotlib import rc
import matplotlib.pyplot as plt


class RegressionDataset(Dataset):
    """A simple regression dataset.
    Args:
        inputs (numpy.ndarray): The input samples.
        outputs (numpy.ndarray): The output samples.
    """
    def __init__(self, inputs, outputs, double_precision=False):
        assert(len(inputs.shape) == 2)
        assert(len(outputs.shape) == 2)
        assert(inputs.shape[0] == outputs.shape[0])

        if double_precision:
            self.inputs = torch.from_numpy(inputs).to(torch.float64)
            self.outputs = torch.from_numpy(outputs).to(torch.float64)
        else:
            self.inputs = torch.from_numpy(inputs).float()
            self.outputs = torch.from_numpy(outputs).float()

    def __len__(self):
        return int(self.inputs.shape[0])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        batch_in = self.inputs[idx, :]
        batch_out = self.outputs[idx, :]

        return batch_in, batch_out


def compute_accuracy(predictions, labels):
    """
    Compute the average accuracy of the given predictions.
    Inspired on
    https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    Args:
        predictions (torch.Tensor): Tensor containing the output of the linear
            output layer of the network.
        labels (torch.Tensor): Tensor containing the labels of the mini-batch
    Returns (float): average accuracy of the given predictions
    """

    _, pred_labels = torch.max(predictions.data, 1)
    total = labels.size(0)
    correct = (pred_labels == labels).sum().item()

    return correct/total


def choose_optimizer(args, net):
    """
    Return the wished optimizer (based on inputs from args).
    Args:
        args: cli
        net: neural network
    Returns: optimizer
    """
    forward_optimizer = OptimizerList(args, net)
    # feedback_optimizer = choose_feedback_optimizer(args, net)
    feedback_optimizer = FbOptimizerList(args, net)

    return forward_optimizer, feedback_optimizer


def choose_forward_optimizer(args, net):
    """
    Return the wished optimizer (based on inputs from args).
    Args:
        args: cli
        net: neural network
    Returns: optimizer
    """
    if args.freeze_BPlayers:
        forward_params = net.get_reduced_forward_parameter_list()
    elif args.network_type in ('BP', 'BPConv'):
        if args.shallow_training:
            print('Shallow training')
            forward_params = net.layers[-1].parameters()
        elif args.only_train_first_layer:
            print('Only training first layer')
            forward_params = net.layers[0].parameters()
        elif args.only_train_last_two_layers:
            raise NotImplementedError('not yet implemented for BP')
        elif args.only_train_last_three_layers:
            raise NotImplementedError('not yet implemented for BP')
        elif args.only_train_last_four_layers:
            raise NotImplementedError('not yet implemented for BP')
        else:
            forward_params = net.parameters()
    else:
        if args.only_train_first_layer:
            print('Only training first layer')
            forward_params = net.get_forward_parameter_list_first_layer()
        elif args.freeze_output_layer:
            print('Freezing output layer')
            forward_params = net.get_reduced_forward_parameter_list()
        elif args.only_train_last_two_layers:
            forward_params = net.get_forward_parameters_last_two_layers()
        elif args.only_train_last_three_layers:
            forward_params = net.get_forward_parameters_last_three_layers()
        elif args.only_train_last_four_layers:
            forward_params = net.get_forward_parameters_last_four_layers()
        else:
            forward_params = net.get_forward_parameter_list()

    if args.optimizer == 'SGD':
        print('Using SGD optimizer')

        forward_optimizer = torch.optim.SGD(forward_params,
                                            lr=args.lr, momentum=args.momentum,
                                            weight_decay=args.forward_wd)
    elif args.optimizer == 'RMSprop':
        print('Using RMSprop optimizer')

        forward_optimizer = torch.optim.RMSprop(
            forward_params,
            lr=args.lr,
            momentum=args.momentum,
            alpha=0.95,
            eps=0.03,
            weight_decay=args.forward_wd,
            centered=True
        )

    elif args.optimizer == 'Adam':
        print('Using Adam optimizer')

        forward_optimizer = torch.optim.Adam(
            forward_params,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.epsilon,
            weight_decay=args.forward_wd
        )

    else:
        raise ValueError('Provided optimizer "{}" is not supported'.format(
            args.optimizer
        ))

    return forward_optimizer


def choose_feedback_optimizer(args, net):
    """
    Return the wished optimizer (based on inputs from args).
    Args:
        args: cli
        net: neural network
    Returns: optimizer
    """

    feedback_params = net.get_feedback_parameter_list()


    if args.optimizer_fb == 'SGD':
        feedback_optimizer = torch.optim.SGD(feedback_params,
                                             lr=args.lr_fb,
                                             weight_decay=args.feedback_wd)
    elif args.optimizer_fb == 'RMSprop':

        feedback_optimizer = torch.optim.RMSprop(
            feedback_params,
            lr=args.lr_fb,
            momentum=args.momentum,
            alpha=0.95,
            eps=0.03,
            weight_decay=args.feedback_wd,
            centered=True
        )

    elif args.optimizer_fb == 'Adam':

        feedback_optimizer = torch.optim.Adam(
            feedback_params,
            lr=args.lr_fb,
            betas=(args.beta1_fb, args.beta2_fb),
            eps=args.epsilon_fb,
            weight_decay=args.feedback_wd
        )

    else:
        raise ValueError('Provided optimizer "{}" is not supported'.format(
            args.optimizer
        ))

    return feedback_optimizer


class OptimizerList(object):
    """ A class for stacking a separate optimizer for each layer in a list. If
    no separate learning rates per layer are required, a single optimizer is
    stored in the optimizer list."""

    def __init__(self, args, net):
        if isinstance(args.lr, float):
            forward_optimizer = choose_forward_optimizer(args, net)
            optimizer_list = [forward_optimizer]
        elif isinstance(args.lr, np.ndarray):
            # if args.network_type in ('BP', 'BPConv'):
            #     raise NetworkError('Multiple learning rates is not yet '
            #                        'implemented for BP')
            if args.freeze_BPlayers:
                raise NotImplementedError('freeze_BPlayers not '
                                          'yet supported in '
                                          'OptimizerList')
            else:
                if not args.network_type == 'BPConv':
                    if args.only_train_first_layer:
                        print('Only training first layer')
                        forward_params = \
                            net.get_forward_parameter_list_first_layer()
                    elif args.freeze_output_layer:
                        print('Freezing output layer')
                        forward_params = net.get_reduced_forward_parameter_list()
                    elif args.only_train_last_two_layers:
                        forward_params = net.get_forward_parameters_last_two_layers()
                    elif args.only_train_last_three_layers:
                        forward_params = net.get_forward_parameters_last_three_layers()
                    elif args.only_train_last_four_layers:
                        forward_params = net.get_forward_parameters_last_four_layers()
                    else:
                        forward_params = net.get_forward_parameter_list()

                    if (not args.no_bias and not args.freeze_output_layer and
                        len(args.lr)*2 != len(forward_params)) or \
                            (args.no_bias and not args.freeze_output_layer and
                             len(args.lr) != len(forward_params)):
                        raise NetworkError('The lenght of the list with learning rates '
                                           'does not correspond with the size of the '
                                           'network.')
            if not (args.optimizer == 'SGD' or args.optimizer == 'Adam'):
                raise NetworkError('multiple learning rates are only supported '
                                   'for SGD optimizer')

            optimizer_list = []
            for i, lr in enumerate(args.lr):
                eps = args.epsilon[i]
                if args.network_type == 'BPConv':
                    if i == 0:
                        j = 0
                    elif i == 1:
                        j = 2
                    elif i == 2:
                        j = 4
                    elif i == 3:
                        j = 5
                    if args.no_bias:
                        parameters = [net.layers[j].weight]
                    else:
                        parameters = [net.layers[j].weight, net.layers[j].bias]
                else:
                    if args.no_bias:
                        parameters = [net.layers[i].weights]
                    else:
                        parameters = [net.layers[i].weights, net.layers[i].bias]
                if args.optimizer == 'SGD':
                    optimizer = torch.optim.SGD(parameters,
                                                lr=lr, momentum=args.momentum,
                                                weight_decay=args.forward_wd)
                elif args.optimizer == 'Adam':
                    optimizer = torch.optim.Adam(
                        parameters,
                        lr=lr,
                        betas=(args.beta1, args.beta2),
                        eps=eps,
                        weight_decay=args.forward_wd)
                optimizer_list.append(optimizer)
        else:
            raise ValueError('Command line argument lr={} is not recognized '
                             'as a float'
                       'or list'.format(args.lr))

        self._optimizer_list = optimizer_list

    def zero_grad(self):
        for optimizer in self._optimizer_list:
            optimizer.zero_grad()

    def step(self, i=None):
        """
        Perform a step on the optimizer of layer i. If i is None, a step is
        performed on all optimizers.
        """
        if i is None:
            for optimizer in self._optimizer_list:
                optimizer.step()
        else:
            self._optimizer_list[i].step()

class FbOptimizerList(object):
    def __init__(self, args, net):
        if isinstance(args.lr_fb, float):
            fb_optimizer = choose_feedback_optimizer(args, net)
            optimizer_list = [fb_optimizer]
        else:
            assert args.network_type == 'DDTPConv'
            assert len(args.lr_fb) == 2
            assert args.optimizer == 'Adam'
            if isinstance(args.epsilon_fb, float):
                args.epsilon_fb = [args.epsilon_fb, args.epsilon_fb]
            else:
                assert len(args.epsilon_fb) == 2
            conv_fb_parameters = net.get_conv_feedback_parameter_list()
            fc_fb_parameters = net.get_fc_feedback_parameter_list()
            conv_fb_optimizer = torch.optim.Adam(
                                    conv_fb_parameters,
                                    lr=args.lr_fb[0],
                                    betas=(args.beta1_fb, args.beta2_fb),
                                    eps=args.epsilon_fb[0],
                                    weight_decay=args.feedback_wd
                                    )
            fc_fb_optimizer = torch.optim.Adam(
                                    fc_fb_parameters,
                                    lr=args.lr_fb[1],
                                    betas=(args.beta1_fb, args.beta2_fb),
                                    eps=args.epsilon_fb[1],
                                    weight_decay=args.feedback_wd
                                    )

            optimizer_list = [conv_fb_optimizer, fc_fb_optimizer]
        self._optimizer_list = optimizer_list

    def step(self):
        for optimizer in self._optimizer_list:
            optimizer.step()

    def zero_grad(self):
        for optimizer in self._optimizer_list:
            optimizer.zero_grad()


def save_logs(writer, step, net, loss, accuracy, test_loss, test_accuracy,
              val_loss, val_accuracy):
    """
    Save logs and plots to tensorboardX
    Args:
        writer (SummaryWriter): TensorboardX summary writer
        step: global step
        net: network
        loss: current loss of the training iteration
    """
    net.save_logs(writer, step) #FIXME: Uncomment again. This does not work for DKDTP
    writer.add_scalar(tag='training_metrics/loss',
                      scalar_value=loss,
                      global_step=step)
    writer.add_scalar(tag='training_metrics/test_loss',
                      scalar_value=test_loss,
                      global_step=step)
    if val_loss is not None:
        writer.add_scalar(tag='training_metrics/val_loss',
                          scalar_value=val_loss,
                          global_step=step)
    if accuracy is not None:
        writer.add_scalar(tag='training_metrics/accuracy',
                          scalar_value=accuracy,
                          global_step=step)
        writer.add_scalar(tag='training_metrics/test_accuracy',
                          scalar_value=test_accuracy,
                          global_step=step)
        if val_accuracy is not None:
            writer.add_scalar(tag='training_metrics/val_accuracy',
                              scalar_value=val_accuracy,
                              global_step=step)

def save_forward_batch_logs(args, writer, step, net, loss, output_activation):
    """
    Save logs and plots for the current mini-batch on tensorboardX
    Args:
        args (Namespace): commandline arguments
        writer (SummaryWriter): TensorboardX summary writer
        step: global step
        net (networks.DTPNetwork): network
        loss (torch.Tensor): loss of the current minibatch
        output_activation (torch.Tensor): output of the network for the current
            minibatch
    """
    if args.save_BP_angle:
        retain_graph = args.save_GN_angle or args.save_GN_activations_angle or \
            args.save_BP_activations_angle or args.save_GNT_angle or \
                       args.save_nullspace_norm_ratio
        # if we need to compute and save other angles afterwards, the gradient
        # graph should be retained such that it can be reused
        net.save_bp_angles(writer, step, loss, retain_graph)
    if args.save_GN_angle:
        retain_graph = args.save_GN_activations_angle or \
                       args.save_BP_activations_angle or args.save_GNT_angle or \
                       args.save_nullspace_norm_ratio
        net.save_gn_angles(writer, step, output_activation, loss,
                           args.gn_damping, retain_graph)
    if args.save_BP_activations_angle:
        retain_graph = args.save_GN_activations_angle or args.save_GNT_angle or \
                       args.save_nullspace_norm_ratio
        net.save_bp_activation_angle(writer, step, loss, retain_graph)

    if args.save_GN_activations_angle:
        retain_graph = args.save_GNT_angle or args.save_nullspace_norm_ratio
        net.save_gn_activation_angle(writer, step, output_activation, loss,
                                     args.gn_damping, retain_graph)
    if args.save_GNT_angle:
        retain_graph = args.save_nullspace_norm_ratio
        net.save_gnt_angles(writer, step, output_activation, loss,
                            args.gn_damping, retain_graph)

    if args.save_nullspace_norm_ratio:
        retain_graph = False
        net.save_nullspace_norm_ratio(writer, step, output_activation,
                                  retain_graph)


def save_feedback_batch_logs(args, writer, step, net, init=False):
    """
    Save logs and plots for the current mini-batch on tensorboardX
    Args:
        args (Namespace): commandline arguments
        writer (SummaryWriter): TensorboardX summary writer
        step: global step
        net (networks.DTPNetwork): network
        init (bool): flag indicating that the training is in the
                initialization phase (only training the feedback weights).
    """
    net.save_feedback_batch_logs(writer, step, init=init)


def save_gradient_hook(module, grad_input, grad_output):
    """ A hook that will be used to save the gradients the loss with respect
             to the output of the network. This gradient is used to compute the
              target for the output layer."""
    print('save grad in module')
    module.output_network_gradient = grad_input[0]


def compute_jacobian(input, output, structured_tensor=False,
                     retain_graph=False):
    """
    Compute the Jacobian matrix of output with respect to input. If input
    and/or output have more than one dimension, the Jacobian of the flattened
    output with respect to the flattened input is returned if
    structured_tensor is False. If structured_tensor is True, the Jacobian is
    structured in dimensions output_shape x flattened input shape. Note that
    output_shape can contain multiple dimensions.
    Args:
        input (list or torch.Tensor): Tensor or sequence of tensors
            with the parameters to which the Jacobian should be
            computed. Important: the requires_grad attribute of input needs to
            be True while computing output in the forward pass.
        output (torch.Tensor): Tensor with the values of which the Jacobian is
            computed
        structured_tensor (bool): A flag indicating if the Jacobian
            should be structured in a tensor of shape
            output_shape x flattened input shape instead of
            flattened output shape x flattened input shape.
    Returns (torch.Tensor): 2D tensor containing the Jacobian of output with
        respect to input if structured_tensor is False. If structured_tensor
        is True, the Jacobian is structured in a tensor of shape
        output_shape x flattened input shape.
    """
    # We will work with a sequence of input tensors in the following, so if
    # we get one Tensor instead of a sequence of tensors as input, make a
    # list out of it.
    if isinstance(input, torch.Tensor):
        input = [input]

    output_flat = output.view(-1)
    numel_input = 0
    for input_tensor in input:
        numel_input += input_tensor.numel()
    jacobian = torch.Tensor(output.numel(), numel_input)

    for i, output_elem in enumerate(output_flat):

        if i == output_flat.numel() - 1:
            # in the last autograd call, the graph should be retained or not
            # depending on our argument retain_graph.
            gradients = torch.autograd.grad(output_elem, input,
                                            retain_graph=retain_graph,
                                            create_graph=False,
                                            only_inputs=True)
        else:
            # if it is not yet the last autograd call, retain_graph should be
            # True such that the remainder parts of the jacobian can be
            # computed.
            gradients = torch.autograd.grad(output_elem, input,
                                            retain_graph=True,
                                            create_graph=False,
                                            only_inputs=True)
        jacobian_row = torch.cat([g.view(-1).detach() for g in gradients])
        jacobian[i, :] = jacobian_row

    if structured_tensor:
        shape = list(output.shape)
        shape.append(-1) # last dimension can be inferred from the jacobian size
        jacobian = jacobian.view(shape)

    return jacobian


def compute_damped_gn_update(jacobian, output_error, damping):
    """
    Compute the damped Gauss-Newton update, based on the given jacobian and
    output error.
    Args:
        jacobian (torch.Tensor): 2D tensor containing the Jacobian of the
            flattened output with respect to the flattened parameters for which
            the GN update is computed.
        output_error (torch.Tensor): tensor containing the gradient of the loss
            with respect to the output layer of the network.
        damping (float): positive damping hyperparameter
    Returns: the damped Gauss-Newton update for the parameters for which the
        jacobian was computed.
    """
    if damping < 0:
        raise ValueError('Positive value for damping expected, got '
                         '{}'.format(damping))
    # The jacobian also flattens the  output dimension, so we need to do
    # the same.
    output_error = output_error.view(-1, 1).detach()

    if damping == 0:
        # if the damping is 0, the curvature matrix C=J^TJ can be
        # rank deficit. Therefore, it is numerically best to compute the
        # pseudo inverse explicitly and after that multiply with it.
        jacobian_pinv = torch.pinverse(jacobian)
        gn_updates = jacobian_pinv.mm(output_error)
    else:
        # If damping is greater than 0, the curvature matrix C will be
        # positive definite and symmetric. Numerically, it is the most
        # efficient to use the cholesky decomposition to compute the
        # resulting Gauss-newton updates

        # As (J^T*J + l*I)^{-1}*J^T = J^T*(JJ^T + l*I)^{-1}, we select
        # the one which is most computationally efficient, depending on
        # the number of rows and columns of J (we want to take the inverse
        # of the smallest possible matrix, as this is the most expensive
        # operation. Note that we solve a linear system with cholesky
        # instead of explicitly computing the inverse, as this is more
        # efficient.
        if jacobian.shape[0] >= jacobian.shape[1]:
            G = jacobian.t().mm(jacobian)
            C = G + damping * torch.eye(G.shape[0])
            C_cholesky = torch.cholesky(C)
            jacobian_error = jacobian.t().matmul(output_error)
            gn_updates = torch.cholesky_solve(jacobian_error, C_cholesky)
        else:
            G = jacobian.mm(jacobian.t())
            C = G + damping * torch.eye(G.shape[0])
            C_cholesky = torch.cholesky(C)
            inverse_error = torch.cholesky_solve(output_error, C_cholesky)
            gn_updates = jacobian.t().matmul(inverse_error)

    return gn_updates


def compute_angle(A, B):
    """
     Compute the angle between two tensors of the same size. The tensors will
     be flattened, after which the angle is computed.
    Args:
        A (torch.Tensor): First tensor
        B (torch.Tensor): Second tensor
    Returns: The angle between the two tensors in degrees
    """
    if contains_nan(A):
        print('tensor A contains nans:')
        print(A)
    if contains_nan(B):
        print('tensor B contains nans:')
        print(B)

    inner_product = torch.sum(A*B)  #equal to inner product of flattened tensors
    cosine = inner_product/(torch.norm(A, p='fro')*torch.norm(B, p='fro'))
    if contains_nan(cosine):
        print('cosine contains nans:')
        print('inner product: {}'.format(inner_product))
        print('norm A: {}'.format(torch.norm(A, p='fro')))
        print('norm B: {}'.format(torch.norm(B, p='fro')))

    if cosine > 1 and cosine < 1 + 1e-5:
        cosine = torch.Tensor([1.])
    angle = 180/np.pi*torch.acos(cosine)
    if contains_nan(angle):
        print('angle computation causes NANs. cosines:')
        print(cosine)
    return angle


def compute_average_batch_angle(A, B):
    """
    Compute the average of the angles between the mini-batch samples of A and B.
    If the samples of the mini-batch have more than one dimension (minibatch
    dimension not included), the tensors will first be flattened
    Args:
        A (torch.Tensor):  A tensor with as first dimension the mini-batch
            dimension
        B (torch.Tensor): A tensor of the same shape as A
    Returns: The average angle between the two tensors in degrees.
    """

    A = A.flatten(1, -1)
    B = B.flatten(1, -1)
    if contains_nan(A):
        print('tensor A contains nans in activation angles:')
        print(A)
    if contains_nan(B):
        print('tensor B contains nans in activation angles:')
        print(B)
    inner_products = torch.sum(A*B, dim=1)
    A_norms = torch.norm(A, p=2, dim=1)
    B_norms = torch.norm(B, p=2, dim=1)
    cosines = inner_products/(A_norms*B_norms)
    if contains_nan(cosines):
        print('cosines contains nans in activation angles:')
        print('inner product: {}'.format(inner_products))
        print('norm A: {}'.format(A_norms))
        print('norm B: {}'.format(B_norms))
    if torch.sum(A_norms == 0) > 0:
        print('A_norms contains zeros')
    if torch.sum(B_norms == 0) > 0:
        print('B_norms contains zeros')
    cosines = torch.min(cosines, torch.ones_like(cosines))
    angles = torch.acos(cosines)
    return 180/np.pi*torch.mean(angles)


class NetworkError(Exception):
    pass


def list_to_str(list_arg, delim=' '):
    """Convert a list of numbers into a string.
    Args:
        list_arg: List of numbers.
        delim (optional): Delimiter between numbers.
    Returns:
        List converted to string.
    """
    ret = ''
    for i, e in enumerate(list_arg):
        if i > 0:
            ret += delim
        ret += str(e)
    return ret


def str_to_list(string, delim=',', type='float'):
    """ Convert a str (that originated from a list) back
    to a list of floats."""

    if string[0] in ('[', '(') and string[-1] in (']', ')'):
        string = string[1:-1]
    if type == 'float':
        lst = [float(num) for num in string.split(delim)]
    elif type == 'int':
        lst = [int(num) for num in string.split(delim)]
    else:
        raise ValueError('type {} not recognized'.format(type))

    return lst


def setup_summary_dict(args):
    """Setup the summary dictionary that is written to the performance
    summary file (in the result folder).
    This method adds the keyword "summary" to `shared`.
    Args:
        config: Command-line arguments.
        shared: Miscellaneous data shared among training functions (summary dict
            will be added to this :class:`argparse.Namespace`).
        experiment: Type of experiment. See argument `experiment` of method
            :func:`probabilistic.prob_mnist.train_bbb.run`.
        mnet: Main network.
        hnet (optional): Hypernetwork.
    """
    if args.hpsearch:
        hpsearch_config = "none"

    summary = dict()

    if args.hpsearch:
        summary_keys = hpsearch_config._SUMMARY_KEYWORDS
    else:
        summary_keys = [
                        # 'acc_train',
                        'acc_train_last',
                        'acc_train_best',
                        # 'loss_train',
                        'loss_train_last',
                        'loss_train_best',
                        # 'acc_test',
                        'acc_test_last',
                        'acc_test_best',
                        'acc_val_last',
                        'acc_val_best',
                        'acc_test_val_best',
                        'acc_train_val_best',
                        'loss_test_val_best',
                        'loss_train_val_best',
                        'loss_val_best',
                        'epoch_best_loss',
                        'epoch_best_acc',
                        # 'loss_test',
                        'loss_test_last',
                        'loss_test_best',
                        'rec_loss',
                        'rec_loss_last',
                        'rec_loss_best',
                        'rec_loss_first',
                        'rec_loss_init',
                        # 'rec_loss_var',
                        'rec_loss_var_av',
                        'finished']


    for k in summary_keys:
        if k == 'finished':
            summary[k] = 0
        else:
            summary[k] = -1

    save_summary_dict(args, summary)

    return summary


def save_summary_dict(args, summary):
    """Write a text file in the result folder that gives a quick
    overview over the results achieved so far.
    Args:
        args (Namespace): command line inputs
        summary (dict): summary dictionary
    """

    if args.hpsearch:
        hpsearch_config = "none"
        summary_fn = hpsearch_config._SUMMARY_FILENAME
    else:
        summary_fn = 'performance_overview.txt'
    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)

    with open(os.path.join(args.out_dir, summary_fn), 'w') as f:
        for k, v in summary.items():
            if isinstance(v, list):
                f.write('%s %s\n' % (k, list_to_str(v)))
            elif isinstance(v, float):
                f.write('%s %f\n' % (k, v))
            elif isinstance(v, (np.ndarray, pd.DataFrame)):
                # we don't want to save arrays and dataframes to text files
                pass
            else:
                f.write('%s %d\n' % (k, v))


def get_av_reconstruction_loss(network):
    """ Get the average reconstruction loss of the network across its layers
    for the current mini-batch.
    Args:
        network (networks.DTPNetwork): network
    Returns (torch.Tensor):
        Tensor containing a scalar of the average reconstruction loss
    """
    reconstruction_losses = np.array([])

    for layer in network.layers[1:]:
        # print(layer.reconstruction_loss)
        reconstruction_losses = np.append(reconstruction_losses,
                                           layer.reconstruction_loss)

        reconstruction_losses = list(filter(lambda x: x != None, reconstruction_losses)) #FIXME: Probably slows everything down, but was needed because there is a None in the first iteration of DKDTP

    return np.mean(reconstruction_losses[:-1])


def int_to_one_hot(class_labels, nb_classes, device, soft_target=1.):
    """ Convert tensor containing a batch of class indexes (int) to a tensor
    containing one hot vectors."""
    one_hot_tensor = torch.zeros((class_labels.shape[0], nb_classes),
                                 device=device)
    for i in range(class_labels.shape[0]):
        one_hot_tensor[i, class_labels[i]] = soft_target

    return one_hot_tensor


def one_hot_to_int(one_hot_tensor):
    return torch.argmax(one_hot_tensor, 1)


def dict2csv(dct, file_path):
    with open(file_path, 'w') as f:
        for key in dct.keys():
            f.write("{}, {} \n".format(key, dct[key]))


def process_lr(lr_str):
    """
    Process the lr provided by argparse.
    Args:
        lr_str (str): a string containing either a single float indicating the
            learning rate, or a list of learning rates, one for each layer of
            the network.
    Returns: a float or a numpy array of learning rates
    """
    if ',' in lr_str:
        return np.array(str_to_list(lr_str, ','))
    else:
        return float(lr_str)

def process_hdim(hdim_str):
    if ',' in hdim_str:
        return str_to_list(hdim_str, ',', type='int')
    else:
        return int(hdim_str)


def process_hdim_fb(hdim_str):
    if ',' in hdim_str:
        return str_to_list(hdim_str, ',', type='int')
    else:
        return [int(hdim_str)]


def check_gpu():
    try:
        name = torch.cuda.current_device()
        print("Using CUDA device {}.".format(torch.cuda.get_device_name(name)))
    except AssertionError:
        print("No CUDA device found.")


def contains_nan(tensor):
    # if not isinstance(tensor, torch.Tensor):
    #     print('input is not a tensor but {}'.format(type(tensor)))
    #     tensor = torch.Tensor(tensor)
    nb_nans = tensor != tensor
    nb_infs = tensor == float('inf')
    if isinstance(nb_nans, bool):
        return nb_nans or nb_infs
    else:
        return torch.sum(nb_nans) > 0 or torch.sum(nb_infs) > 0


def logit(x):
    if torch.sum(x < 1e-12) > 0 or torch.sum(x > 1 - 1e-12) > 0:
        warnings.warn('Input to inverse sigmoid is out of'
                      'bound: x={}'.format(x))
    inverse_sigmoid = torch.log(x / (1 - x))
    if contains_nan(inverse_sigmoid):
        raise ValueError('inverse sigmoid function outputted a NaN')
    return torch.log(x / (1 - x))


def plot_loss(summary, logdir, logplot=False):
    plt.figure()
    plt.plot(summary['loss_train'])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Train loss')
    if logplot:
        plt.yscale('log')
    plt.savefig(os.path.join(logdir, 'loss_train.svg'))
    plt.close()
    plt.figure()
    plt.plot(summary['loss_test'])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Test loss')
    if logplot:
        plt.yscale('log')
    plt.savefig(os.path.join(logdir, 'loss_test.svg'))
    plt.close()


def make_plot_output_space(args, net, i, loss_function,
                           targets, inputs, steps=20):
    """
    Make a plot of how the output activations would change if the update 
    for the parameters of layer(s) i is applied with a varying stepsize from 
    zero to one. 
    Args:
        args: command line arguments
        net: network
        i: layer index. If None, all layers are updated
        loss_function: loss function
        targets: true labels for the current batch
        inputs: batch with inputs for the network
        steps: amount of interpolation steps
    Returns: Saves a plot and the sequence of activations
    """

    if args.output_space_plot_bp:
        args.network_type = 'BP'

    # take the first input sample from the batch
    inputs = inputs.flatten(1, -1)
    inputs = inputs[0:1, :]
    targets = targets[0:1, :]

    # Get the parameters
    if i is None:
        parameters = net.get_forward_parameter_list()

    else:
        parameters = net.layers[i].get_forward_parameter_list()

    alpha = 1e-5
    sgd_optimizer = torch.optim.SGD(parameters, lr=alpha)
    sgd_optimizer.zero_grad()

    # compute update based on the single input sample
    predictions = net.forward(inputs)
    loss = loss_function(predictions, targets)

    if args.output_space_plot_bp:
        gradients = torch.autograd.grad(loss, parameters)
        for i, param in enumerate(parameters):
            param.grad = gradients[i].detach()
    else:
        net.backward(loss, args.target_stepsize, save_target=False,
                     norm_ratio=args.norm_ratio)


    # compute the start output value
    # output_traj = np.empty((steps + 1, 2))
    output_start = net.forward(inputs)

    # compute the output value after a very small step size
    sgd_optimizer.step()
    output_next = net.forward(inputs)

    output_update = (output_next - output_start)[0, 0:2].detach().cpu().numpy()

    # Make the plot
    ax = plt.axes()
    plot_contours(output_start[0, 0:2], targets[0, 0:2], loss_function, ax)

    # dimensions
    distance = np.linalg.norm(output_start.detach().cpu().numpy() -
                              targets.detach().cpu().numpy())
    x_low = targets[0, 0].detach().cpu().numpy() - 1.1 * distance
    x_high = targets[0, 0].detach().cpu().numpy() + 1.1 * distance
    y_low = targets[0, 1].detach().cpu().numpy() - 1.1 * distance
    y_high = targets[0, 1].detach().cpu().numpy() + 1.1 * distance

    plt.ylim(y_low, y_high)
    plt.xlim(x_low, x_high)

    # make the output arrow:
    output_arrow = distance / 2 / np.linalg.norm(output_update) * output_update
    output_arrow_start = output_start[0, 0:2].detach().cpu().numpy()

    ax.arrow(output_arrow_start[0], output_arrow_start[1],
              output_arrow[0], output_arrow[1],
              width=0.05,
              head_width=0.3
              )

    file_name = 'output_space_updates_fig_' + args.network_type + '.svg'
    plt.savefig(os.path.join(args.out_dir, file_name))
    plt.close()

    file_name = 'output_arrow_' + args.network_type + '.npy'
    np.save(os.path.join(args.out_dir, file_name),
            output_arrow)
    file_name = 'output_arrow_start_' + args.network_type + '.npy'
    np.save(os.path.join(args.out_dir, file_name),
            output_arrow_start)
    file_name = 'output_space_label_' + args.network_type + '.npy'
    np.save(os.path.join(args.out_dir, file_name),
            targets[0, 0:2].detach().cpu().numpy())


def plot_contours(y, label, loss_function, ax, fontsize=26):
    """
    Make a 2D contour plot of loss_function(y, targets)
    """
    gridpoints = 100

    distance = np.linalg.norm(y.detach().cpu().numpy() -
                              label.detach().cpu().numpy())
    y1 = np.linspace(label[0].detach().cpu().numpy() - 1.1*distance,
                     label[0].detach().cpu().numpy() + 1.1*distance,
                     num=gridpoints)
    y2 = np.linspace(label[1].detach().cpu().numpy() - 1.1*distance,
                     label[1].detach().cpu().numpy() + 1.1*distance,
                     num=gridpoints)

    Y1, Y2 = np.meshgrid(y1, y2)

    L = np.zeros(Y1.shape)
    for i in range(gridpoints):
        for j in range(gridpoints):
            y_sample = torch.Tensor([Y1[i,j], Y2[i, j]])
            L[i,j] = loss_function(y_sample, label).item()

    levels = np.linspace(1.01*L.min(), L.max(), num=9)

    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    CS = ax.contour(Y1, Y2, L, levels=levels)


def make_plot_output_space_bp(args, net, i, loss_function,
                           targets, inputs, steps=20):
    """
    Make a plot of how the output activations would change if the update
    for the parameters of layer(s) i is applied with a varying stepsize from
    zero to one.
    Args:
        args: command line arguments
        net: network
        i: layer index. If None, all layers are updated
        loss_function: loss function
        targets: true labels for the current batch
        inputs: batch with inputs for the network
        steps: amount of interpolation steps
    Returns: Saves a plot and the sequence of activations
    """

    # Get the parameters
    if i is None:
        parameters = net.parameters()

    else:
        parameters = net.layers[i].parameters()

    # create sgd optimizer
    alpha = 1e-5
    sgd_optimizer = torch.optim.SGD(parameters, lr=alpha)
    sgd_optimizer.zero_grad()

    # take the first input sample from the batch
    inputs = inputs.flatten(1, -1)
    inputs = inputs[0:1, :]
    targets = targets[0:1, :]

    # compute update based on the single input sample
    predictions = net(inputs)
    loss = loss_function(predictions, targets)
    loss.backward()



    # Interpolate the output trajectory
    # output_traj = np.empty((steps + 1, 2))
    output_start = net(inputs)
    # output_traj[0, :] = output_start[0, 0:2].detach().cpu().numpy()

    sgd_optimizer.step()
    output_next = net(inputs)

    output_update = (output_next - output_start)[0, 0:2].detach().cpu().numpy()

    # Make the plot
    ax = plt.axes()
    plot_contours(output_start[0, 0:2], targets[0, 0:2], loss_function, ax)

    # dimensions
    distance = np.linalg.norm(output_start.detach().cpu().numpy() -
                              targets.detach().cpu().numpy())
    x_low = targets[0,0].detach().cpu().numpy() - 1.1 * distance
    x_high = targets[0, 0].detach().cpu().numpy() + 1.1 * distance
    y_low = targets[0, 1].detach().cpu().numpy() - 1.1 * distance
    y_high = targets[0, 1].detach().cpu().numpy() + 1.1 * distance

    plt.ylim(y_low, y_high)
    plt.xlim(x_low, x_high)

    # make the output arrow:
    output_arrow = distance / 2 / np.linalg.norm(output_update) * output_update
    output_arrow_start = output_start[0, 0:2].detach().cpu().numpy()

    ax.arrow(output_arrow_start[0], output_arrow_start[1],
              output_arrow[0], output_arrow[1],
              width=0.05,
              head_width=0.3
              )

    file_name = 'output_space_updates_fig_' + args.network_type + '.svg'
    plt.savefig(os.path.join(args.out_dir, file_name))
    plt.close()
    # file_name = 'output_space_updates_traj_' + args.network_type + '.npy'
    # np.save(os.path.join(args.out_dir, file_name), output_traj)
    file_name = 'output_arrow_' + args.network_type + '.npy'
    np.save(os.path.join(args.out_dir, file_name),
            output_arrow)
    file_name = 'output_arrow_start_' + args.network_type + '.npy'
    np.save(os.path.join(args.out_dir, file_name),
            output_arrow_start)
    file_name = 'output_space_label_' + args.network_type + '.npy'
    np.save(os.path.join(args.out_dir, file_name),
            targets[0, 0:2].detach().cpu().numpy())


def nullspace(A, tol=1e-12):
    U, S, V = torch.svd(A, some=False)
    if S.min() >= tol:
        null_start = len(S)
    else:
        null_start = int(len(S) - torch.sum(S<tol))

    V_null = V[:, null_start:]
    return V_null


def nullspace_relative_norm(A, x, tol=1e-12):
    """
    Compute the ratio between the norm
    of components of x that are in the nullspace of A
    and the norm of x
    """

    if len(x.shape) == 1:
        x = x.unsqueeze(1)
    A_null = nullspace(A, tol=tol)
    x_null_coordinates = A_null.t().mm(x)
    ratio = x_null_coordinates.norm()/x.norm()
    return ratio




### DTP LAYERS
class DTPLayer(nn.Module):
    """ An abstract base class for a layer of an MLP that will be trained by the
    differece target propagation method. Child classes should specify which
    activation function is used.
    Attributes:
        weights (torch.Tensor): The forward weight matrix :math:`W` of the layer
        bias (torch.Tensor): The forward bias :math:`\mathbf{b}`
            of the layer.
            Attribute is ``None`` if argument ``bias`` was passed as ``None``
            in the constructor.
        feedback_weights (torch.Tensor): The feedback weight matrix :math:`Q`
            of the layer. Warning: if we use the notation of the theoretical
            framework, the feedback weights are actually $Q_{i-1}$ from the
            previous layer!! We do this because this makes the implementation
            of the reconstruction loss and training the feedback weights much
            easier (as g_{i-1} and hence Q_{i-1} needs to approximate
            f_i^{-1}). However for the direct feedback connection layers, it
            might be more logical to let the feedbackweights represent Q_i
            instead of Q_{i-1}, as now only direct feedback connections exist.
        feedback_bias (torch.Tensor): The feedback bias :math:`\mathbf{b}`
            of the layer.
            Attribute is ``None`` if argument ``bias`` was passed as ``None``
            in the constructor.
        forward_requires_grad (bool): Flag indicating whether the computational
            graph with respect to the forward parameters should be saved. This
            flag should be True if you want to compute BP or GN updates. For
            TP updates, computational graphs are not needed (custom
            implementation by ourselves)
        reconstruction_loss (float): The reconstruction loss of this layer
            evaluated at the current mini-batch.
        forward_activation (str): String indicating the forward nonlinear
            activation function used by the layer. Choices: 'tanh', 'relu',
            'linear'.
        feedback_activation (str): String indicating the feedback nonlinear
            activation function used by the layer. Choices: 'tanh', 'relu',
            'linear'.
        target (torch.Tensor or None): The target for this layer on the current
            minibatch. During normal training, it is not needed
            to save the targets so this attribute will stay None. If the user
            wants to compute the angle between (target - activation) and a
            BP update or GN update, the target needs to be saved in the layer
            object to use later on to compute the angles. The saving happens in
            the backward method of the network.
    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        bias (bool): If set to ``False``, the layer will not learn an additive
            bias.
        forward_requires_grad (bool): Flag indicating whether the forward
            parameters require gradients that can be computed with autograd.
            This might be needed when comparing the DTP updates with BP updates
            and GN updates.
        forward_activation (str): String indicating the forward nonlinear
            activation function used by the layer. Choices: 'tanh', 'relu',
            'linear'.
        feedback_activation (str): String indicating the feedback nonlinear
            activation function used by the layer. Choices: 'tanh', 'relu',
            'linear'.
    """

    def __init__(self, in_features, out_features, bias=True,
                 forward_requires_grad=False, forward_activation='tanh',
                 feedback_activation='tanh', initialization='orthogonal'):
        nn.Module.__init__(self)

        self._weights = nn.Parameter(torch.Tensor(out_features, in_features),
                                     requires_grad=forward_requires_grad)
        self._feedbackweights = nn.Parameter(torch.Tensor(in_features,
                                                          out_features),
                                             requires_grad=False)
        if bias:
            self._bias = nn.Parameter(torch.Tensor(out_features),
                                      requires_grad=forward_requires_grad)
            self._feedbackbias = nn.Parameter(torch.Tensor(in_features),
                                              requires_grad=False)
        else:
            self._bias = None
            self._feedbackbias = None

        # Initialize the weight matrices following Lee 2015
        # TODO: try other initializations, such as the special one to mimic
        # batchnorm
        if initialization == 'orthogonal':
            gain = np.sqrt(6. / (in_features + out_features))
            nn.init.orthogonal_(self._weights, gain=gain)
            nn.init.orthogonal_(self._feedbackweights, gain=gain)
        elif initialization == 'xavier':
            nn.init.xavier_uniform_(self._weights)
            nn.init.xavier_uniform_(self._feedbackweights)
        elif initialization == 'xavier_normal':
            nn.init.xavier_normal_(self._weights)
            nn.init.xavier_normal_(self._feedbackweights)
        elif initialization == 'teacher':
            nn.init.xavier_normal_(self._weights, gain=3.)
            nn.init.xavier_normal_(self._feedbackweights)
        else:
            raise ValueError('Provided weight initialization "{}" is not '
                             'supported.'.format(initialization))

        if bias:
            nn.init.constant_(self._bias, 0)
            nn.init.constant_(self._feedbackbias, 0)

        self._activations = None
        self._linearactivations = None
        self._reconstruction_loss = None
        self._forward_activation = forward_activation
        self._feedback_activation = feedback_activation
        self._target = None

    @property
    def weights(self):
        """Getter for read-only attribute :attr:`weights`."""
        return self._weights

    @property
    def bias(self):
        """Getter for read-only attribute :attr:`bias`."""
        return self._bias

    @property
    def feedbackweights(self):
        """Getter for read-only attribute :attr:`feedbackweights`."""
        return self._feedbackweights

    @property
    def feedbackbias(self):
        """Getter for read-only attribute :attr:`feedbackbias`."""
        return self._feedbackbias

    @property
    def activations(self):
        """Getter for read-only attribute :attr:`activations` """
        return self._activations

    @activations.setter
    def activations(self, value):
        """ Setter for the attribute activations"""
        self._activations = value

    @property
    def linearactivations(self):
        """Getter for read-only attribute :attr:`linearactivations` """
        return self._linearactivations

    @linearactivations.setter
    def linearactivations(self, value):
        """Setter for the attribute :attr:`linearactivations` """
        self._linearactivations = value

    @property
    def reconstruction_loss(self):
        """ Getter for attribute reconstruction_loss."""
        return self._reconstruction_loss

    @reconstruction_loss.setter
    def reconstruction_loss(self, value):
        """ Setter for attribute reconstruction_loss."""
        self._reconstruction_loss = value

    @property
    def forward_activation(self):
        """ Getter for read-only attribute forward_activation"""
        return self._forward_activation

    @property
    def feedback_activation(self):
        """ Getter for read-only attribute feedback_activation"""
        return self._feedback_activation

    @property
    def target(self):
        """ Getter for attribute target"""
        return self._target

    @target.setter
    def target(self, value):
        """ Setter for attribute target"""
        self._target = value

    def get_forward_parameter_list(self):
        """ Return forward weights and forward bias if applicable"""
        parameterlist = []
        parameterlist.append(self.weights)
        if self.bias is not None:
            parameterlist.append(self.bias)
        return parameterlist

    def forward_activationfunction(self, x):
        """ Element-wise forward activation function"""
        if self.forward_activation == 'tanh':
            return torch.tanh(x)
        elif self.forward_activation == 'relu':
            return F.relu(x)
        elif self.forward_activation == 'linear':
            return x
        elif self.forward_activation == 'leakyrelu':
            return F.leaky_relu(x, 0.2)
        elif self.forward_activation == 'sigmoid':
            return torch.sigmoid(x)
        else:
            raise ValueError('The provided forward activation {} is not '
                             'supported'.format(self.forward_activation))

    def feedback_activationfunction(self, x):
        """ Element-wise feedback activation function"""
        if self.feedback_activation == 'tanh':
            return torch.tanh(x)
        elif self.feedback_activation == 'relu':
            return F.relu(x)
        elif self.feedback_activation == 'linear':
            return x
        elif self.feedback_activation == 'leakyrelu':
            return F.leaky_relu(x, 5)
        elif self.feedback_activation == 'sigmoid':
            if torch.sum(x < 1e-12) > 0 or torch.sum(x > 1-1e-12) > 0:
                warnings.warn('Input to inverse sigmoid is out of'
                                 'bound: x={}'.format(x))
            inverse_sigmoid = torch.log(x/(1-x))
            if contains_nan(inverse_sigmoid):
                raise ValueError('inverse sigmoid function outputted a NaN')
            return torch.log(x/(1-x))
        else:
            raise ValueError('The provided feedback activation {} is not '
                             'supported'.format(self.feedback_activation))

    def compute_vectorized_jacobian(self, a):
        """ Compute the vectorized Jacobian of the forward activation function,
        evaluated at a. The vectorized Jacobian is the vector with the diagonal
        elements of the real Jacobian, as it is a diagonal matrix for element-
        wise functions. As a is a minibatch, the output will also be a
        mini-batch of vectorized Jacobians (thus a matrix).
        Args:
            a (torch.Tensor): linear activations
        """
        if self.forward_activation == 'tanh':
            return 1. - torch.tanh(a)**2
        elif self.forward_activation == 'relu':
            J = torch.ones_like(a)
            J[a < 0.] = 0.
            return J
        elif self.forward_activation == 'leakyrelu':
            J = torch.ones_like(a)
            J[a < 0.] = 0.2
            return J
        elif self.forward_activation == 'linear':
            return torch.ones_like(a)
        elif self.forward_activation == 'sigmoid':
            s = torch.sigmoid(a)
            return s * (1 - s)
        else:
            raise ValueError('The provided forward activation {} is not '
                             'supported'.format(self.forward_activation))

    def requires_grad(self):
        """ Set require_grad attribute of the activations of this layer to
        True, such that the gradient will be saved in the activation tensor."""
        self._activations.requires_grad = True

    def forward(self, x):
        """Compute the output activation of the layer.
        This method applies first a linear mapping with the
        parameters ``weights`` and ``bias``, after which it applies the
        forward activation function.
        Args:
            x: A mini-batch of size B x in_features with input activations from
            the previous layer or input.
        Returns:
            The mini-batch of output activations of the layer.
        """

        h = x.mm(self.weights.t())
        if self.bias is not None:
            h += self.bias.unsqueeze(0).expand_as(h)
        self.linearactivations = h

        self.activations = self.forward_activationfunction(h)
        return self.activations

    def dummy_forward(self, x):
        """ Same as the forward method, besides that the activations and
        linear activations are not saved in self."""
        h = x.mm(self.weights.t())
        if self.bias is not None:
            h += self.bias.unsqueeze(0).expand_as(h)
        h = self.forward_activationfunction(h)
        return h

    def dummy_forward_linear(self, x):
        """ Propagate the input of the layer forward to the linear activation
        of the current layer (so no nonlinearity applied), without saving the
        linear activations."""
        a = x.mm(self.weights.t())
        if self.bias is not None:
            a += self.bias.unsqueeze(0).expand_as(a)

        return a

    def propagate_backward(self, h):
        """ Propagate the activation h backward through the backward mapping
        function g(h) = t(Q*h + d)
        Args:
            h (torch.Tensor): a mini-batch of activations
        """
        h = h.mm(self.feedbackweights.t())
        if self.feedbackbias is not None:
            h += self.feedbackbias.unsqueeze(0).expand_as(h)
        return self.feedback_activationfunction(h)


    def backward(self, h_target, h_previous, h_current):
        """Compute the target activation for the previous layer, based on the
        provided target.
        Args:
            h_target: a mini-batch of the provided targets for this layer.
            h_previous: the activations of the previous layer, used for the
                DTP correction term.
            h_current: the activations of the current layer, used for the
                DTP correction term.
        Returns:
            h_target_previous: The mini-batch of target activations for
                the previous layer.
        """

        h_target_previous = self.propagate_backward(h_target)
        h_tilde_previous = self.propagate_backward(h_current)
        h_target_previous = h_target_previous + h_previous - h_tilde_previous

        return h_target_previous

    def compute_forward_gradients(self, h_target, h_previous, norm_ratio=1.):
        """ Compute the gradient of the forward weights and bias, based on the
        local mse loss between the layer activation and layer target.
        The gradients are saved in the .grad attribute of the forward weights
        and forward bias.
        Args:
            h_target (torch.Tensor): the DTP target of the current layer
            h_previous (torch.Tensor): the rate activation of the previous
                layer
            norm_ratio (float): Depreciated.
        """

        if self.forward_activation == 'linear':
            teaching_signal = 2 * (self.activations - h_target)
        else:
            vectorized_jacobians = self.compute_vectorized_jacobian(
                self.linearactivations)
            teaching_signal = 2 * vectorized_jacobians * (
                    self.activations - h_target)
        batch_size = h_target.shape[0]
        bias_grad = teaching_signal.mean(0)
        weights_grad = 1./batch_size * teaching_signal.t().mm(h_previous)

        if self.bias is not None:
            self._bias.grad = bias_grad.detach()
        self._weights.grad = weights_grad.detach()

    def set_feedback_requires_grad(self, value):
        """
        Sets the 'requires_grad' attribute of the feedback weights and bias to
        the given value
        Args:
            value (bool): True or False
        """
        if not isinstance(value, bool):
            raise TypeError('The given value should be a boolean.')
        self._feedbackweights.requires_grad = value
        if self._feedbackbias is not None:
            self._feedbackbias.requires_grad = value

    def compute_feedback_gradients(self, h_previous_corrupted, sigma):
        """ Compute the gradient of the backward weights and bias, based on the
        local reconstruction loss of a corrupted sample of the previous layer
        activation. The gradients are saved in the .grad attribute of the
        feedback weights and feedback bias."""

        self.set_feedback_requires_grad(True)

        h_current = self.dummy_forward(h_previous_corrupted)
        h = self.propagate_backward(h_current)

        if sigma < 0.02:
            scale = 1/0.02**2
        else:
            scale = 1/sigma**2
        reconstruction_loss = scale * F.mse_loss(h, h_previous_corrupted)

        self.save_feedback_gradients(reconstruction_loss)

        self.set_feedback_requires_grad(False)

    def save_feedback_gradients(self, reconstruction_loss):
        """
        Compute the gradients of the reconstruction_loss with respect to the
        feedback parameters by help of autograd and save them in the .grad
        attribute of the feedback parameters
        Args:
            reconstruction_loss: the reconstruction loss
        """
        self.reconstruction_loss = reconstruction_loss.item()
        if self.feedbackbias is not None:
            grads = torch.autograd.grad(reconstruction_loss, [
                self.feedbackweights, self.feedbackbias], retain_graph=False)
            self._feedbackbias.grad = grads[1].detach()
        else:
            grads = torch.autograd.grad(reconstruction_loss,
                                        self.feedbackweights,
                                        retain_graph=False
                                        )
        self._feedbackweights.grad = grads[0].detach()

    def compute_bp_update(self, loss, retain_graph=False):
        """ Compute the error backpropagation update for the forward
        parameters of this layer, based on the given loss.
        Args:
            loss (nn.Module): network loss
            retain_graph (bool): flag indicating whether the graph of the
                network should be retained after computing the gradients or
                jacobians. If the graph will not be used anymore for the current
                minibatch afterwards, retain_graph should be False.
        """

        if self.bias is not None:
            grads = torch.autograd.grad(loss, [self.weights, self.bias],
                                        retain_graph=retain_graph)
        else:
            grads = torch.autograd.grad(loss, self.weights,
                                        retain_graph=retain_graph)

        return grads

    def compute_gn_update(self, output_activation, loss, damping=0.,
                          retain_graph=False):
        """
        Compute the Gauss Newton update for the parameters of this layer based
        on the current minibatch.
        Args:
            output_activation (torch.Tensor): The tensor containing the output
                activations of the network for the current mini-batch
            loss (torch.Tensor): The 0D tensor containing the loss value of the
                current mini-batch.
            damping (float): the damping coefficient to damp the GN curvature
                matrix J^TJ. Default: 0.
            retain_graph (bool): flag indicating whether the graph of the
                network should be retained after computing the gradients or
                jacobians. If the graph will not be used anymore for the current
                minibatch afterwards, retain_graph should be False.
        Returns (tuple): A tuple containing the Gauss Newton updates for the
            forward parameters (at index 0 the weight updates, at index 1
            the bias updates if the layer has a bias)
        """
        output_error = torch.autograd.grad(loss, output_activation,
                                           retain_graph=True)[0].detach()
        parameters = self.get_forward_parameters()
        jacobian = compute_jacobian(parameters, output_activation,
                                          retain_graph=retain_graph)

        gn_updates = compute_damped_gn_update(jacobian, output_error,
                                                    damping)

        if self.bias is not None:
            weight_update_flattened = gn_updates[:self.weights.numel(), :]
            bias_update_flattened = gn_updates[self.weights.numel():, :]
            weight_update = weight_update_flattened.view_as(self.weights)
            bias_update = bias_update_flattened.view_as(self.bias)
            return (weight_update, bias_update)
        else:
            weight_update = gn_updates.view(self.weights.shape)
            return (weight_update, )

    def compute_gn_activation_updates(self, output_activation, loss,
                                      damping=0., retain_graph=False,
                                      linear=False):
        """
        Compute the Gauss Newton update for activations of the layer. Target
        propagation tries to approximate these updates by the difference between
        the layer targets and the layer activations.
        Args:
            output_activation (torch.Tensor): The tensor containing the output
                activations of the network for the current mini-batch
            loss (torch.Tensor): The 0D tensor containing the loss value of the
                current mini-batch.
            damping (float): the damping coefficient to damp the GN curvature
                matrix J^TJ. Default: 0.
            retain_graph (bool): flag indicating whether the graph of the
                network should be retained after computing the gradients or
                jacobians. If the graph will not be used anymore for the current
                minibatch afterwards, retain_graph should be False.
            linear (bool): Flag indicating whether the GN update for the
                linear activations should be computed instead of for the
                nonlinear activations.
        Returns (torch.Tensor): A tensor containing the Gauss-Newton updates
            for the layer activations of the current mini-batch. The size is
            minibatchsize x layersize
        """
        output_error = torch.autograd.grad(loss, output_activation,
                                           retain_graph=True)[0].detach()
        if linear:
            activations = self.linearactivations
        else:
            activations = self.activations
        activations_updates = torch.Tensor(activations.shape)
        layersize = activations.shape[1]

        # compute the GN update for each batch sample separate, as we are now
        # computing 'updates' for the activations of the layer instead of the
        # parameters of the layers
        for batch_idx in range(activations.shape[0]):
            # print(batch_idx)
            #  compute jacobian for one batch sample:
            if batch_idx == activations.shape[0] - 1:
                retain_graph_flag = retain_graph
            else:
                # if not yet at the end of the batch, we should retain the graph
                # used for computing the jacobian, as the graph needs to be
                # reused for the computing the jacobian of the next batch sample
                retain_graph_flag = True
            jacobian = compute_jacobian(activations,
                                              output_activation[batch_idx,
                                              :],
                                            retain_graph=retain_graph_flag)
            # torch.autograd.grad only accepts the original input tensor,
            # not a subpart of it. Thus we compute the jacobian to all the
            # batch samples from activations and then select the correct
            # part of it
            jacobian = jacobian[:, batch_idx*layersize:
                                   (batch_idx+1)*layersize]

            gn_updates = compute_damped_gn_update(jacobian,
                                                output_error[batch_idx, :],
                                                        damping)
            activations_updates[batch_idx, :] = gn_updates.view(-1)
        return activations_updates

    def compute_bp_activation_updates(self, loss, retain_graph=False,
                                      linear=False):
        """ Compute the error backpropagation teaching signal for the
        activations of this layer, based on the given loss.
        Args:
            loss (nn.Module): network loss
            retain_graph (bool): flag indicating whether the graph of the
                network should be retained after computing the gradients or
                jacobians. If the graph will not be used anymore for the current
                minibatch afterwards, retain_graph should be False.
            linear (bool): Flag indicating whether the GN update for the
                linear activations should be computed instead of for the
                nonlinear activations.
        Returns (torch.Tensor): A tensor containing the BP updates for the layer
            activations for the current mini-batch.
                """

        if linear:
            activations = self.linearactivations
        else:
            activations = self.activations
        grads = torch.autograd.grad(loss, activations,
                                    retain_graph=retain_graph)[0].detach()
        return grads

    def compute_gnt_updates(self, output_activation, loss, h_previous, damping=0.,
                            retain_graph=False, linear=False):
        """ Compute the angle with the GNT updates for the parameters of the
        network."""
        gn_activation_update = self.compute_gn_activation_updates(output_activation=output_activation,
                                                                  loss=loss,
                                                                  damping=damping,
                                                                  retain_graph=retain_graph,
                                                                  linear=linear)

        if not linear:
            vectorized_jacobians = self.compute_vectorized_jacobian(
                self.linearactivations)
            teaching_signal = 2 * vectorized_jacobians * (
                        gn_activation_update)
        else:
            teaching_signal = 2 * gn_activation_update

        batch_size = self.activations.shape[0]
        bias_grad = teaching_signal.mean(0)
        weights_grad = 1. / batch_size * teaching_signal.t().mm(h_previous)

        if self.bias is not None:
            return (weights_grad, bias_grad)
        else:
            return (weights_grad, )

    def compute_nullspace_relative_norm(self, output_activation, retain_graph=False):
        """ Compute the norm of the components of weights.grad that are in the nullspace
        of the jacobian of the output with respect to weights, relative to the norm of
        weights.grad."""
        J = compute_jacobian(self.weights, output_activation,
                                   structured_tensor=False,
                                   retain_graph=retain_graph)
        weights_update_flat = self.weights.grad.view(-1)
        relative_norm = nullspace_relative_norm(J, weights_update_flat)
        return relative_norm

    def save_logs(self, writer, step, name, no_gradient=False,
                  no_fb_param=False):
        """
        Save logs and plots of this layer on tensorboardX
        Args:
            writer (SummaryWriter): summary writer from tensorboardX
            step (int): the global step used for the x-axis of the plots
            name (str): The name of the layer
            no_gradient (bool): flag indicating whether we should skip saving
                the gradients of the feedback weights.
            no_fb_param (bool): don't log the feedback parameters
        """
        forward_weights_norm = torch.norm(self.weights)
        writer.add_scalar(tag='{}/forward_weights_norm'.format(name),
                          scalar_value=forward_weights_norm,
                          global_step=step)
        if self.weights.grad is not None:
            forward_weights_gradients_norm = torch.norm(self.weights.grad)
            writer.add_scalar(tag='{}/forward_weights_gradients_norm'.format(name),
                              scalar_value=forward_weights_gradients_norm,
                              global_step=step)
        if self.bias is not None:
            forward_bias_norm = torch.norm(self.bias)

            writer.add_scalar(tag='{}/forward_bias_norm'.format(name),
                              scalar_value=forward_bias_norm,
                              global_step=step)
            if self.bias.grad is not None:
                forward_bias_gradients_norm = torch.norm(self.bias.grad)
                writer.add_scalar(tag='{}/forward_bias_gradients_norm'.format(name),
                                  scalar_value=forward_bias_gradients_norm,
                                  global_step=step)
        if not no_fb_param:
            feedback_weights_norm = torch.norm(self.feedbackweights)
            writer.add_scalar(tag='{}/feedback_weights_norm'.format(name),
                              scalar_value=feedback_weights_norm,
                              global_step=step)
            if self.feedbackbias is not None:
                feedback_bias_norm = torch.norm(self.feedbackbias)
                writer.add_scalar(tag='{}/feedback_bias_norm'.format(name),
                                  scalar_value=feedback_bias_norm,
                                  global_step=step)

            if not no_gradient and self.feedbackweights.grad is not None:
                feedback_weights_gradients_norm = torch.norm(
                    self.feedbackweights.grad)
                writer.add_scalar(
                    tag='{}/feedback_weights_gradients_norm'.format(name),
                    scalar_value=feedback_weights_gradients_norm,
                    global_step=step)
                if self.feedbackbias is not None:
                    feedback_bias_gradients_norm = torch.norm(
                        self.feedbackbias.grad)
                    writer.add_scalar(
                        tag='{}/feedback_bias_gradients_norm'.format(name),
                        scalar_value=feedback_bias_gradients_norm,
                        global_step=step)

    def save_feedback_batch_logs(self, writer, step, name, no_gradient=False,
                                 init=False):
        """
        Save logs for one minibatch.
        Args:
            writer (SummaryWriter): summary writer from tensorboardX
            step (int): the global step used for the x-axis of the plots
            name (str): The name of the layer
            no_gradient (bool): flag indicating whether we should skip saving
                the gradients of the feedback weights.
            init (bool): flag indicating that the training is in the
                initialization phase (only training the feedback weights).
        """
        if not init:
            if not no_gradient and self.reconstruction_loss is not None:
                writer.add_scalar(
                    tag='{}/reconstruction_loss'.format(name),
                    scalar_value=self.reconstruction_loss,
                    global_step=step)
        else:
            if not no_gradient and self.reconstruction_loss is not None:
                writer.add_scalar(
                    tag='{}/reconstruction_loss_init'.format(name),
                    scalar_value=self.reconstruction_loss,
                    global_step=step)

    def get_forward_parameters(self):
        """ Return a list containing the forward parameters."""
        if self.bias is not None:
            return [self.weights, self.bias]
        else:
            return [self.weights]

    def get_forward_gradients(self):
        """ Return a tuple containing the gradients of the forward
        parameters."""

        if self.bias is not None:
            return (self.weights.grad, self.bias.grad)
        else:
            return (self.weights.grad, )




#### NETWORKS
class DTPNetwork(nn.Module):
    """ A multilayer perceptron (MLP) network that will be trained by the
    difference target propagation (DTP) method.
    Attributes:
        layers (nn.ModuleList): a ModuleList with the layer objects of the MLP
        depth: the depth of the network (# hidden layers + 1)
        input (torch.Tensor): the input minibatch of the current training
                iteration. We need
                to save this tensor for computing the weight updates for the
                first hidden layer
        sigma: standard deviation of the gaussian that corrupts layer
                activations for computing the reconstruction losses.
        update_idx (None or int): the layer index of which the layer parameters
            are updated for the current mini-batch, when working in a randomized
            setting. If the randomized setting is not used, it is equal to None.
    Args:
        n_in: input dimension (flattened input assumed)
        n_hidden: list with hidden layer dimensions
        n_out: output dimension
        activation: activation function indicator for the hidden layers
        output_activation: activation function indicator for the output layer
        bias: boolean indicating whether the network uses biases or not
        sigma: standard deviation of the gaussian that corrupts layer
                activations for computing the reconstruction losses.
        forward_requires_grad (bool): Flag indicating whether the forward
            parameters require gradients that can be computed with autograd.
            This might be needed when comparing the DTP updates with BP updates
            and GN updates.
        initialization (str): the initialization method used for the forward
                and feedback matrices of the layers
    """

    def __init__(self, n_in, n_hidden, n_out, activation='relu',
                 output_activation='linear', bias=True, sigma=0.36,
                 forward_requires_grad=False,
                 initialization='orthogonal',
                 fb_activation='relu',
                 plots=None):
        nn.Module.__init__(self)

        self._depth = len(n_hidden) + 1
        self._layers = self.set_layers(n_in, n_hidden, n_out, activation,
                                       output_activation, bias,
                                       forward_requires_grad,
                                       initialization,
                                       fb_activation)
        self._input = None
        self._sigma = sigma
        self._forward_requires_grad = forward_requires_grad
        self._update_idx = None
        self._plots = plots
        if plots is not None:
            self.bp_angles = pd.DataFrame(columns=[i for i in range(0, self._depth)])
            self.gn_angles = pd.DataFrame(columns=[i for i in range(0, self._depth)])
            self.gnt_angles = pd.DataFrame(columns=[i for i in range(0, self._depth)])
            self.bp_activation_angles = pd.DataFrame(columns=[i for i in range(0, self._depth)])
            self.gn_activation_angles = pd.DataFrame(columns=[i for i in range(0, self._depth)])

            self.reconstruction_loss_init = pd.DataFrame(columns=[i for i in range(0, self._depth)])
            self.reconstruction_loss = pd.DataFrame(columns=[i for i in range(0, self._depth)])

            self.td_activation = pd.DataFrame(columns=[i for i in range(0, self._depth)])
            self.gn_activation = pd.DataFrame(columns=[i for i in range(0, self._depth)])
            self.bp_activation = pd.DataFrame(columns=[i for i in range(0, self._depth)])

            self.nullspace_relative_norm = pd.DataFrame(columns=[i for i in range(0, self._depth)])



    def set_layers(self, n_in, n_hidden, n_out, activation, output_activation,
                   bias, forward_requires_grad, initialization,
                   fb_activation):
        """
        Create the layers of the network and output them as a ModuleList.
        Args:
            n_in: input dimension (flattened input assumed)
            n_hidden: list with hidden layer dimensions
            n_out: output dimension
            activation: activation function indicator for the hidden layers
            output_activation: activation function indicator for the output
                layer
            bias: boolean indicating whether the network uses biases or not
            forward_requires_grad (bool): Flag indicating whether the forward
            parameters require gradients that can be computed with autograd.
            This might be needed when comparing the DTP updates with BP updates
            and GN updates.
            initialization (str): the initialization method used for the forward
                and feedback matrices of the layers
            fb_activation (str): activation function indicator for the feedback
                path of the hidden layers
        """
        n_all = [n_in] + n_hidden + [n_out]
        layers = nn.ModuleList()
        for i in range(1, len(n_all) - 1):
            layers.append(
                DTPLayer(n_all[i - 1], n_all[i], bias=bias,
                         forward_activation=activation,
                         feedback_activation=fb_activation,
                         forward_requires_grad=forward_requires_grad,
                         initialization=initialization
                         ))
        layers.append(DTPLayer(n_all[-2], n_all[-1], bias=bias,
                               forward_activation=output_activation,
                               feedback_activation=fb_activation,
                               forward_requires_grad=forward_requires_grad,
                               initialization=initialization))
        return layers

    @property
    def depth(self):
        """Getter for read-only attribute :attr:`depth`."""
        return self._depth

    @property
    def layers(self):
        """Getter for read-only attribute :attr:`layers`."""
        return self._layers

    @property
    def sigma(self):
        """ Getter for read-only attribute sigma"""
        return self._sigma

    @property
    def input(self):
        """ Getter for attribute input."""
        return self._input

    @input.setter
    def input(self, value):
        """ Setter for attribute input."""
        self._input = value

    @property
    def forward_requires_grad(self):
        """ Getter for read-only attribute forward_requires_grad"""
        return self._forward_requires_grad

    @property
    def update_idx(self):
        """ Getter for attribute update_idx"""
        return self._update_idx

    @update_idx.setter
    def update_idx(self, value):
        """Setter for attribute update_idx"""
        self._update_idx = value

    def forward(self, x):
        """ Propagate the input forward through the MLP network.
        Args:
            x: the input to the network
        returns:
            y: the output of the network
            """
        self.input = x
        y = x

        for layer in self.layers:
            y = layer.forward(y)

        # the output of the network requires a gradient in order to compute the
        # target (in compute_output_target() )
        if y.requires_grad == False:
            y.requires_grad = True

        return y

    def compute_output_target(self, loss, target_lr):
        """
        Compute the output target.
        Args:
            loss (nn.Module): output loss of the network
            target_lr: the learning rate for computing the output target based
                on the gradient of the loss w.r.t. the output layer
        Returns: Mini-batch of output targets
        """
        output_activations = self.layers[-1].activations

        gradient = torch.autograd.grad(loss, output_activations,
                                       retain_graph=self.forward_requires_grad)\
                                        [0].detach()
        output_targets = output_activations - \
                         target_lr*gradient
        return output_targets

    def propagate_backward(self, h_target, i):
        """
        Propagate the output target backwards to layer i in a DTP-like fashion.
        Args:
            h_target (torch.Tensor): the output target
            i: the layer index to which the target must be propagated
        Returns: the target for layer i
        """
        for k in range(self.depth-1, i, -1):
            h_current = self.layers[k].activations
            h_previous = self.layers[k-1].activations
            h_target = self.layers[k].backward(h_target, h_previous, h_current)
        return h_target

    def backward_random(self, loss, target_lr, i, save_target=False,
                        norm_ratio=1.):
        """ Propagate the output target backwards through the network until
        layer i. Based on this target, compute the gradient of the forward
        weights and bias of layer i and save them in the parameter tensors.
        Args:
            loss (nn.Module): output loss of the network
            target_lr: the learning rate for computing the output target based
                on the gradient of the loss w.r.t. the output layer
            i: layer index to which the target needs to be propagated and the
                gradients need to be computed
            save_target (bool): flag indicating whether the target should be
                saved in the layer object for later use.
            norm_ratio (float): will only be used in children of DTPLayer for
                the minimal_norm update
        """

        self.update_idx = i

        h_target = self.compute_output_target(loss, target_lr)

        h_target = self.propagate_backward(h_target, i)

        if save_target:
            self.layers[i].target = h_target

        if i == 0: # first hidden layer needs to have the input
                   # for computing gradients
            self.layers[i].compute_forward_gradients(h_target, self.input,
                                                     norm_ratio=norm_ratio)
        else:
            self.layers[i].compute_forward_gradients(h_target,
                                                 self.layers[i-1].activations,
                                                     norm_ratio=norm_ratio)

    def backward_all(self, output_target, save_target=False, norm_ratio=1.):
        """ Propagate the output_target backwards through all the layers. Based
        on these local targets, compute the gradient of the forward weights and
        biases of all layers.
        Args:
            output_target (torch.Tensor): a mini-batch of targets for the
                output layer.
            save_target (bool): flag indicating whether the target should be
                saved in the layer object for later use.
            norm_ratio (float): will only be used in children of DTPLayer for
                the minimal_norm update
            """

        h_target = output_target

        if save_target:
            self.layers[-1].target = h_target
        for i in range(self.depth-1, 0, -1):
            h_current = self.layers[i].activations
            h_previous = self.layers[i-1].activations
            self.layers[i].compute_forward_gradients(h_target, h_previous,
                                                     norm_ratio=norm_ratio)
            h_target = self.layers[i].backward(h_target, h_previous, h_current)
            if save_target:
                self.layers[i-1].target = h_target

        self.layers[0].compute_forward_gradients(h_target, self.input,
                                                 norm_ratio=norm_ratio)

    def backward(self, loss, target_lr, save_target=False, norm_ratio=1.):
        """ Compute and propagate the output_target backwards through all the
        layers. Based on these local targets, compute the gradient of the
        forward weights and biases of all layers.
        Args:
            loss (nn.Module): output loss of the network
            target_lr: the learning rate for computing the output target based
                on the gradient of the loss w.r.t. the output layer
            save_target (bool): flag indicating whether the target should be
                saved in the layer object for later use.
            norm_ratio (float): will only be used in children of DTPLayer for
                the minimal_norm update
            """

        output_target = self.compute_output_target(loss, target_lr)
        self.backward_all(output_target, save_target, norm_ratio=norm_ratio)

    def compute_feedback_gradients(self):
        """ Compute the local reconstruction loss for each layer and compute
        the gradient of those losses with respect to
        the feedback weights and biases. The gradients are saved in the
        feedback parameter tensors."""

        for i in range(1, self.depth):
            h_corrupted = self.layers[i-1].activations + \
                    self.sigma * torch.randn_like(self.layers[i-1].activations)
            self.layers[i].compute_feedback_gradients(h_corrupted, self.sigma)

    def get_forward_parameter_list(self):
        """
        Args:
            freeze_ouptut_layer (bool): flag indicating whether the forward
                parameters of the output layer should be excluded from the
                returned list.
        Returns: a list with all the forward parameters (weights and biases) of
            the network.
        """
        parameterlist = []
        for layer in self.layers:
            parameterlist.append(layer.weights)
            if layer.bias is not None:
                parameterlist.append(layer.bias)
        return parameterlist

    def get_reduced_forward_parameter_list(self):
        """
        Returns: a list with all the forward parameters of the network, except
        from the ones of the output layer.
        """
        parameterlist = []
        for layer in self.layers[:-1]:
            parameterlist.append(layer.weights)
            if layer.bias is not None:
                parameterlist.append(layer.bias)
        return parameterlist

    def get_forward_parameters_last_two_layers(self):
        parameterlist = []
        for layer in self.layers[-2:]:
            parameterlist.append(layer.weights)
            if layer.bias is not None:
                parameterlist.append(layer.bias)
        return parameterlist

    def get_forward_parameters_last_three_layers(self):
        parameterlist = []
        for layer in self.layers[-3:]:
            parameterlist.append(layer.weights)
            if layer.bias is not None:
                parameterlist.append(layer.bias)
        return parameterlist

    def get_forward_parameters_last_four_layers(self):
        parameterlist = []
        for layer in self.layers[-4:]:
            parameterlist.append(layer.weights)
            if layer.bias is not None:
                parameterlist.append(layer.bias)
        return parameterlist

    def get_forward_parameter_list_first_layer(self):
        """
        Returns: a list with only the forward parameters of the first layer.
        """
        parameterlist = []
        parameterlist.append(self.layers[0].weights)
        if self.layers[0].bias is not None:
            parameterlist.append(self.layers[0].bias)
        return parameterlist

    def get_feedback_parameter_list(self):
        """
        Returns (list): a list with all the feedback parameters (weights and
            biases) of the network. Note that the first hidden layer does not
            need feedback parameters, so they are not put in the list.
        """
        parameterlist = []
        for layer in self.layers[1:]:
            parameterlist.append(layer.feedbackweights)
            if layer.feedbackbias is not None:
                parameterlist.append(layer.feedbackbias)
        return parameterlist

    def get_BP_updates(self, loss, i):
        """
        Compute the gradients of the loss with respect to the forward
        parameters of layer i.
        Args:
            loss (torch.Tensor): the loss value of the current minibatch.
            i (int): layer index
        Returns (tuple): a tuple with the gradients of the loss with respect to
            the forward parameters of layer i, computed with backprop.
        """
        return self.layers[i].compute_bp_update(loss)

    def compute_bp_angles(self, loss, i, retain_graph=False):
        """
        Compute the angles of the current forward parameter updates of layer i
        with the backprop update for those parameters.
        Args:
            loss (torch.Tensor): the loss value of the current minibatch.
            i (int): layer index
            retain_graph (bool): flag indicating whether the graph of the
                network should be retained after computing the gradients or
                jacobians. If the graph will not be used anymore for the current
                minibatch afterwards, retain_graph should be False.
        Returns (tuple): Tuple containing the angle in degrees between the
            updates for the forward weights at index 0 and the forward bias
            at index 1 (if bias is not None).
        """
        bp_gradients = self.layers[i].compute_bp_update(loss,
                                                        retain_graph)
        gradients = self.layers[i].get_forward_gradients()
        if contains_nan(bp_gradients[0].detach()):
            print('bp update contains nan (layer {}):'.format(i))
            print(bp_gradients[0].detach())
        if contains_nan(gradients[0].detach()):
            print('weight update contains nan (layer {}):'.format(i))
            print(gradients[0].detach())
        if torch.norm(gradients[0].detach(), p='fro') < 1e-14:
            print('norm updates approximately zero (layer {}):'.format(i))
            print(torch.norm(gradients[0].detach(), p='fro'))
            print(gradients[0].detach())
        if torch.norm(gradients[0].detach(), p='fro') == 0:
            print('norm updates exactly zero (layer {}):'.format(i))
            print(torch.norm(gradients[0].detach(), p='fro'))
            print(gradients[0].detach())

        weights_angle = compute_angle(bp_gradients[0].detach(),
                                            gradients[0])
        if self.layers[i].bias is not None:
            bias_angle = compute_angle(bp_gradients[1].detach(),
                                             gradients[1])
            return (weights_angle, bias_angle)
        else:
            return (weights_angle, )

    def compute_gn_angles(self, output_activation, loss, damping, i,
                          retain_graph=False):
        """
        Compute the angles of the current forward parameter updates of layer i
        with the GN update for those parameters.
        Args:
            see lib.dtp_layers.DTPLayer.compute_gn_updates(...)
            i (int): layer index
            retain_graph (bool): flag indicating whether the graph of the
                network should be retained after computing the gradients or
                jacobians. If the graph will not be used anymore for the current
                minibatch afterwards, retain_graph should be False.
        Returns (tuple): Tuple containing the angle in degrees between the
            updates for the forward weights at index 0 and the forward bias
            at index 1 (if bias is not None).
        """
        gn_gradients = self.layers[i].compute_gn_update(output_activation,
                                                        loss,
                                                        damping,
                                                        retain_graph)
        gradients =self.layers[i].get_forward_gradients()
        weights_angle = compute_angle(gn_gradients[0],
                                            gradients[0])
        if self.layers[i].bias is not None:
            bias_angle = compute_angle(gn_gradients[1],
                                             gradients[1])
            return (weights_angle, bias_angle)
        else:
            return (weights_angle,)

    def compute_gn_activation_angle(self, output_activation, loss,
                                    damping, i, step,
                                    retain_graph=False,
                                    linear=False):
        """
        Compute the angle between the difference between the target and layer
        activation and the gauss-newton update for the layers activation
        Args:
            see lib.dtp_layers.DTPLayer.compute_gn_activation_updates(...)
            i (int): layer index
            step (int): epoch step, just for logging
            retain_graph (bool): flag indicating whether the graph of the
                network should be retained after computing the gradients or
                jacobians. If the graph will not be used anymore for the current
                minibatch afterwards, retain_graph should be False.
            linear (bool): Flag indicating whether the GN update for the
                linear activations should be computed instead of for the
                nonlinear activations.
        Returns: The average angle in degrees
        """
        if linear:
            target_difference = self.layers[i].linearactivations - \
                                self.layers[i].target
        else:
            target_difference = self.layers[i].activations - \
                                self.layers[i].target
        gn_updates = self.layers[i].compute_gn_activation_updates(
            output_activation,
            loss,
            damping,
            retain_graph=retain_graph,
            linear=linear
        )
        # print(f"Layer {i}:")
        # print(torch.mean(target_difference).item())
        # print(torch.mean(gn_updates).item())
        if self._plots is not None:
            self.td_activation.at[step, i] = torch.mean(target_difference).item()
            self.gn_activation.at[step, i] = torch.mean(gn_updates).item()

        # exit()
        gn_activationav = compute_average_batch_angle(target_difference, gn_updates)
        return gn_activationav

    def compute_bp_activation_angle(self, loss, i, retain_graph=False,
                                    linear=False):
        """
        Compute the angle between the difference between the target and layer
        activation and the backpropagation update for the layers activation
        Args:
            loss (torch.Tensor): the loss value of the current minibatch.
            i (int): layer index
            retain_graph (bool): flag indicating whether the graph of the
                network should be retained after computing the gradients or
                jacobians. If the graph will not be used anymore for the current
                minibatch afterwards, retain_graph should be False.
            linear (bool): Flag indicating whether the GN update for the
                linear activations should be computed instead of for the
                nonlinear activations.
        Returns : The average angle in degrees
        """
        if linear:
            target_difference = self.layers[i].linearactivations - \
                                self.layers[i].target
        else:
            target_difference = self.layers[i].activations - \
                                self.layers[i].target
        bp_updates = self.layers[i].compute_bp_activation_updates(
            loss=loss,
            retain_graph=retain_graph,
            linear=linear
        ).detach()

        angle = compute_average_batch_angle(target_difference.detach(),
                                                  bp_updates)

        return angle

    def compute_gnt_angle(self, output_activation, loss, damping,
                          i, step, retain_graph=False, linear=False):
        if i == 0:
            h_previous = self.input
        else:
            h_previous = self.layers[i-1].activations

        gnt_updates = self.layers[i].compute_gnt_updates(
            output_activation=output_activation,
            loss=loss,
            h_previous=h_previous,
            damping=damping,
            retain_graph=retain_graph,
            linear=linear
        )

        gradients = self.layers[i].get_forward_gradients()
        weights_angle = compute_angle(gnt_updates[0], gradients[0])
        if self.layers[i].bias is not None:
            bias_angle = compute_angle(gnt_updates[1], gradients[1])
            return (weights_angle, bias_angle)
        else:
            return (weights_angle, )


    def save_logs(self, writer, step):
        """ Save logs and plots for tensorboardX.
        Args:
            writer (SummaryWriter): summary writer from tensorboardX
            step (int): the global step used for the x-axis of the plots
            """

        for i in range(len(self.layers)):
            name = 'layer {}'.format(i+1)
            self.layers[i].save_logs(writer, step, name,
                                     no_gradient=i==0)

    def save_feedback_batch_logs(self, writer, step, init=False):
        """
        Save the logs for the current minibatch on tensorboardX.
        Args:
            writer (SummaryWriter): summary writer from tensorboardX
            step (int): the global step used for the x-axis of the plots
            init (bool): flag indicating that the training is in the
                initialization phase (only training the feedback weights).
        """
        for i in range(len(self.layers)):
            name = 'layer {}'.format(i+1)
            self.layers[i].save_feedback_batch_logs(writer, step, name,
                                     no_gradient=i == 0, init=init)

    def save_bp_angles(self, writer, step, loss, retain_graph=False):
        """
        Save the angles of the current forward parameter updates
        with the backprop update for those parameters on tensorboardX.
        Args:
            writer (SummaryWriter): summary writer from tensorboardX
            step (int): the global step used for the x-axis of the plots
            loss (torch.Tensor): the loss value of the current minibatch.
            retain_graph (bool): flag indicating whether the graph of the
                network should be retained after computing the gradients or
                jacobians. If the graph will not be used anymore for the current
                minibatch afterwards, retain_graph should be False.
        """
        # if self.update_idx is None, the randomized setting is not used and
        # all the layers have their parameters updated. The angle should thus
        # be computed for all layers
        if self.update_idx is None:
            layer_indices = range(len(self.layers))
        else:
            layer_indices = [self.update_idx]

        for i in layer_indices:
            name = 'layer {}'.format(i+1)
            if i != layer_indices[-1]:  # if it is not the last index, the graph
                # should be saved for the next index
                retain_graph_flag = True
            else:
                retain_graph_flag = retain_graph
            angles = self.compute_bp_angles(loss, i, retain_graph_flag)
            writer.add_scalar(
                tag='{}/weight_bp_angle'.format(name),
                scalar_value=angles[0],
                global_step=step
            )

            if self._plots is not None:
                self.bp_angles.at[step, i] = angles[0].item()


            if self.layers[i].bias is not None:
                writer.add_scalar(
                    tag='{}/bias_bp_angle'.format(name),
                    scalar_value=angles[1],
                    global_step=step
                )

    def save_gn_angles(self, writer, step, output_activation, loss, damping,
                       retain_graph=False):
        """
        Save the angles of the current forward parameter updates of layer i
        with the backprop update for those parameters. on tensorboardX.
        Args:
            writer (SummaryWriter): summary writer from tensorboardX
            step (int): the global step used for the x-axis of the plots
            see lib.dtp_layers.DTPLayer.compute_gn_updates(...)
            retain_graph (bool): flag indicating whether the graph of the
                network should be retained after computing the gradients or
                jacobians. If the graph will not be used anymore for the current
                minibatch afterwards, retain_graph should be False.
        """
        # if self.update_idx is None, the randomized setting is not used and
        # all the layers have their parameters updated. The angle should thus
        # be computed for all layers
        if self.update_idx is None:
            layer_indices = range(len(self.layers))
        else:
            layer_indices = [self.update_idx]

        for i in layer_indices:
            name = 'layer {}'.format(i+1)
            if i != layer_indices[-1]:  # if it is not the last index, the graph
                                        # should be saved for the next index
                retain_graph_flag = True
            else:
                retain_graph_flag = retain_graph
            angles = self.compute_gn_angles(output_activation, loss, damping,
                                            i, retain_graph_flag)
            writer.add_scalar(
                tag='{}/weight_gn_angle'.format(name),
                scalar_value=angles[0],
                global_step=step
            )

            if self._plots is not None:
                self.gn_angles.at[step, i] = angles[0].item()

            if self.layers[i].bias is not None:
                writer.add_scalar(
                    tag='{}/bias_gn_angle'.format(name),
                    scalar_value=angles[1],
                    global_step=step
                )

    def save_gnt_angles(self, writer, step, output_activation, loss,
                        damping, retain_graph=False, custom_result_df=None):
        # if self.update_idx is None, the randomized setting is not used and
        # all the layers have their parameters updated. The angle should thus
        # be computed for all layers
        # print('saving gnt angles')
        if self.update_idx is None:
            layer_indices = range(len(self.layers)-1)
        else:
            layer_indices = [self.update_idx]

        # assign a damping constant for each layer for computing the gnt angles
        if isinstance(damping, float):
            damping = [damping for i in range(self.depth)]
        else:
            # print(damping)
            # print(len(damping))
            # print(layer_indices)
            # print(len(layer_indices))
            assert len(damping) == len(layer_indices)

        for i in layer_indices:
            name = 'layer {}'.format(i + 1)
            if i != layer_indices[-1]:  # if it is not the last index, the graph
                # should be saved for the next index
                retain_graph_flag = True
            else:
                retain_graph_flag = retain_graph
            angles = self.compute_gnt_angle(output_activation=output_activation,
                                            loss=loss,
                                            damping=damping[i],
                                            i=i,
                                            step=step,
                                            retain_graph=retain_graph_flag)
            if custom_result_df is not None:
                custom_result_df.at[step,i] = angles[0].item()
            else:
                writer.add_scalar(
                    tag='{}/weight_gnt_angle'.format(name),
                    scalar_value=angles[0],
                    global_step=step
                )

                if self._plots is not None:
                    # print('saving gnt angles')
                    # print(angles[0].item())
                    self.gnt_angles.at[step, i] = angles[0].item()

                if self.layers[i].bias is not None:
                    writer.add_scalar(
                        tag='{}/bias_gnt_angle'.format(name),
                        scalar_value=angles[1],
                        global_step=step
                    )

    def save_nullspace_norm_ratio(self, writer, step, output_activation,
                                  retain_graph=False):
        if self.update_idx is None:
            layer_indices = range(len(self.layers))
        else:
            layer_indices = [self.update_idx]

        for i in layer_indices:
            name = 'layer {}'.format(i + 1)
            if i != layer_indices[-1]:  # if it is not the last index, the graph
                                        # should be saved for the next index
                retain_graph_flag = True
            else:
                retain_graph_flag = retain_graph

            relative_norm = self.layers[i].compute_nullspace_relative_norm(
                output_activation,
                retain_graph=retain_graph_flag
            )

            writer.add_scalar(
                tag='{}/nullspace_relative_norm'.format(name),
                scalar_value=relative_norm,
                global_step=step
            )

            if self._plots is not None:
                self.nullspace_relative_norm.at[step, i] = relative_norm.item()


    def save_bp_activation_angle(self, writer, step, loss,
                                 retain_graph=False):
        """
        Save the angle between the difference between the target and layer
        activation and the backpropagation update for the layers activation
        Args:
            writer (SummaryWriter): summary writer from tensorboardX
            step (int): the global step used for the x-axis of the plots
            see lib.dtp_layers.DTPLayer.compute_bp_activation_updates(...)
            retain_graph (bool): flag indicating whether the graph of the
                network should be retained after computing the gradients or
                jacobians. If the graph will not be used anymore for the current
                minibatch afterwards, retain_graph should be False.
        """
        # if self.update_idx is None, the randomized setting is not used and
        # all the layers have their parameters updated. The angle should thus
        # be computed for all layers
        if self.update_idx is None:
            layer_indices = range(len(self.layers))
        else:
            layer_indices = [self.update_idx]

        for i in layer_indices:
            name = 'layer {}'.format(i + 1)
            if i != layer_indices[-1]:  # if it is not the last index, the graph
                # should be saved for the next index
                retain_graph_flag = True
            else:
                retain_graph_flag = retain_graph
            angle = self.compute_bp_activation_angle(loss, i,
                                                      retain_graph_flag)


            writer.add_scalar(
                tag='{}/activation_bp_angle'.format(name),
                scalar_value=angle,
                global_step=step
            )
            if self._plots is not None:
                self.bp_activation_angles.at[step, i] = angle.item()
        return

    def save_gn_activation_angle(self, writer, step, output_activation, loss,
                                 damping, retain_graph=False):
        """
        Save the angle between the difference between the target and layer
        activation and the gauss-newton update for the layers activation
        Args:
            writer (SummaryWriter): summary writer from tensorboardX
            step (int): the global step used for the x-axis of the plots
            see lib.dtp_layers.DTPLayer.compute_bp_activation_updates(...)
            retain_graph (bool): flag indicating whether the graph of the
                network should be retained after computing the gradients or
                jacobians. If the graph will not be used anymore for the current
                minibatch afterwards, retain_graph should be False.
        """
        # if self.update_idx is None, the randomized setting is not used and
        # all the layers have their parameters updated. The angle should thus
        # be computed for all layers
        if self.update_idx is None:
            layer_indices = range(len(self.layers))
        else:
            layer_indices = [self.update_idx]

        for i in layer_indices:
            name = 'layer {}'.format(i + 1)
            if i != layer_indices[-1]:  # if it is not the last index, the graph
                # should be saved for the next index
                retain_graph_flag = True
            else:
                retain_graph_flag = retain_graph
            angle = self.compute_gn_activation_angle(output_activation, loss,
                                                     damping, i, step,
                                                     retain_graph_flag)
            writer.add_scalar(
                tag='{}/activation_gn_angle'.format(name),
                scalar_value=angle,
                global_step=step
            )

            if self._plots is not None:
                self.gn_activation_angles.at[step, i] = angle.item()
        return


    def get_av_reconstruction_loss(self):
        """ Get the average reconstruction loss of the network across its layers
        for the current mini-batch.
        Args:
            network (networks.DTPNetwork): network
        Returns (torch.Tensor):
            Tensor containing a scalar of the average reconstruction loss
        """
        reconstruction_losses = np.array([])

        for layer in self.layers[1:]:
            reconstruction_losses = np.append(reconstruction_losses,
                                              layer.reconstruction_loss)

        return np.mean(reconstruction_losses)


## BUILD NETWORK
def build_network(args):
    """
    Create the network based on the provided command line arguments
    Args:
        args: command line arguments
    Returns: a network
    """
    if args.hidden_layers is None:
        if isinstance(args.size_hidden, list):
            n_hidden = args.size_hidden
        elif isinstance(args.size_hidden, int):
            n_hidden = [args.size_hidden] * args.num_hidden
    else:
        n_hidden = args.hidden_layers

    forward_requires_grad = args.save_BP_angle or args.save_GN_angle or\
                            args.save_GN_activations_angle or \
                            args.save_BP_activations_angle or \
                            args.save_GNT_angle or \
                            args.network_type in ['GN', 'GN2'] or \
                            args.output_space_plot_bp or \
                            args.gn_damping_hpsearch or \
                            args.save_nullspace_norm_ratio
    if args.classification:
        assert (args.output_activation == 'softmax' or \
               args.output_activation == 'sigmoid'), "Output layer should " \
                    "represent probabilities => use softmax or sigmoid!"
        if args.output_activation == 'sigmoid':
            if args.network_type == 'LeeDTP':
                raise ValueError('For the LeeDTP network, only softmax output'
                                 'activtion is supported')
            output_activation = 'sigmoid'
        elif args.output_activation == 'softmax':
            output_activation = 'linear'  # the softmax function is incorporated
                                        # in the loss function in Pytorch
        else:
            assert False
    else:
        output_activation = args.output_activation

    kwargs = {'n_in': args.size_input,
              'n_hidden': n_hidden,
              'n_out': args.size_output,
              'activation': args.hidden_activation,
              'bias': not args.no_bias,
              'sigma': args.sigma,
              'forward_requires_grad': forward_requires_grad,
              'initialization': args.initialization,
              'output_activation': output_activation,
              'fb_activation': args.fb_activation,
              'plots': args.plots,
              }

    net = DTPNetwork(**kwargs)

    return net


## SET ARGUMENTS

## First chunk of arguments set to match other networks
dtp_args = Namespace(network_type='DTP',
        batch_size= 4,
        dataset= 'cifar10',
        epochs=25,
        hidden_activation='tanh',
        lr=0.01,
        momentum=0.0,
        no_val_set=True,
        num_hidden=3,
        num_test=10000,
        num_train=50000,
        size_hidden=500,
        size_input=3072,
        size_output=10,
        optimizer='SGD',
        output_activation='softmax',
        classification=True,

        # The rest are kept as their default values
        beta1= 0.99, 
        beta1_fb= 0.99,
        beta2=0.99,
        beta2_fb=0.99,
        create_plots=False,
        cuda_deterministic=False,
        diff_rec_loss=False,
        direct_fb=False,
        double_precision=False,
        epochs_fb=1,
        epsilon=1e-4,
        epsilon_fb=1e-4,
        evaluate=False,
        extra_fb_epochs=0,
        extra_fb_minibatches=0,
        fb_activation='tanh',
        feedback_wd=0.0,
        forward_wd=0.0,
        freeze_BPlayers=False,
        freeze_fb_weights=False,
        freeze_forward_weights=False,
        freeze_output_layer=False,
        gn_damping=0.,
        gn_damping_hpsearch=False,
        gn_damping_training=0.0,
        hidden_fb_activation='tanh',
        hidden_layers=None,
        hpsearch=False,
        initialization='xavier',
        load_weights=False,
        log_interval=None,
        loss_scale=1.0,
        lr_fb=0.000101149118237,
        multiple_hpsearch=False,
        no_bias=False,
        no_cuda=False,
        no_preprocessing_mnist=False,
        norm_ratio=1.0,
        normalize_lr=False,
        not_randomized=False,
        not_randomized_fb=False,
        num_val=1000,
        only_train_first_layer=False,
        only_train_last_four_layers=False,
        only_train_last_three_layers=False,
        only_train_last_two_layers=False,
        optimizer_fb='SGD',
        out_dir='logs',
        output_space_plot=False,
        output_space_plot_bp=False,
        output_space_plot_layer_idx=None,
        parallel=False,
        plots=None,
        random_seed=42,
        recurrent_input=False,
        save_BP_activations_angle=False,
        save_BP_angle=False,
        save_GNT_angle=False,
        save_GN_activations_angle=False,
        save_GN_angle=False,
        save_angle=False,
        save_logs=False,
        save_loss_plot=False,
        save_nullspace_norm_ratio=False,
        save_weights=False,
        shallow_training=False,
        sigma=0.08,
        size_hidden_fb=500,
        size_mlp_fb=100,
        soft_target=0.9,
        target_stepsize=0.01,
        train_only_feedback_parameters=False,
        train_randomized=False,
        train_randomized_fb=False, 
        train_separate=False)

dtp_model = build_network(dtp_args)