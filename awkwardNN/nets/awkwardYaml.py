import torch
import torch.nn as nn
import torch.nn.functional as F

from awkwardNN.nets.deepset import DeepSetNetwork, AwkwardDeepSetDoubleJagged, AwkwardDeepSetSingleJagged
from awkwardNN.nets.awkwardRNN import AwkwardRNNDoubleJagged
from awkwardNN.nets.mlp import MLP
from awkwardNN.utils.yaml_utils import get_nested_yaml, parse_hidden_size_string


ACTIVATIONS = {'tanh': torch.tanh, 'relu': torch.relu}

####################################################################
#                   UTILITY FUNCTIONS FOR init                     #
####################################################################

def _get_deepset_sizes(yaml_dict):
    if 'phi_sizes' in yaml_dict:
        phi_sizes = parse_hidden_size_string(yaml_dict['phi_sizes'])
    else:
        phi_sizes = [32, 32]
    if 'rho_sizes' in yaml_dict:
        rho_sizes = parse_hidden_size_string(yaml_dict['rho_sizes'])
    else:
        rho_sizes = [32, 32]
    return phi_sizes, rho_sizes


def _get_hidden_size(yaml_dict):
    if yaml_dict['mode'] == 'mlp':
        if 'hidden_sizes' in yaml_dict:
            hidden_size = parse_hidden_size_string(yaml_dict['hidden_sizes'])
        else:
            hidden_size = [32, 32]
    else:
        hidden_size = yaml_dict['embed_dim'] if 'embed_dim' in yaml_dict else 32
    return hidden_size


def _get_basic_parameters(yaml_dict):
    dropout = yaml_dict['dropout'] if 'dropout' in yaml_dict else 0
    hidden_size = _get_hidden_size(yaml_dict)
    mode = yaml_dict['mode']
    nonlinearity = yaml_dict['nonlinearity'] if 'nonlinearity' in yaml_dict else 'tanh'
    return dropout, hidden_size, mode, nonlinearity


def _get_kwargs(yaml_dict):
    dropout, hidden_size, mode, nonlinearity = _get_basic_parameters(yaml_dict)
    kwargs = {'dropout': dropout}
    if mode == 'deepset':
        phi_sizes, rho_sizes = _get_deepset_sizes(yaml_dict)
        kwargs.update({'output_size': hidden_size, 'nonlinearity': nonlinearity,
                       'phi_sizes': phi_sizes, 'rho_sizes': rho_sizes})
    elif mode == 'mlp':
        kwargs.update({'hidden_size': hidden_size, 'output_size': yaml_dict['embed_dim'],
                       'nonlinearity': nonlinearity})
    elif mode == 'vanilla_rnn':
        kwargs.update({'hidden_size': hidden_size, 'nonlinearity': nonlinearity})
    elif mode in ['lstm', 'gru']:
        kwargs.update({'hidden_size': hidden_size})
    return kwargs


def _get_nested_network_input_size(yaml_dict, dataset):
    input_size = 0
    if yaml_dict['mode'] == 'mlp':
        for subfield_i, dataset_i in zip(yaml_dict['fields'], dataset.nested_datasets):
            subfield_size = list(subfield_i.values())[0]['embed_dim']
            if dataset_i.use_dataset:
                input_size += subfield_size
    else:
        input_size_list = []
        for subfield_i, dataset_i in zip(yaml_dict['fields'], dataset.nested_datasets):
            subfield_size = list(subfield_i.values())[0]['embed_dim']
            if dataset_i.use_dataset:
                input_size_list.append(subfield_size)
        if len(set(input_size_list)) > 1:
            raise ValueError("Input sizes (embed_dim) of each field to nested networks must all be the "
                             "size. Received sizes: {}".format(input_size_list))
        input_size = input_size_list[0]
    return input_size


def _get_output_network_input_size(yaml_dict, dataset):
    input_size = 0
    if dataset.use_fixed_data:
        input_size += yaml_dict['fixed_fields']['embed_dim']
    if dataset.use_jagged_data:
        input_size += yaml_dict['jagged_fields']['embed_dim']
    if dataset.use_object_data:
        input_size += yaml_dict['object_fields']['embed_dim']
    if dataset.use_nested_data:
        input_size += yaml_dict['nested_fields']['embed_dim']
    return input_size


####################################################################
#                  UTILITY FUNCTIONS FOR forward()                 #
####################################################################

def _append_lists(X_list):
    X_prime = []
    for X_i in X_list:
        X_prime = _append(X_prime, X_i, axis=2)
    return X_prime


def _append(X1, X2, axis):
    if X1 == [] and X2 == []:
        return []
    elif X1 == []:
        return X2
    elif X2 == []:
        return X1
    return torch.cat((X1, X2), dim=axis)


def _parse_output(mode, output, output_size):
    if mode == 'mlp':
        if output == []:
            return torch.zeros(1, 1, output_size)
        return output
    elif mode == 'deepset':
        return output
    elif mode == 'lstm':
        _, (hidden, _) = output
        return hidden
    else:
        _, hidden = output
        return hidden

####################################################################


class AwkwardFixedNet(nn.Module):
    def __init__(self, yaml_dict, dataset):
        super(AwkwardFixedNet, self).__init__()
        self._use_fixed_net = dataset.use_fixed_data
        if self._use_fixed_net:
            yaml_fixed_dict = yaml_dict['fixed_fields']
            kwargs = _get_kwargs(yaml_fixed_dict)
            kwargs.update({'input_size': dataset.fixed_input_size})
            if yaml_fixed_dict['mode'] == 'deepset':
                self.fixed_net = DeepSetNetwork(**kwargs)
            elif yaml_fixed_dict['mode'] == 'mlp':
                self.fixed_net = MLP(**kwargs)
            else:
                mode_list = ['deepset', 'mlp']
                raise ValueError("The fixed mode '%s' is not supported. Supported modes"
                                 " are %s." % (yaml_fixed_dict['mode'], mode_list))

    def forward(self, X):
        if self._use_fixed_net and X != []:
            return self.fixed_net(X)
        return X


class AwkwardJaggedNet(nn.Module):
    def __init__(self, yaml_dict, dataset):
        super(AwkwardJaggedNet, self).__init__()
        self._use_jagged_net = dataset.use_jagged_data
        if self._use_jagged_net:
            yaml_jagged_dict = yaml_dict['jagged_fields']
            kwargs = _get_kwargs(yaml_jagged_dict)
            kwargs.update({'input_size': dataset.jagged_input_size})
            self.jagged_mode = yaml_jagged_dict['mode']
            if self.jagged_mode == 'deepset':
                self.jagged_net = AwkwardDeepSetSingleJagged(**kwargs)
            elif self.jagged_mode == 'vanilla_rnn':
                self.jagged_net = nn.RNN(**kwargs)
            elif self.jagged_mode == 'gru':
                self.jagged_net = nn.GRU(**kwargs)
            elif self.jagged_mode == 'lstm':
                self.jagged_net = nn.LSTM(**kwargs)
            else:
                mode_list = ['vanilla_rnn', 'lstm', 'gru', 'deepset']
                raise ValueError("The jagged mode '%s' is not supported. Supported modes"
                                 " are %s." % (yaml_jagged_dict['mode'], mode_list))

    def forward(self, X):
        if self._use_jagged_net and X != []:
            X = X.squeeze(0).unsqueeze(1)
            jagged_output = self.jagged_net(X)
            return _parse_output(self.jagged_mode, jagged_output, -1)
        return X


class AwkwardObjectNet(nn.Module):
    def __init__(self, yaml_dict, dataset):
        super(AwkwardObjectNet, self).__init__()
        self._use_object_net = dataset.use_object_data
        if self._use_object_net:
            yaml_object_dict = yaml_dict['object_fields']
            kwargs = _get_kwargs(yaml_object_dict)
            mode = yaml_object_dict['mode']
            if mode == 'deepset':
                self.object_net = AwkwardDeepSetDoubleJagged(**kwargs)
            elif mode in ['vanilla_rnn', 'lstm', 'gru']:
                nonlinearity = yaml_dict['nonlinearity'] if 'nonlinearity' in yaml_dict else 'tanh'
                kwargs.update({'mode': mode, 'nonlinearity': nonlinearity})
                self.object_net = AwkwardRNNDoubleJagged(**kwargs)
            else:
                mode_list = ['vanilla_rnn', 'lstm', 'gru', 'deepset']
                raise ValueError("The object mode '%s' is not supported. Supported modes"
                                 " are %s." % (mode, mode_list))

    def forward(self, X):
        if self._use_object_net and X != []:
            return self.object_net(X)
        return X


class AwkwardNestedNet(nn.Module):
    def __init__(self, yaml_dict, dataset):
        super(AwkwardNestedNet, self).__init__()
        self._use_nested_net = dataset.use_nested_data
        self.nested_subnetworks = []
        if self._use_nested_net:
            # initialize subnetworks
            nested_yaml_dict = yaml_dict['nested_fields']
            self._init_subnetworks(nested_yaml_dict, dataset)

            # initialize network that takes subnetworks as input
            kwargs = _get_kwargs(nested_yaml_dict)
            self.nested_mode = nested_yaml_dict['mode']
            kwargs.update({'input_size': _get_nested_network_input_size(nested_yaml_dict, dataset)})
            if self.nested_mode == 'deepset':
                self.nested_net = AwkwardDeepSetSingleJagged(**kwargs)
            elif self.nested_mode == 'vanilla_rnn':
                self.nested_net = nn.RNN(**kwargs)
            elif self.nested_mode == 'gru':
                self.nested_net = nn.GRU(**kwargs)
            elif self.nested_mode == 'lstm':
                self.nested_net = nn.LSTM(**kwargs)
            elif self.nested_mode == 'mlp':
                self.nested_net = MLP(**kwargs)
            else:
                mode_list = ['vanilla_rnn', 'lstm', 'gru', 'deepset', 'mlp']
                raise ValueError("The nested mode '%s' is not supported. Supported modes"
                                 " are %s." % (nested_yaml_dict['mode'], mode_list))
        return

    def _init_subnetworks(self, nested_yaml_dict, dataset):
        nested_fields, nested_names = get_nested_yaml(nested_yaml_dict)
        zip_iter = zip(nested_fields, dataset.nested_datasets, nested_names)
        for field_i, dataset_i, name_i in zip_iter:
            if dataset_i.use_dataset:
                self.nested_subnetworks.append(AwkwardYaml(field_i, dataset_i, name_i))

    def forward(self, X):
        if not self._use_nested_net or X == []:
            return []
        subnested_output = self._forward_subnest_networks(X)
        return self.nested_net(subnested_output)

    def _forward_subnest_networks(self, X):
        subnested_output = []
        for net, x_nest in zip(self.nested_subnetworks, X):
            sub_output = net(x_nest)
            output = _parse_output(self.nested_mode, sub_output, net.output_size)
            if self.nested_mode == 'mlp':
                subnested_output = _append(subnested_output, output, axis=2)
            elif output != []:
                subnested_output = _append(subnested_output, output, axis=0)
        return subnested_output


class AwkwardYaml(nn.Module):
    def __init__(self, yaml_dict, dataset, name, topnetwork=False):
        """"""
        super(AwkwardYaml, self).__init__()
        self.name = name
        self._use_output_net = dataset.use_dataset
        if self._use_output_net is False:
            return

        self.yaml_dict = yaml_dict
        self.fixed_net = AwkwardFixedNet(yaml_dict, dataset)
        self.jagged_net = AwkwardJaggedNet(yaml_dict, dataset)
        self.object_net = AwkwardObjectNet(yaml_dict, dataset)
        self.nested_net = AwkwardNestedNet(yaml_dict, dataset)
        self._init_output_network(dataset)
        self.topnetwork = topnetwork
        if topnetwork:
            self.fc = nn.Linear(yaml_dict['embed_dim'], dataset.target_size)

    def _init_output_network(self, dataset):
        kwargs = _get_kwargs(self.yaml_dict)
        self.output_size = kwargs['output_size']
        kwargs.update({'input_size': _get_output_network_input_size(self.yaml_dict, dataset)})
        self.output_mode = self.yaml_dict['mode']
        if self.output_mode == 'deepset':
            self.output_net = AwkwardDeepSetSingleJagged(**kwargs)
        elif self.output_mode == 'mlp':
            self.output_net = MLP(**kwargs)
        else:
            mode_list = ['mlp', 'deepset']
            raise ValueError("The mode '%s' is not supported. Supported modes"
                             " are %s." % (self.output_mode, mode_list))

    def forward(self, X):
        """"""
        X_fixed, X_jagged, X_object, X_nested = X

        # go through data
        hidden_nested = self.nested_net(X_nested)
        hidden_fixed = self.fixed_net(X_fixed)
        hidden_jagged = self.jagged_net(X_jagged)
        hidden_object = self.object_net(X_object)

        # append it all and go through the output network
        hidden = [hidden_fixed, hidden_jagged, hidden_object, hidden_nested]
        latent_state = _append_lists(hidden)
        return self._forward_output_network(latent_state)

    def _forward_output_network(self, X):
        if X == []:
            return []
        output = self.output_net(X)
        return self._forward_output_nonlinearity(output)

    def _forward_output_nonlinearity(self, output):
        if self.topnetwork:
            x = self.fc(output)[0] #TODO: look at this again later
            y = F.log_softmax(x, dim=1)
            return y
        return output
