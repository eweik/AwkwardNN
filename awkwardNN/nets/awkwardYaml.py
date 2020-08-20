import torch
import torch.nn as nn
import torch.nn.functional as F

from awkwardNN.nets.deepset import AwkwardDeepSetDoubleJagged, AwkwardDeepSetSingleJagged
from awkwardNN.nets.awkwardRNN import AwkwardRNNDoubleJagged, AwkwardRNNSingleJagged
from awkwardNN.nets.awkwardMLP import AwkwardMLP
from awkwardNN.utils.yaml_utils import get_nested_yaml


ACTIVATIONS = {'tanh': torch.tanh, 'relu': torch.relu}

###############################################################
#              UTILITY FUNCTIONS FOR AwkwardYaml              #
###############################################################

def _get_hidden_size(yaml_dict):
    if yaml_dict['mode'] == 'mlp':
        if 'hidden_sizes' in yaml_dict:
            hidden_size = yaml_dict['hidden_sizes'].strip()
        else:
            hidden_size = "(32, 32)"
    else:
        hidden_size = yaml_dict['embed_dim'] if 'embed_dim' in yaml_dict else 32
    return hidden_size


def _get_basic_parameters(yaml_dict):
    dropout = yaml_dict['dropout'] if 'dropout' in yaml_dict else 0
    hidden_size = _get_hidden_size(yaml_dict)
    mode = yaml_dict['mode']
    nonlinearity = yaml_dict['nonlinearity'] if 'nonlinearity' in yaml_dict else 'tanh'
    number_layers = yaml_dict['number_layers'] if 'number_layers' in yaml_dict else 1
    return dropout, hidden_size, mode, nonlinearity, number_layers


def _get_kwargs(yaml_dict):
    dropout, hidden_size, mode, nonlinearity, number_layers = _get_basic_parameters(yaml_dict)
    kwargs = {'dropout': dropout, 'nonlinearity': nonlinearity}
    if mode == 'deepset':
        kwargs.update({'output_size': yaml_dict['embed_dim']})
        phi_sizes, rho_sizes = (32, 32), (32, 32)
        kwargs.update({'phi_sizes': phi_sizes, 'rho_sizes': rho_sizes})
    elif mode == 'mlp':
        kwargs.update({'hidden_size': hidden_size, 'output_size': yaml_dict['embed_dim']})
    elif mode in ['vanilla_rnn', 'lstm', 'gru']:
        kwargs.update({'mode': mode, 'hidden_size': hidden_size, 'num_layers': number_layers})
    return kwargs


def _get_output_network_input_size(yaml_dict, dataset):
    input_size = 0
    if dataset.use_fixed_data:
        if yaml_dict['fixed_fields']['mode'] == 'identity':
            input_size += dataset.fixed_input_size
        else:
            input_size += yaml_dict['fixed_fields']['embed_dim']
    if dataset.use_jagged_data or dataset.use_nested_data:
        if yaml_dict['jagged_fields']['mode'] == 'identity':
            input_size += dataset.fixed_input_size
        else:
            input_size += yaml_dict['jagged_fields']['embed_dim']
    return input_size


def _append(X1, X2):
    if X1 == [] and X2 == []:
        return []
    elif X1 == []:
        return X2
    elif X2 == []:
        return X1
    return torch.cat((X1, X2), dim=2)


def _extend(X1, X2):
    if X1 == [] and X2 == []:
        return []
    elif X1 == []:
        return X2
    elif X2 == []:
        return X1
    X1.extend(X2)
    return X1

###############################################################


class AwkwardYaml(nn.Module):
    def __init__(self, yaml_dict, dataset, name, topnetwork=False):
        """"""
        super(AwkwardYaml, self).__init__()
        self.name = name
        self._check_network_usage(dataset)
        if self._use_output_net is False:
            return

        self._init_fixed_network(yaml_dict['fixed_fields'], dataset)
        self._init_jagged_network(yaml_dict['jagged_fields'], dataset)
        self._init_nested_networks(yaml_dict['jagged_fields'], dataset)
        self._init_output_network(yaml_dict, dataset)
        self.topnetwork = topnetwork
        if topnetwork:
            self.fc = nn.Linear(yaml_dict['embed_dim'], dataset.target_size)

    def forward(self, X):
        """"""
        self._check_correct_call()
        X_fixed, X_jagged, X_nested = X
        nested_output = self._forward_nested_network(X_nested)
        jagged_input = _extend(nested_output, X_jagged)
        hidden_jagged = self._forward_jagged_network(jagged_input)
        hidden_fixed = self._forward_fixed_network(X_fixed)
        latent_state = _append(hidden_fixed, hidden_jagged)
        return self._forward_output_network(latent_state)

    ########################################################################
    #                     helper functions for init                        #
    ########################################################################

    def _init_fixed_network(self, yaml_fixed_dict, dataset):
        if self._use_fixed_net:
            kwargs = _get_kwargs(yaml_fixed_dict)
            kwargs.update({'input_size': dataset.fixed_input_size})
            if yaml_fixed_dict['mode'] == 'deepset':
                self.fixed_net = AwkwardDeepSetSingleJagged(**kwargs)
            elif yaml_fixed_dict['mode'] in ['vanilla_rnn', 'lstm', 'gru']:
                self.fixed_net = AwkwardRNNSingleJagged(**kwargs)
            elif yaml_fixed_dict['mode'] == 'mlp':
                self.fixed_net = AwkwardMLP(**kwargs)
            elif yaml_fixed_dict['mode'] == 'identity':
                self.fixed_net = nn.Identity()

    def _init_jagged_network(self, yaml_jagged_dict, dataset):
        if self._use_jagged_net:
            kwargs = _get_kwargs(yaml_jagged_dict)
            if yaml_jagged_dict['mode'] == 'deepset':
                self.jagged_net = AwkwardDeepSetDoubleJagged(**kwargs)
            elif yaml_jagged_dict['mode'] in ['vanilla_rnn', 'lstm', 'gru']:
                self.jagged_net = AwkwardRNNDoubleJagged(**kwargs)
            elif yaml_jagged_dict['mode'] == 'mlp':
                kwargs.update({'input_size': dataset.jagged_input_size})
                self.jagged_net = AwkwardMLP(**kwargs)
            elif yaml_jagged_dict['mode'] == 'identity':
                self.fixed_net = nn.Identity()

    def _init_nested_networks(self, yaml_jagged_dict, dataset):
        self.nested_nets = []
        if self._use_nested_net:
            nested_fields, nested_names = get_nested_yaml(yaml_jagged_dict)
            zip_iter = zip(nested_fields, dataset.nested_datasets, nested_names)
            for field_i, dataset_i, name_i in zip_iter:
                self.nested_nets.append(AwkwardYaml(field_i, dataset_i, name_i))
        return

    def _init_output_network(self, yaml_dict, dataset):
        kwargs = self._init_output_net_parameters(yaml_dict, dataset)
        if self.output_mode == 'deepset':
            self.output_net = AwkwardDeepSetSingleJagged(**kwargs)
        elif self.output_mode == 'vanilla_rnn':
            self.output_net = nn.RNN(**kwargs)
        elif self.output_mode == 'gru':
            self.output_net = nn.GRU(**kwargs)
        elif self.output_mode == 'lstm':
            self.output_net = nn.LSTM(**kwargs)
        elif self.output_mode == 'mlp':
            self.output_net = AwkwardMLP(**kwargs)
        elif self.output_mode == 'identity':
            self.output_net = nn.Identity()

    def _init_output_net_parameters(self, yaml_dict, dataset):
        kwargs = _get_kwargs(yaml_dict)
        input_size = _get_output_network_input_size(yaml_dict, dataset)
        kwargs.update({'input_size': input_size})
        self.output_mode = yaml_dict['mode']
        self.output_nonlinearity = ACTIVATIONS[kwargs['nonlinearity']]
        if self.output_mode in ['gru', 'lstm', 'vanilla_rnn']:
            kwargs.pop('mode')
        if self.output_mode in ['gru', 'lstm']:
            kwargs.pop('nonlinearity')
        return kwargs

    ########################################################################
    #                   helper functions for forward                       #
    ########################################################################

    def _forward_fixed_network(self, X):
        if self._use_fixed_net and X != []:
            return self.fixed_net(X)
        return X

    def _forward_jagged_network(self, X):
        if self._use_jagged_net and X != []:
            return self.jagged_net(X)
        return X

    def _forward_nested_network(self, X):
        nested_output = []
        for net, x_nest in zip(self.nested_nets, X):
            nest_out_i = net(x_nest)
            if nest_out_i != []:
                nested_output.append(nest_out_i[0][0])
        return nested_output

    def _forward_output_network(self, X):
        if X == []:
            return []
        hidden = None
        if self.output_mode in ['deepset', 'mlp', 'identity']:
            output = self.output_net(X)
        elif self.output_mode in ['vanilla_rnn', 'gru']:
            output, hidden = self.output_net(X)
        elif self.output_mode == 'lstm':
            output, (hidden, cell) = self.output_net(X)
        return self._forward_output_nonlinearity(output, hidden)

    def _forward_output_nonlinearity(self, output, hidden):
        if self.topnetwork:
            x = self.fc(output)[0] #TODO: look at this again later
            y = F.log_softmax(x, dim=1)
            return y
        elif self.output_mode in ['deepset', 'mlp']:
            return self.output_nonlinearity(output)
        return self.output_nonlinearity(hidden)

    ########################################################################
    #                        misc. helper functions                        #
    ########################################################################

    def _check_network_usage(self, dataset):
        self._use_fixed_net = dataset.use_fixed_data
        self._use_jagged_net = dataset.use_jagged_data or dataset.use_nested_data
        self._use_nested_net = dataset.use_nested_data
        self._use_output_net = dataset.use_dataset
        return

    def _check_correct_call(self):
        if self._use_output_net is False:
            raise AssertionError("This network should not be called on data.")
