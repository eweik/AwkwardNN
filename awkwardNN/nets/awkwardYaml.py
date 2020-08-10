import torch
import torch.nn as nn
import torch.nn.functional as F

from awkwardNN.nets.deepset import AwkwardDeepSetDoubleJagged, AwkwardDeepSetSingleJagged
from awkwardNN.nets.awkwardRNN import AwkwardRNNDoubleJagged, AwkwardRNNSingleJagged
from awkwardNN.nets.awkwardMLP import AwkwardMLP


def _get_basic_parameters(yaml_dict):
    activation = yaml_dict['nonlinearity'] if 'nonlinearity' in yaml_dict else 'tanh'
    dropout = yaml_dict['dropout'] if 'dropout' in yaml_dict else 0
    number_layers = yaml_dict['number_layers'] if 'number_layers' in yaml_dict else 1
    mode = yaml_dict['mode']
    if mode == 'mlp':
        hidden_size = yaml_dict['hidden_sizes'] if 'hidden_sizes' in yaml_dict else "(32, 32)"
    else:
        hidden_size = yaml_dict['embed_dim'] if 'embed_dim' in yaml_dict else 32
    return activation, dropout, hidden_size, number_layers, mode


class AwkwardYaml(nn.Module):
    def __init__(self, yaml_dict, dataset, lastnetwork=False):
        """"""
        super(AwkwardYaml, self).__init__()
        self._init_fixed_network(yaml_dict['fixed_fields'], dataset)
        self._init_jagged_network(yaml_dict['jagged_fields'], dataset)
        self._init_nested_networks(yaml_dict, dataset)
        self._init_output_network(yaml_dict)
        self.lastnetwork = lastnetwork
        if lastnetwork:
            self.fc = nn.Linear(yaml_dict['embed_dim'], dataset.output_size)

    def forward(self, X):
        """"""
        X_fixed, X_jagged, X_nested = X

        # go through nested datasets first
        nested_output = []
        for net, x_nest in zip(self.nested_nets, X_nested):
            nested_output.extend(net(x_nest))

        # then go through jagged data + nested output
        jagged_input = X_jagged + nested_output
        hidden_jagged = self.jagged_net(jagged_input)

        # go through fixed data
        hidden_fixed = self.fixed_net(X_fixed)

        # finally go through final net
        latent_state = hidden_fixed + hidden_jagged
        return self._forward_output_network(latent_state)

    def _init_fixed_network(self, fixed_dict, dataset):
        activation, dropout, hidden_size, number_layers, mode = _get_basic_parameters(fixed_dict)
        kwargs = {'input_size': dataset.fixed_input_size, 'nonlinearity': activation, 'dropout': dropout}

        # initialize network
        if mode == 'deepset':
            # kwargs.update({'phi_sizes': phi_sizes, 'rho_sizes': rho_sizes})
            self.fixed_net = AwkwardDeepSetSingleJagged(**kwargs)
        elif mode in ['vanilla_rnn', 'lstm', 'gru']:
            kwargs.update({'mode': mode, 'hidden_size': hidden_size, 'num_layers': number_layers})
            self.fixed_net = AwkwardRNNSingleJagged(**kwargs)
        elif mode == 'mlp':
            kwargs.update({'hidden_size': hidden_size, 'output_size': fixed_dict['embed_dim']})
            self.fixed_net = AwkwardMLP(**kwargs)

    def _init_jagged_network(self, jagged_dict, dataset):
        activation, dropout, hidden_size, number_layers, mode = _get_basic_parameters(jagged_dict)
        kwargs = {'nonlinearity': activation, 'dropout': dropout}

        # initialize network
        if mode == 'deepset':
            # kwargs.update({'phi_sizes': phi_sizes, 'rho_sizes': rho_sizes})
            self.jagged_net = AwkwardDeepSetDoubleJagged(**kwargs)
        elif mode in ['vanilla_rnn', 'lstm', 'gru']:
            kwargs.update({'mode': mode, 'hidden_size': hidden_size, 'num_layers': number_layers})
            self.jagged_net = AwkwardRNNDoubleJagged(**kwargs)
        elif mode == 'mlp':
            kwargs.update({'input_size': dataset.jagged_input_size,
                           'hidden_size': hidden_size,
                           'output_size': jagged_dict['embed_dim']})
            self.jagged_net = AwkwardMLP(**kwargs)
        return

    def _init_nested_networks(self, yaml_dict, dataset):
        self.nested_nets = []
        jagged_field_list = yaml_dict['jagged_fields']['fields']
        for field_i, dataset_i in zip(jagged_field_list, dataset.nested_datasets):
            if isinstance(field_i, dict):
                self.nested_nets.append(AwkwardYaml(field_i, dataset_i))
        return

    def _init_output_network(self, yaml_dict):
        activation, dropout, hidden_size, number_layers, mode = _get_basic_parameters(yaml_dict)
        self.mode = mode
        self.activation = activation
        input_size = yaml_dict['fixed_fields']['embed_dim'] + yaml_dict['jagged_fields']['embed_dim']
        kwargs = {'input_size': input_size, 'nonlinearity': activation, 'dropout': dropout}

        # initialize network
        if mode == 'deepset':
            # kwargs.update({'phi_sizes': phi_sizes, 'rho_sizes': rho_sizes})
            self.output_net = AwkwardDeepSetSingleJagged(**kwargs)
        elif mode == 'vanilla_rnn':
            kwargs.update({'hidden_size': hidden_size, 'num_layers': number_layers})
            self.output_net = nn.RNN(**kwargs)
        elif mode == 'gru':
            kwargs.update({'hidden_size': hidden_size, 'num_layers': number_layers})
            self.output_net = nn.GRU(**kwargs)
        elif mode == 'lstm':
            kwargs.update({'hidden_size': hidden_size, 'num_layers': number_layers})
            self.output_net = nn.LSTM(**kwargs)
        elif mode == 'mlp':
            kwargs.update({'hidden_size': hidden_size, 'output_size': yaml_dict['embed_dim']})
            self.output_net = AwkwardMLP(**kwargs)

    def _forward_output_network(self, X):
        if self.mode in ['deepset', 'mlp']:
            output = self.output_net(X)
        elif self.mode in ['vanilla_rnn', 'gru']:
            output, hidden = self.output_net(X)
        elif self.mode == 'lstm':
            output, (hidden, cell) = self.output_net(X)

        if self.lastnetwork:
            return F.log_softmax(self.fc(output), dim=1)
        elif self.mode in ['deepset', 'mlp']:
            return self.activation(output)
        return self.activation(hidden)
