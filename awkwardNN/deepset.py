import torch
import torch.nn as nn
import torch.nn.functional as F


ACTIVATIONS = {'tanh': torch.tanh, 'relu': torch.relu}


class FCNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, activation, dropout):
        super(FCNetwork, self).__init__()
        self.nonlinear = ACTIVATIONS[activation]
        self.input_size = input_size
        self.hidden_layers = [nn.Linear(input_size, hidden_sizes[0])]
        self.dropout = nn.Dropout(dropout)
        for in_size, out_size in zip(hidden_sizes, hidden_sizes[1:]):
            self.hidden_layers.append(nn.Linear(in_size, out_size))

    def forward(self, x):
        for linear in self.hidden_layers:
            x = self.nonlinear(linear(x))
            x = self.dropout(x)
        return x


class DeepSetNetwork(nn.Module):
    def __init__(self, input_size, phi_sizes, rho_sizes, output_size, activation, dropout):
        """
        :param input_size: _int_
            the number of features in the input
        :param phi_sizes: {_tuple_, _list_} of _int_
            the number of nodes in the hidden layers of the phi network
        :param rho_sizes: {_tuple_, _list_} of _int_
            the number of nodes in the hidden layers of the rho network
        :param output_size: _int_
            the number of nodes in the output layer
        :param activation: _str_
            the non-linear function to use
        :param dropout: _float_
            probability of a node to be zeroed out
        """
        super(DeepSetNetwork, self).__init__()
        self.phi = FCNetwork(input_size, phi_sizes, activation, dropout)
        self.rho = FCNetwork(phi_sizes[-1], rho_sizes, activation, dropout)
        self.output = nn.Linear(rho_sizes[-1], output_size)

    def forward(self, data):
        phi_output = self.phi(data[0])
        for data_i in data[1:]:
            phi_output += self.phi(data_i)
        rho_output = self.rho(phi_output)
        return self.output(rho_output)


class AwkwardDeepSetSingleJagged(nn.Module):
    def __init__(self, *, input_size, phi_sizes, rho_sizes,
                 output_size, activation, dropout):
        """
        Deepset for single-jagged data
        e.g. list of events with varying number of particles with
             fixed number of features

        :param input_size:
        :param phi_sizes:
        :param rho_sizes:
        :param output_size:
        :param activation:
        :param dropout:
        """
        super(AwkwardDeepSetSingleJagged, self).__init__()
        self.deepset = DeepSetNetwork(input_size, phi_sizes, rho_sizes,
                                      output_size, activation, dropout)

    def forward(self, data):
        data = torch.tensor([[i] for i in data], dtype=torch.float32)
        output = self.deepset(data)
        return F.log_softmax(output, dim=1)


class AwkwardDeepSetDoubleJagged(nn.Module):
    def __init__(self, *, phi_sizes, rho_sizes,
                 output_size, activation, dropout):
        """
        Deepset for double-jagged data
        e.g. list of events with varying number of particles with
             varying number of features

        :param phi_sizes:
        :param rho_sizes:
        :param output_size:
        :param activation:
        :param dropout:
        """
        super(AwkwardDeepSetDoubleJagged, self).__init__()
        input_size = 1
        self.activation = ACTIVATIONS[activation]
        self.deepset1 = DeepSetNetwork(input_size, phi_sizes, rho_sizes,
                                       rho_sizes[-1], activation, dropout)
        self.deepset2 = DeepSetNetwork(rho_sizes[-1], phi_sizes, rho_sizes,
                                       output_size, activation, dropout)

    def forward(self, data):
        deepset2_input = []
        for data_i in data:
            data_i = torch.tensor([[[i]] for i in data_i], dtype=torch.float32)
            deepset1_output = self.activation(self.deepset1(data_i))
            deepset2_input.append(deepset1_output)
        deepset2_output = self.deepset2(deepset2_input)
        return F.log_softmax(deepset2_output, dim=1)





