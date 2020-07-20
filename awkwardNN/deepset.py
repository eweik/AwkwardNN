import torch
import torch.nn as nn
import torch.nn.functional as F


ACTIVATIONS = {'tanh': F.tanh, 'relu': F.relu}


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
        for data_i in data:
            phi_output += self.phi(data_i)
        rho_output = self.rho(phi_output)
        return self.output(rho_output)
        #return F.log_softmax(rho_output, dim=1)


class AwkwardDeepSet(nn.Module):
    def __init__(self, input_size, phi_sizes, rho_sizes, activation, dropout):
        super(AwkwardDeepSet, self).__init__()
        output_size = 32
        self.particle_deepset = DeepSetNetwork(1, phi_sizes, rho_sizes,
                                               output_size, activation, dropout)
        self.event_deepset = DeepSetNetwork(output_size, phi_sizes, rho_sizes,
                                            2, activation, dropout)

    def forward(self, event):
        particle_deepset_output = []
        for particle in event:
            particle = torch.tensor([[[i]] for i in particle], dtype=torch.float32)
            particle_deepset_output.append(self.particle_deepset(particle))
        event_deepset_output = self.event_deepset(particle_deepset_output)
        return F.log_softmax(event_deepset_output, dim=1)

