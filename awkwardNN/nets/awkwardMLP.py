import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_hidden_size_string(hidden_size_string):
    hidden_size_string = hidden_size_string[1:-1]
    hidden_size_string = hidden_size_string.replace(" ", "")
    hidden_size_string_list = hidden_size_string.split(",")
    hidden_size = [int(x) for x in hidden_size_string_list]
    return hidden_size


class AwkwardMLP(nn.Module):
    def __init__(self, *, input_size, hidden_size, output_size, nonlinearity, dropout):
        """
        RNN for single-jagged data
        e.g. list of events with varying number of particles with
             fixed number of features
        """
        super(AwkwardMLP, self).__init__()
        ACTIVATIONS = {'tanh': torch.tanh, 'relu': torch.relu}
        self.nonlinearity = ACTIVATIONS[nonlinearity]
        self.hidden_sizes = parse_hidden_size_string(hidden_size)
        self.nets = [nn.Linear(input_size, self.hidden_sizes[0])]
        for in_size, out_size in zip(self.hidden_sizes[1:-1], self.hidden_sizes[2:]):
            self.nets.append(nn.Dropout(dropout))
            self.nets.append(nn.Linear(in_size, out_size))


    def forward(self, X):
        for net in self.nets:
            X = self.nonlinearity(net(X))
        return X
