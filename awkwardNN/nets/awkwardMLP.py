import torch
import torch.nn as nn

from awkwardNN.utils.yaml_utils import parse_hidden_size_string

ACTIVATIONS = {'tanh': torch.tanh, 'relu': torch.relu}

class AwkwardMLP(nn.Module):
    def __init__(self, *, input_size, hidden_size, output_size, nonlinearity, dropout):
        """
        RNN for single-jagged data
        e.g. list of events with varying number of particles with
             fixed number of features
        """
        super(AwkwardMLP, self).__init__()
        self.nonlinearity = ACTIVATIONS[nonlinearity]
        self.hidden_sizes = parse_hidden_size_string(hidden_size)
        self.nets = [nn.Linear(input_size, self.hidden_sizes[0]), nn.Dropout(dropout)]
        for in_size, out_size in zip(self.hidden_sizes[0:-1], self.hidden_sizes[1:]):
            self.nets.append(nn.Linear(in_size, out_size))
            self.nets.append(nn.Dropout(dropout))
        self.nets.append(nn.Linear(self.hidden_sizes[-1], output_size))


    def forward(self, X):
        if isinstance(X, list):
            X = [torch.tensor(i, dtype=torch.float32).unsqueeze(0) for i in X]
        for x in X:
            for net in self.nets:
                x = self.nonlinearity(net(x)) # TODO fix this
        return x.unsqueeze(0)
