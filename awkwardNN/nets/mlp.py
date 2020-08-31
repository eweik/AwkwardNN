import torch
import torch.nn as nn

ACTIVATIONS = {'tanh': torch.tanh, 'relu': torch.relu}

class MLP(nn.Module):
    def __init__(self, *, input_size, hidden_size, output_size, nonlinearity, dropout):
        """
        RNN for single-jagged data
        e.g. list of events with varying number of particles with
             fixed number of features
        """
        super(MLP, self).__init__()
        self.nonlinearity = ACTIVATIONS[nonlinearity]
        self.nets = [nn.Linear(input_size, hidden_size[0]), nn.Dropout(dropout)]
        for in_size, out_size in zip(hidden_size[0:-1], hidden_size[1:]):
            self.nets.append(nn.Linear(in_size, out_size))
            self.nets.append(nn.Dropout(dropout))
        self.nets.append(nn.Linear(hidden_size[-1], output_size))


    def forward(self, X):
        for net in self.nets:
            X = self.nonlinearity(net(X))
        return X
