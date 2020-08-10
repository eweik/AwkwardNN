import torch
import torch.nn as nn
import torch.nn.functional as F
from awkwardNN.nets.deepset import AwkwardDeepSetDoubleJagged
from awkwardNN.nets.awkwardRNN import AwkwardRNNDoubleJagged


class AwkwardMixed(nn.Module):
    def __init__(self, modes, hidden_size, num_layers, phi_sizes, 
                 rho_sizes, output_size, activation, dropout):
        """"""
        super(AwkwardMixed, self).__init__()
        self.networks = []
        self.modes = modes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        latent = 100
        for mode in modes:
            kwargs = {'output_size': latent, 'activation': activation, 'dropout': dropout}
            if mode == 'deepset':
                kwargs.update({'phi_sizes': phi_sizes, 'rho_sizes': rho_sizes})
                net = AwkwardDeepSetDoubleJagged(**kwargs)
            else:
                kwargs.update({'mode': mode, 'hidden_size': hidden_size, 'num_layers': num_layers})
                net = AwkwardRNNDoubleJagged(**kwargs)
            self.networks.append(net)

        self._init_output_network(modes, latent, output_size)


    def forward(self, X):
        """"""
        latent = []
        for field_chunk, net, mode in zip(X, self.networks, self.modes):
            if mode in ['rnn', 'gru']:
                out = net(field_chunk)
            elif mode == 'lstm':
                out = net(field_chunk)
            elif mode == 'deepset':
                out = net(field_chunk)
            latent.extend(out)

        latent = torch.cat(latent)
        latent = torch.tanh(self.fc1(latent))
        latent = self.dropout1(latent)
        latent = torch.tanh(self.fc2(latent))
        latent = self.dropout2(latent)
        latent = self.fc3(latent).view(1, -1)
        return F.log_softmax(latent, dim=1)

    def _init_output_network(self, modes, latent, output_size):
        input_size = len(modes) * latent
        fc_hidden_size = 100
        self.fc1 = nn.Linear(input_size, fc_hidden_size)
        self.dropout1 = nn.Dropout(self.dropout)
        self.fc2 = nn.Linear(fc_hidden_size, fc_hidden_size)
        self.dropout2 = nn.Dropout(self.dropout)
        self.fc3 = nn.Linear(fc_hidden_size, output_size)
