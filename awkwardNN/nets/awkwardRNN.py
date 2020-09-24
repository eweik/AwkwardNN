import torch
import torch.nn as nn


######################################################################
#                   helper functions for awkward rnn                 #
######################################################################

def _get_rnn_subnetwork(mode, input_size, hidden_size, nonlinearity, dropout):
    kwargs = {'dropout': dropout}
    if mode == 'vanilla_rnn':
        kwargs.update({'nonlinearity': nonlinearity})
        return nn.RNN(input_size, hidden_size, **kwargs)
    elif mode == 'lstm':
        return nn.LSTM(input_size, hidden_size, **kwargs)
    return nn.GRU(input_size, hidden_size, **kwargs)

def _get_hidden_state(mode, rnn_output):
    if mode == 'lstm':
        _, (hidden, _) = rnn_output
    else:
        _, hidden = rnn_output
    return hidden


######################################################################

class RNNDoubleStacked(nn.Module):
    def __init__(self, *, mode, hidden_size, nonlinearity, dropout):
        """
        RNN for double-jagged data
        e.g. list of events with varying number of particles with
             varying number of features

        :param mode: _str_ in ['lstm', 'rnn', 'gru']
        :param hidden_size: int
        :param nonlinearity: str in ['tanh', 'relu']
        :param dropout: int in range [0, 1)
        """
        super(RNNDoubleStacked, self).__init__()
        input_size = 1
        self.hidden_size = int(hidden_size)
        self.mode = mode
        self.net1 = _get_rnn_subnetwork(mode, input_size, hidden_size, nonlinearity, dropout)
        self.net2 = _get_rnn_subnetwork(mode, hidden_size, hidden_size, nonlinearity, dropout)

    def forward(self, event):
        """
        """
        hidden_list = []

        # forward through first network
        for particle in event:
            particle = particle.squeeze(0)
            hidden = _get_hidden_state(self.mode, self.net1(particle))
            hidden_list.append(hidden)

        # forward through second network
        hidden_list = torch.cat(hidden_list, dim=0)
        hidden = _get_hidden_state(self.mode, self.net2(hidden_list))
        return hidden

