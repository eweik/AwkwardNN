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


def _reset_state(hidden_event, cell_event):
    hidden_particle = torch.zeros_like(hidden_event)
    cell_particle = torch.zeros_like(cell_event)
    hidden = torch.cat((hidden_event, hidden_particle), dim=2)
    cell = torch.cat((cell_event, cell_particle), dim=2)
    return hidden, cell


def _extract_event_state(hidden, cell, hidden_size):
    hidden_event = hidden.clone().detach()[:, :, hidden_size:].requires_grad_(True)
    cell_event = cell.clone().detach()[:, :, hidden_size:].requires_grad_(True)
    # hidden_event = hidden.clone().detach()[:, :, hidden_size:]
    # cell_event = cell.clone().detach()[:, :, hidden_size:]
    return hidden_event, cell_event

######################################################################


class AwkwardRNNDoubleJagged(nn.Module):
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
        super(AwkwardRNNDoubleJagged, self).__init__()
        input_size = 1
        # halve hidden size is because of double jaggedness
        self.hidden_size = int(hidden_size / 2)
        self.mode = mode
        self.net = _get_rnn_subnetwork(mode, input_size, hidden_size,
                                       nonlinearity, dropout)

    def forward(self, event):
        """
        TODO: walk through this in debug mode, make sure it's working correctly
        :param event:
        :return:
        """
        hidden_event = torch.zeros(1, 1, self.hidden_size)
        cell_event = torch.zeros(1, 1, self.hidden_size)
        for particle in event:
            hidden, cell = _reset_state(hidden_event, cell_event)
            particle = torch.squeeze(particle, dim=0)
            if self.mode == 'lstm':
                output, (hidden, cell) = self.net(particle, (hidden, cell))
            else:
                output, hidden = self.net(particle, hidden)
            hidden_event, cell_event = _extract_event_state(hidden, cell, self.hidden_size)
        return hidden


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
        super(AwkwardRNNDoubleJagged, self).__init__()
        input_size = 1
        # halve hidden size is because of double jaggedness
        self.hidden_size = int(hidden_size)
        self.mode = mode
        self.net1 = _get_rnn_subnetwork(mode, input_size, hidden_size, nonlinearity, dropout)
        self.net2 = _get_rnn_subnetwork(mode, hidden_size, hidden_size, nonlinearity, dropout)

    def forward(self, event):
        """
        """
        hidden_list = []
        for particle in event:
            if self.mode == 'lstm':
                _, (hidden, _) = self.net1(particle)
            else:
                _, hidden = self.net1(particle)
            hidden_list.append(hidden)
        if self.mode == 'lstm':
            _, (hidden, _) = self.net2(hidden_list)
        else:
            _, hidden = self.net1(hidden_list)
        return hidden

