import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_rnn_subnetwork(mode, input_size, hidden_size, num_layers, activation, dropout):
    kwargs = {'num_layers': num_layers, 'dropout': dropout}
    if mode == 'rnn':
        kwargs.update({'nonlinearity': activation})
        return nn.RNN(input_size, hidden_size, **kwargs)
    elif mode == 'lstm':
        return nn.LSTM(input_size, hidden_size, **kwargs)
    # elif mode == 'gru':
    return nn.GRU(input_size, hidden_size, **kwargs)


def _reset_state(hidden_event, cell_event):
    # create hidden (and cell) state for rnn (and lstm) going through particle
    hidden_particle = torch.zeros_like(hidden_event)
    cell_particle = torch.zeros_like(cell_event)
    hidden = torch.cat((hidden_event, hidden_particle), dim=2)
    cell = torch.cat((cell_event, cell_particle), dim=2)
    return hidden, cell


def _extract_event_state(hidden, cell, hidden_size):
    # hidden_event = hidden.clone().detach()[:, :, hidden_size:].requires_grad_(True)
    # cell_event = cell.clone().detach()[:, :, hidden_size:].requires_grad_(True)
    hidden_event = hidden.clone().detach()[:, :, hidden_size:]
    cell_event = cell.clone().detach()[:, :, hidden_size:]
    return hidden_event, cell_event


class AwkwardRNNDoubleJagged(nn.Module):
    def __init__(self, *, mode, hidden_size, num_layers,
                 output_size, activation, dropout):
        """
        RNN for double-jagged data
        e.g. list of events with varying number of particles with
             varying number of features

        :param mode: _str_ in ['lstm', 'rnn', 'gru']
        :param hidden_size: _int_
            the number of nodes in the hidden layers of the rnn
        :param num_layers: _int_
            the number of hidden layers in the rnn
        :param output_size: _int_
            the number of nodes in output layer
        :param activation: _str_ in ['tanh', 'relu']
            only relevant for 'rnn' mode
        :param dropout: _int_ in range [0, 1)
        """
        super(AwkwardRNNDoubleJagged, self).__init__()
        input_size = 1
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.mode = mode
        self.net = _get_rnn_subnetwork(mode, input_size, 2*hidden_size,
                                       num_layers, activation, dropout)
        self.output = nn.Linear(2*hidden_size, output_size)

    def forward(self, event):
        """
        TODO: walk through this in debug mode, make sure it's working correctly
        :param event:
        :return:
        """
        hidden_event = torch.zeros(self.num_layers, 1, self.hidden_size)
        cell_event = torch.zeros(self.num_layers, 1, self.hidden_size)
        for particle in event:
            hidden, cell = _reset_state(hidden_event, cell_event)
            particle = torch.tensor([[[i]] for i in particle], dtype=torch.float32)
            if self.mode == 'lstm':
                output, (hidden, cell) = self.net(particle, (hidden, cell))
            else:
                output, hidden = self.net(particle, hidden)
            hidden_event, cell_event = _extract_event_state(hidden, cell)
        return F.log_softmax(self.output(output[-1]), dim=1)


class AwkwardRNNSingleJagged(nn.Module):
    def __init__(self, *, mode, input_size, hidden_size,
                 num_layers, output_size, activation, dropout):
        """
        RNN for single-jagged data
        e.g. list of events with varying number of particles with
             fixed number of features
        """
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.mode = mode
        self.net = _get_rnn_subnetwork(mode, input_size, hidden_size,
                                       num_layers, activation, dropout)
        self.output = nn.Linear(2 * hidden_size, output_size)

    def forward(self, event):
        hidden = torch.zeros(self.num_layers, 1, self.hidden_size)
        cell = torch.zeros(self.num_layers, 1, self.hidden_size)
        if self.mode == 'lstm':
            output, (_, _) = self.net(event, (hidden, cell))
        else:
            output, _ = self.net(event, hidden)
        return F.log_softmax(self.output(output[-1]), dim=1)

