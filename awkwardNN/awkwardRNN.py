import torch
import torch.nn as nn
import torch.nn.functional as F


def get_rnn_subnetwork(mode, input_size, hidden_size, num_layers, activation, dropout):
    kwargs = {'num_layers': num_layers, 'dropout': dropout}
    hidden_size *= 2
    if mode == 'rnn':
        kwargs.update({'nonlinearity': activation})
        return nn.RNN(input_size, hidden_size, **kwargs)
    elif mode == 'lstm':
        return nn.LSTM(input_size, hidden_size, **kwargs)
    # elif mode == 'gru':
    return nn.GRU(input_size, hidden_size, **kwargs)


class AwkwardRNN(nn.Module):
    def __init__(self, mode, input_size, hidden_size, num_layers, activation, dropout):
        """
        :param mode: _str_ in ['lstm', 'rnn', 'gru']
        :param input_size: _int_
            the number of features in the input
        :param hidden_size: _int_
            the number of nodes in the hidden layers of the rnn
        :param num_layers: _int_
            the number of hidden layers in the rnn
        :param activation: _str_ in ['tanh', 'relu']
        :param dropout: _int_ in range [0, 1)
        """
        super(AwkwardRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.mode = mode
        #self.net = get_rnn_subnetwork(mode, input_size, hidden_size,
        #                              num_layers, activation, dropout)
        in_size = 1
        self.net = get_rnn_subnetwork(mode, in_size, hidden_size,
                                      num_layers, activation, dropout)
        self.output = nn.Linear(hidden_size*2, 2)

    def forward(self, event):
        hidden_event = torch.zeros(self.num_layers, 1, self.hidden_size)
        cell_event = torch.zeros(self.num_layers, 1, self.hidden_size)
        for particle in event:
            hidden_particle = torch.zeros(self.num_layers, 1, self.hidden_size)
            cell_particle = torch.zeros(self.num_layers, 1, self.hidden_size)
            hidden = torch.cat((hidden_event, hidden_particle), dim=2)
            cell = torch.cat((cell_event, cell_particle), dim=2)
            particle = torch.tensor([[[i]] for i in particle], dtype=torch.float32)
            if self.mode in ['rnn', 'gru']:
                output, hidden = self.net(particle, hidden)
            else: # lstm
                output, (hidden, cell) = self.net(particle, (hidden, cell))
        linear_output = self.output(output[-1])
        out = F.log_softmax(linear_output, dim=1)
        return out

'''
class AwkwardRNN(nn.Module):
    def __init__(self, mode, max_depth, input_size, hidden_size,
                 output_size, activation, dropout):
        super(AwkwardRNN, self).__init__()
        self.max_depth = max_depth
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.mode = mode
        self.layers = []
        for _ in range(max_depth):
            subnetwork = get_rnn_subnetwork(mode, input_size, hidden_size, activation, dropout)
            self.layers.append(subnetwork)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, data, markers):
        i, j = 0, self.input_size
        data, markers = data[0], markers[0]
        hidden = torch.zeros(1, self.hidden_size)
        cell = torch.zeros(1, self.hidden_size)
        for marker, layer in zip(markers, self.layers):
            for _ in range(marker):
                x = torch.tensor([[data[i:j]]], dtype=torch.float32)
                if self.mode in ['rnn', 'gru']:
                    hidden = layer(x, hidden)
                else:  # lstm
                    hidden, cell = layer(x, (hidden, cell))
                i += self.input_size
                j += self.input_size
        return F.log_softmax(self.output(hidden), dim=1)
'''
