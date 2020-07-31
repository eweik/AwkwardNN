import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from awkwardNN.deepset import DeepSetNetwork


def _get_rnn_subnetwork(mode, input_size, hidden_size, num_layers, activation, dropout):
    kwargs = {'num_layers': num_layers, 'dropout': dropout}
    if mode == 'rnn':
        kwargs.update({'nonlinearity': activation})
        return nn.RNN(input_size, hidden_size, **kwargs)
    elif mode == 'lstm':
        return nn.LSTM(input_size, hidden_size, **kwargs)
    # elif mode == 'gru':
    return nn.GRU(input_size, hidden_size, **kwargs)


class AwkwardMixed(nn.Module):
    def __init__(self, *, num_rnns, num_grus, num_lstms, num_deepsets,
                 rnn_input_sizes, rnn_hidden_size,
                 rnn_num_layers, rnn_activation,
                 lstm_input_sizes, lstm_hidden_size, lstm_num_layers,
                 gru_input_sizes, gru_hidden_size, gru_num_layers,
                 deepset_input_sizes, deepset_phi_sizes,
                 deepset_rho_sizes, deepset_activation,
                 output_size, dropout):
        """

        :param num_rnns:
        :param num_grus:
        :param num_lstms:
        :param num_deepsets:
        :param rnn_hidden_size:
        :param rnn_num_layers:
        :param rnn_activation:
        :param lstm_hidden_size:
        :param lstm_num_layers:
        :param gru_hidden_size:
        :param gru_num_layers:
        :param deepset_input_sizes:
        :param deepset_phi_sizes:
        :param deepset_rho_sizes:
        :param output_size:
        :param deepset_activation:
        """
        super(AwkwardMixed, self).__init__()
        self.num_rnns = num_rnns
        self.num_grus = num_grus
        self.num_lstms = num_lstms
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.gru_hidden_size = gru_hidden_size
        self.gru_num_layers = gru_num_layers

        self.dropout = dropout
        self._init_rnns(rnn_input_sizes, rnn_hidden_size,
                        rnn_num_layers, rnn_activation)
        self._init_grus(gru_input_sizes, gru_hidden_size, gru_num_layers)
        self._init_lstms(lstm_input_sizes, lstm_hidden_size, lstm_num_layers)
        self._init_deepsets(deepset_input_sizes, deepset_phi_sizes,
                            deepset_rho_sizes, deepset_activation)
        self._init_output_network(num_rnns, rnn_hidden_size,
                                  num_grus, gru_hidden_size,
                                  num_lstms, lstm_hidden_size,
                                  num_deepsets, deepset_rho_sizes,
                                  output_size)


    def forward(self, X_rnn, X_gru, X_lstm, X_deepset, y):
        """

        :param X_rnn:
        :param X_lstm:
        :param X_gru:
        :param X_deepset:
        :param y:
        :return:
        """
        self._init_hidden_state()
        latent = []

        for x_rnn, hidden, rnn in zip(X_rnn, self.hidden_rnns, self.rnns):
            output, hidden = rnn(x_rnn, hidden)
            latent.append(hidden)
        for x_lstm, (hidden, cell), lstm in zip(X_lstm, self.hidden_lstms, self.lstms):
            output, (hidden, cell) = lstm(x_lstm, (hidden, cell))
            latent.append(hidden)
        for x_gru, hidden, gru in zip(X_gru, self.hidden_grus, self.grus):
            output, hidden = gru(x_gru, hidden)
            latent.append(hidden)
        for x_deepset, deepset in zip(X_deepset, self.deepsets):
            output = deepset(x_deepset)
            latent.append(output)

        latent = F.tanh(self.fc1(latent))
        latent = self.dropout1(latent)
        latent = F.tanh(self.fc2(latent))
        latent = self.dropout2(latent)
        latent = F.tanh(self.fc3(latent))
        return F.log_softmax(latent, dim=1)

    def _init_hidden_state(self):
        self.hidden_rnns = [torch.zeros(self.num_layers, 1, self.hidden_size)
                            for _ in range(self.num_rnns)]

        # tuple of (hidden, cell) state for each lstm
        self.hidden_lstms = [(torch.zeros(self.lstm_num_layers, 1, self.lstm_hidden_size),
                              torch.zeros(self.lstm_num_layers, 1, self.lstm_hidden_size))
                             for _ in range(self.lstm_num_layers)]
        self.hidden_grus = [torch.zeros(self.gru_num_layers, 1, self.gru_hidden_size)
                            for _ in range(self.num_grus)]

    def _init_rnns(self, input_sizes, hidden_size, num_layers, activation):
        self.rnns = []
        kwargs = {'num_layers': num_layers, 'dropout': self.dropout,
                  'activation': activation}
        for in_sz in input_sizes:
            self.rnns.append(nn.RNN(in_sz, hidden_size, **kwargs))

    def _init_lstms(self, input_sizes, hidden_size, num_layers):
        self.lstms = []
        kwargs = {'num_layers': num_layers, 'dropout': self.dropout}
        for in_sz in input_sizes:
            self.lstms.append(nn.LSTM(in_sz, hidden_size, **kwargs))

    def _init_grus(self, input_sizes, hidden_size, num_layers):
        self.grus = []
        kwargs = {'num_layers': num_layers, 'dropout': self.dropout}
        for in_sz in input_sizes:
            self.grus.append(nn.GRU(in_sz, hidden_size, **kwargs))

    def _init_deepsets(self, input_sizes, phi_sizes, rho_sizes, activation):
        self.deepsets = []
        for in_sz in input_sizes:
            deepset_net = DeepSetNetwork(in_sz, phi_sizes, rho_sizes,
                                         activation, self.dropout)
            self.deepsets.append(deepset_net)

    def _init_output_network(self, num_rnns, rnn_hidden_size,
                             num_grus, gru_hidden_size,
                             num_lstms, lstm_hidden_size,
                             num_deepsets, deepset_rho_size,
                             output_size):
        input_size = num_rnns * rnn_hidden_size + num_grus * gru_hidden_size + \
                     num_lstms * lstm_hidden_size + num_deepsets * deepset_rho_size[-1]
        hidden_size = 100
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(self.dropout)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(self.dropout)
        self.fc3 = nn.Linear(hidden_size, output_size)
