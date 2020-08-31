import torch
import torch.nn as nn


ACTIVATIONS = {'tanh': torch.tanh, 'relu': torch.relu}


class FCNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, activation, dropout):
        super(FCNetwork, self).__init__()
        # self.nonlinear = ACTIVATIONS[activation]
        self.nonlinear = torch.relu
        self.input_size = input_size
        self.hidden_layers = [nn.Linear(input_size, hidden_size[0])]
        self.dropout = nn.Dropout(dropout)
        for in_size, out_size in zip(hidden_size, hidden_size[1:]):
            self.hidden_layers.append(nn.Linear(in_size, out_size))

    def forward(self, x):
        for linear in self.hidden_layers:
            x = self.nonlinear(linear(x))
            x = self.dropout(x)
        return x


class DeepSetNetwork(nn.Module):
    def __init__(self, *, phi_sizes, rho_sizes, output_size, nonlinearity, dropout):
        """
        :param input_size: int
        :param phi_sizes: {tuple, list} of int
        :param rho_sizes: {tuple, list} of int
        :param output_size: int
        :param nonlinearity: str
        :param dropout: float
        """
        super(DeepSetNetwork, self).__init__()
        self.input_size = 1
        self.phi = FCNetwork(self.input_size, phi_sizes, nonlinearity, dropout)
        self.rho = FCNetwork(phi_sizes[-1], rho_sizes, nonlinearity, dropout)
        self.output = nn.Linear(rho_sizes[-1], output_size)

    def forward(self, data):
        phi_output = self.phi(data[0])
        for data_i in data[1:]:
            phi_output += self.phi(data_i)
        rho_output = self.rho(phi_output)
        return self.output(rho_output)


class AwkwardDeepSetSingleJagged(nn.Module):
    def __init__(self, *, input_size, phi_sizes, rho_sizes, output_size, nonlinearity, dropout):
        """
        Deepset for single-jagged data
        e.g. list of events with varying number of particles with
             fixed number of features

        :param input_size:
        :param phi_sizes:
        :param rho_sizes:
        :param output_size:
        :param nonlinearity:
        :param dropout:
        """
        super(AwkwardDeepSetSingleJagged, self).__init__()
        self.input_size = input_size
        self.deepset = DeepSetNetwork(input_size=input_size, phi_sizes=phi_sizes,
                                      rho_sizes=rho_sizes, output_size=output_size,
                                      nonlinearity=nonlinearity, dropout=dropout)

    def forward(self, data):
        if isinstance(data, list):
            data = torch.tensor([[i] for i in data], dtype=torch.float32) # TODO: try unsqueeze later
        return self.deepset(data).unsqueeze(0)


class AwkwardDeepSetDoubleJagged(nn.Module):
    def __init__(self, *, phi_sizes, rho_sizes, output_size, nonlinearity, dropout):
        """
        Deepset for double-jagged data
        e.g. list of events with varying number of particles with
             varying number of features

        :param phi_sizes:
        :param rho_sizes:
        :param output_size:
        :param nonlinearity:
        :param dropout:
        """
        super(AwkwardDeepSetDoubleJagged, self).__init__()
        self.input_size = 1
        self.phi_sizes = phi_sizes
        self.rho_sizes = rho_sizes
        self.activation = ACTIVATIONS[nonlinearity]
        self.deepset1 = DeepSetNetwork(input_size=self.input_size, phi_sizes=phi_sizes,
                                       rho_sizes=rho_sizes, output_size=rho_sizes[-1],
                                       nonlinearity=nonlinearity, dropout=dropout)
        self.deepset2 = DeepSetNetwork(input_size=rho_sizes[-1], phi_sizes=phi_sizes,
                                       rho_sizes=rho_sizes, output_size=output_size,
                                       nonlinearity=nonlinearity, dropout=dropout)

    def forward(self, data):
        deepset2_input = []
        for data_i in data:
            data_i = torch.tensor([[[i]] for i in data_i], dtype=torch.float32)
            deepset1_output = self.activation(self.deepset1(data_i))
            deepset2_input.append(deepset1_output)
        return self.deepset2(deepset2_input).unsqueeze(0)



