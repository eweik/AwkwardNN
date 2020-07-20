"""Base network for Awkward NN"""


import torch.nn as nn


class AwkwardBase(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AwkwardBase, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size




