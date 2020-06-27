# awkwardNet.py
# Create RNN for awkward (variable-length and nested) data structre
#

import torch
import torch.nn as nn
import torch.nn.functional as F


class AwkwardNN(nn.Module):
    def __init__(self, max_depth, input_sz, hidden_sz, output_sz):
        super(AwkwardNN, self).__init__()
        self.max_depth = max_depth
        self.input_sz = input_sz
        self.hidden_sz = hidden_sz
        self.layers = []
        for _ in range(max_depth):
            self.layers.append( nn.Linear(input_sz + hidden_sz, hidden_sz) )
        self.output = nn.Linear(hidden_sz, output_sz)

    def forward(self, input_data, markers, hidden):
        i = 0
        # since we're not iterating over batches
        input_data, markers = input_data[0], markers[0]
        for marker, net_layer in zip(markers, self.layers):
            if marker == 0:
                continue
            for _ in range(marker):
                x = torch.tensor([[input_data[i]]], dtype=torch.float32)
                combined = torch.cat((x, hidden), 1)
                hidden = F.relu(net_layer(combined))
                i += 1
        output = F.log_softmax(self.output(hidden), dim=1)
        return output, hidden

