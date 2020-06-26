# preprocessAwkwardData.py
# Preprocess Awkward data structure to make it readable for RNN
#

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from AwkwardNN.createAwkwardData import generate_data_target
import re


def get_max_depth(awkward_data):
    awk_type_string = str(awkward_data.type)
    match = re.findall(r'->', awk_type_string)
    return len(match) - 1


def separate_awkward_nests_in_event(awkward_data, max_depth):
    separated = [[] for _ in range(max_depth)]
    this_layer, next_layer = awkward_data, []
    for layer in range(max_depth):
        for i in this_layer:
            if isinstance(i, np.float64) or isinstance(i, float):
                separated[layer].append(i)
            else:
                next_layer.extend(i)
        this_layer, next_layer = next_layer, []
    return separated


def flatten_and_get_markers_in_event(denested_data):
    flattened = []
    markers = []
    for layer in denested_data:
        flattened.extend(layer)
        markers.append(len(layer))
    return torch.tensor(flattened), torch.tensor(markers)


class AwkwardDataset(Dataset):
    def __init__(self, num_events, prob_nest, prob_signal, prob_noise, max_len, max_depth):
        #self.awkd_data = awkward.load(awkd_file)
        self.awkd_data, self.target = generate_data_target(num_events, prob_nest, prob_signal,
                                                           prob_noise, max_len, max_depth)
        self.target = torch.tensor(self.target)
        self.max_depth = get_max_depth(self.awkd_data)
        self.flattened = []
        self.markers = []
        for i in self.awkd_data:
            separated_event = separate_awkward_nests_in_event(i, self.max_depth)
            f, m = flatten_and_get_markers_in_event(separated_event)
            self.flattened.append(f)
            self.markers.append(m)

    def __len__(self):
        return len(self.awkd_data)

    def __getitem__(self, item):
        return self.flattened[item], self.markers[item], self.target[item]


def get_dataloader(dataset_size, batch_size, prob_nest, prob_signal, prob_noise, max_len, max_depth):
    dataset = AwkwardDataset(dataset_size, prob_nest, prob_signal, prob_noise, max_len, max_depth)
    return DataLoader(dataset, batch_size=batch_size)
