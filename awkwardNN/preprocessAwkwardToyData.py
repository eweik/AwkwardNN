# preprocessAwkwardToyData.py
# Preprocess Awkward data structure to make it readable for RNN
#

import torch
import awkward
import numpy as np
from torch.utils.data import Dataset, DataLoader
from awkwardNN.createAwkwardToyData import generate_data_target
import re


NESTED_TYPES = [awkward.array.jagged.JaggedArray,
                awkward.array.union.UnionArray,
                np.ndarray, list]


def get_max_depth(awkward_data):
    awk_type_string = str(awkward_data.type)
    match = re.findall(r'->', awk_type_string)
    return len(match) - 1


def is_nest(element):
    for i in NESTED_TYPES:
        if isinstance(element, i):
            return True
    return False


def get_data_from_layer(data_layer):
    elements_in_layer, next_layer = [], []
    for i in data_layer:
        if not is_nest(i):
            elements_in_layer.append(i)
        else:
            next_layer.extend(i)
    return elements_in_layer, next_layer


def separate_awkward_nests_in_event(awkward_data, max_depth):
    '''
    :param awkward_data:
    :param max_depth:
    :return: list of lists, where the i'th list contains all non-nested elements
             from the i'th nested layer of the awkward-array input.
    '''
    separated = []
    for layer in range(max_depth):
        elements_in_layer, awkward_data = get_data_from_layer(awkward_data)
        separated.append(elements_in_layer)
    return separated


def separate_awkward_nests_in_dataset(awkward_dataset, max_depth):
    separated_dataset = []
    for i in awkward_dataset:
        separated_event = separate_awkward_nests_in_event(i, max_depth)
        separated_dataset.append(separated_event)
    return separated_dataset


def flatten_and_get_markers_in_event(denested_data):
    flattened_event = []
    markers_event = []
    for layer in denested_data:
        flattened_event.extend(layer)
        markers_event.append(len(layer))
    return flattened_event, markers_event


def flatten_and_get_markers_in_dataset(denested_dataset):
    flattened_dataset = []
    marker_dataset = []
    for event in denested_dataset:
        f, m = flatten_and_get_markers_in_event(event)
        flattened_dataset.append(f)
        marker_dataset.append(m)
    return flattened_dataset, marker_dataset


def deflatten_event(flattened, markers):
    '''
    :param flattened:
    :param markers: list of indices indicating when
    :return: list of lists, where the i'th list contains all non-nested elements
             from the i'th nested layer of the awkward-array input.
             (same as separate_awkward_nests_in_event(...), but different input)
    '''
    separated = []
    j = 0
    for i in markers:
        next_level = flattened[j:i+j]
        separated.append(next_level)
        j += i
    return separated


def deflatten_dataset(flattened_dataset, marker_dataset):
    separated_dataset = []
    for f, m in zip(flattened_dataset, marker_dataset):
        sep_event = deflatten_event(f, m)
        separated_dataset.append(sep_event)
    return separated_dataset


class AwkwardDataset(Dataset):
    def __init__(self, X, y, mode):
        self.awkd_data = X
        self.target = y
        self.mode = mode
        self.output_size = len(set(self.target))
        self.target = torch.tensor(self.target)
        self.max_depth = get_max_depth(self.awkd_data)
        self.separated_dataset = separate_awkward_nests_in_dataset(self.awkd_data, self.max_depth)
        self.flattened, self.markers = flatten_and_get_markers_in_dataset(self.separated_dataset)

    def __len__(self):
        return len(self.awkd_data)

    def __getitem__(self, item):
        if self.target.shape[0] == 0:
            return self.flattened[item], self.markers[item]
        if self.mode == 'deepset':
            return self.separated_dataset[item], self.markers[item], self.target[item]
        return self.flattened[item], self.markers[item], self.target[item]

    def get_max_depth(self):
        return self.max_depth

    def get_output_size(self):
        return self.output_size

