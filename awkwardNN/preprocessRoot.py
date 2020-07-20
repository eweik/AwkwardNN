import uproot
import uproot_methods
import awkward
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split

TYPES = [np.ndarray, awkward.array.jagged.JaggedArray]
#         awkward.array.objects.JaggedArrayMethods,
#         awkward.array.objects.ObjectArray]

# fields that don't work:
# b'Particle.fBits', b'Track.fBits', b'Tower.fBits'
# b'EFlowTrack.fBits', b'EFlowPhoton.fBits', b'EFlowNeutralHadron.fBits'

'''
# all keys in test_qcd_1000.root
[b'Event', b'Event_size', b'Particle', b'Particle_size', b'Track',
 b'Track_size', b'Tower', b'Tower_size', b'EFlowTrack', b'EFlowTrack_size',
 b'EFlowPhoton', b'EFlowPhoton_size', b'EFlowNeutralHadron', b'EFlowNeutralHadron_size',
 b'Jet', b'Jet_size', b'Electron', b'Electron_size', b'Photon',
 b'Photon_size', b'Muon', b'Muon_size', b'FatJet', b'FatJet_size',
 b'MissingET', b'MissingET_size', b'ScalarHT', b'ScalarHT_size']
 
 # trial
 ['Particle.E', 'Particle.P[xyz]']
 '''


def get_events(tree, col_names=None):
    '''

    :param tree: _TTree_
    :param col_names: _list_ of _str_
        list of column names from which to convert into rows of data
    :param batchsize: _int_
    :return:
    '''
    data = []
    steps = None
    for col_batch in tree.iterate(branches=col_names, entrysteps=steps, namedecode='ascii'):
        data.extend(columns2rows(col_batch))
        #break
    return data


def columns2rows(column_dict):
    row_list = []
    for key, column in column_dict.items():
        #print("{} {}".format(key, column))
        if isinstance(column, np.ndarray):
            continue
        #print("{} {}".format(len(column[0]), len(column[1])))
        column = check_lorentz_vector(column)
        if len(row_list) == 0:
            row_list = partition_rows(column)
        else:
            row_list = append_axis2(row_list, column)
    return row_list


def partition_rows(field_list):
    new_field_list = []
    for event in field_list:
        if isinstance(event[0], np.uint32):
            event = event.astype(np.int32)
        elements2lists = [[el] for el in event]
        new_field_list.append(elements2lists)
    return new_field_list


def append_axis2(a, b, flatten=True):
    '''
    append/extend jagged array `b` to jagged array `a` element wise along axis=2
    :param a: _awkward-array_
    :param b: _awkward-array_
    :param flatten: _bool_
        only useful if b elements are lists
        if true (default), extend corresponding `b` lists onto `a` list
        if false, append corresponding `b` lists onto `a` list
    :return: _awkward-array_

    e.g. of appending
        a = [[[1] [2] [3]] [[11] [12]]]
        b = [[21 22 23] [31 32]]
        return: [[[1 21] [2 22] [3 23]] [[11 31] [12 32]]]

    e.g. of extending
        a = [[[1] [2] [3]] [[11] [12]]]
        b = [[[21] [22] [23]] [[31] [32]]]
        return: [[[1 21] [2 22] [3 23]] [[11 31] [12 32]]]
    '''
    for a_i, b_i in zip(a, b):
        step = int(len(b_i) / len(a_i))
        if isinstance(b_i[0], np.uint32):
            b_i = b_i.astype(np.int32)
        for j1, j2 in zip(range(len(a_i)), range(0, len(b_i), step)):
            if flatten and type(b_i[j2]) in [np.ndarray, list, uproot.rootio.TRefArray]:
                a_i[j1].extend(b_i[j2])
            else:
                a_i[j1].extend(b_i[j2:j2 + step])
    return a


def check_lorentz_vector(field):
    if isinstance(field[0][0], uproot_methods.classes.TLorentzVector.TLorentzVector):
        return field.E
    return field


def get_input_size(data):
    # calculate input size (# of fields per particle)
    while isinstance(data[0], list):
        data = data[0]
    return len(data)


class AwkwardDataset(Dataset):
    def __init__(self, X, y):
        self.y = [torch.tensor(y)]*len(X) if isinstance(y, int) else torch.tensor(y)
        self._output_size = len(set(self.y))
        self._input_size = get_input_size(X)
        self.X = X


    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        #if self.X.shape[0] == 0:
        #    return self.X[item]
        return self.X[item], self.y[item]

    @property
    def output_size(self):
        return self._output_size

    @property
    def input_size(self):
        return self._input_size


if __name__ == "__main__":
    tree1 = uproot.open("./data/test_qcd_1000.root")["Delphes"]
    tree2 = uproot.open("./data/test_ttbar_1000.root")["Delphes"]
    deepset_fields = [['Jet.fUniqueID'], ['Jet.PT', 'Jet.Flavor']]
    rnn_fields = [['Jet.Eta', 'Jet.Tau[5]'], ['Jet.Mass', 'Jet.TrimmedP4[5]']]
    #fields = ['Jet.Mass', 'Jet.PT', 'Jet.TrimmedP4[5]']
    #fields = ["Particle*"]
    fields = ["Jet*"]
    X1 = get_events(tree1, fields)
    X2 = get_events(tree2, fields)
    y1 = [1] * len(X1)
    y2 = [0] * len(X2)
    print()
    for i in X1:
        print("{} {} {}".format(len(i[0]), i, type(i)))
    print()
    dataset1 = AwkwardDataset(X1, 1)
    print("input size = {}".format(dataset1.input_size))

    print()
    for i in X2:
        print("{} {} {}".format(len(i[0]), i, type(i)))
    print()
    dataset2 = AwkwardDataset(X2, 1)
    print("input size = {}".format(dataset2.input_size))

    print()
    X = X1 + X2
    y = y1 + y2
    print()
    for i, j in zip(X, y):
        print("{} {} {}".format(j, i, type(i)))
    print()

    print()
    print("Random")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    print()
    print("trainset:")
    for x, y in zip(X_train, y_train):
        print("{}, {}".format(x, y))

    print()
    print("validset:")
    for x, y in zip(X_test, y_test):
        print("{}, {}".format(x, y))

    print()
    print(len(X[0]))
    print(len(X[0][0]))
    print()
    for i in range(len(X)):
        for j in range(len(X[i])):
            print(len(X[i][j]))
