import uproot
import uproot_methods
import awkward
import numpy as np
import torch
from torch.utils.data import Dataset

TYPES = [np.ndarray, awkward.array.jagged.JaggedArray]

# fields that don't work:
# b'Particle.fBits', b'Track.fBits', b'Tower.fBits'
# b'EFlowTrack.fBits', b'EFlowPhoton.fBits', b'EFlowNeutralHadron.fBits'

'''
# keys in test_qcd_1000.root
[b'Event', b'Event_size', b'Particle', b'Particle_size', b'Track',
 b'Track_size', b'Tower', b'Tower_size', b'EFlowTrack', b'EFlowTrack_size',
 b'EFlowPhoton', b'EFlowPhoton_size', b'EFlowNeutralHadron', b'EFlowNeutralHadron_size',
 b'Jet', b'Jet_size', b'Electron', b'Electron_size', b'Photon',
 b'Photon_size', b'Muon', b'Muon_size', b'FatJet', b'FatJet_size',
 b'MissingET', b'MissingET_size', b'ScalarHT', b'ScalarHT_size']
 
 # trial
 ['Particle.E', 'Particle.P[xyz]']
 '''




def get_events_from_tree(tree, col_names=None):
    '''
    :param tree: _TTree_
    :param col_names: _list_ of _str_
        list of column names from which to convert into rows of data
    :param batchsize: _int_
    :return:
    '''
    data = []
    # steps = 2
    # for col_batch in tree.iterate(branches=col_names, entrysteps=steps, namedecode='ascii'):
    #     data.extend(columns2rows(col_batch))
    #     break
    for col_batch in tree.iterate(branches=col_names, namedecode='ascii'):
        data.extend(columns2rows(col_batch))
    return data


def columns2rows(column_dict):
    event_list = []
    for key, column in column_dict.items():
        if isinstance(column, np.ndarray):
            continue
        #print("{} {} {}".format(type(column[0][0]), key, column))
        #print("{} {}".format(len(column[0]), len(column[1])))
        column = check_lorentz_vector(column)
        if len(event_list) == 0:
            event_list = partition_rows(column)
        else:
            event_list = append_axis2(event_list, column)
    return event_list


def partition_rows(field_list):
    """
    :param field_list: _list_ of _list_ of numbers
    :return: _list_ of _list_ of _list_ of numbers

    e.g.
        field_list = [[11 12 13 14 15] [51 52]]
        return b = [[[11] [12] [13] [14] [15]] [[51] [52]]]
    """
    new_field_list = []
    for event in field_list:
        if isinstance(event[0], np.uint32):
            event = event.astype(np.int32)
        elements2lists = [[el] for el in event]
        new_field_list.append(elements2lists)
    return new_field_list


def append_axis2(a, b):
    '''
    append/extend jagged array `b` to jagged array `a` element wise along axis=2
    :param a: _list_ of _list_ of _list_ of numbers
        e.g. list of events, which are lists of particles,
             which are lists of field values
    :param b: _list_ of _list_ of numbers (append)
           OR _list_ of _list_ of _list_ of numbers (extend)
           Unfortunately, no fast way to tell if a field in root is formatted in
           the former or latter way. e.g. 'Jet.Tau[5]' is in the latter
           but have to check the elements to see how they are formatted
    :return: _list_ of _list_ of _list_ of numbers

    e.g. of appending
        a = [[[1] [2] [3]] [[11] [12]]]
        b = [[21 22 23] [31 32]]
        return: [[[1 21] [2 22] [3 23]] [[11 31] [12 32]]]

    e.g. of appending, sometimes array `b` will be longer than array `a`
         by a constant factor (e.g len(b) = 5 * len(a)), so append to `a`
         from `b` the amount of elements that equals the factor
        a = [[[1] [2] [3]] [[11] [12]]]
        b = [[11 12 13 14 15 16] [51 52 53 54]]
        return: [[[1 11 12] [2 13 14] [3 15 16]] [[11 51 52] [12 53 54]]]

    e.g. of extending
        a = [[[1] [2] [3]] [[11] [12]]]
        b = [[[21] [22] [23]] [[31] [32]]]
        return: [[[1 21] [2 22] [3 23]] [[11 31] [12 32]]]
    '''
    assert len(a) == len(b)
    for i, event in enumerate(b):
        if isinstance(event[0], np.uint32):  # torch has issues with np.uint32 type
            event = event.astype(np.int32)
        a[i] = _append_axis1(a[i], event)
    return a


def _append_axis1(a, b):
    assert len(b) % len(a) == 0
    step = int(len(b) / len(a))
    for i, j in zip(range(len(a)), range(0, len(b), step)):
        if type(b[j]) in [np.ndarray, list, uproot.rootio.TRefArray]:
            a[i].extend(b[j])
        else:
            a[i].extend(b[j:j + step])
    return a


def check_lorentz_vector(field):
    '''
    What to do with TLorentzVector object. Currently just return the energy.
    :param field: _list_ of features
    :return: _float_
    '''
    if isinstance(field[0][0], uproot_methods.classes.TLorentzVector.TLorentzVector):
        return field.E
    return field


def get_input_size(data, feature_size_fixed):
    '''
    Determines input size for Awkward-NN. Assume data is triple-nested.
    :param data: _list_ of _list_ of numbers
    :return: _int_
    '''
    if feature_size_fixed:
        #return data[0].shape[-1]
        return len(data[0][0])
        #return len(data[0][0][0])
    return 1


def wrap_particles_in_list(event_list):
    # if field is fixed size
    for event_i in range(len(event_list)):
        for particle_j in range(len(event_list[event_i])):
            event_list[event_i][particle_j] = [event_list[event_i][particle_j]]
    return [torch.tensor(i) for i in event_list]
    #return event_list


class AwkwardDataset(Dataset):
    def __init__(self, X, y, feature_size_fixed=False):
        """
        :param X: _list_ of _list_ of _list_ of {_int_, _float_}
        :param y: _int_ or _list_ of _int_
        :param feature_size_fixed: _bool_
            specifies whether the final nested list (e.g. the list
            of features for a particle) has a fixed size or not
        """
        self.y = [torch.tensor(y)]*len(X) if isinstance(y, int) else torch.tensor(y)
        self._output_size = len(set(self.y))
        #self.X = wrap_particles_in_list(X) if feature_size_fixed else X
        self.X = X
        self.input_size = get_input_size(self.X, feature_size_fixed)


    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        return self.X[item], self.y[item]

    @property
    def output_size(self):
        return self._output_size

    @property
    def input_size(self):
        return self._input_size

    @input_size.setter
    def input_size(self, value):
        self._input_size = value


class AwkwardDatasetFromYaml(Dataset):
    def __init__(self, yamlfile, roottree, y, *, rnn_fields, lstm_fields,
                 gru_fields, deepset_fields):
        """"""
        self.X_rnn = [get_events_from_tree(roottree, col_names=fields) for fields in rnn_fields]
        self.X_lstm = [get_events_from_tree(roottree, col_names=fields) for fields in lstm_fields]
        self.X_gru = [get_events_from_tree(roottree, col_names=fields) for fields in gru_fields]
        self.X_deepset = [get_events_from_tree(roottree, col_names=fields) for fields in deepset_fields]
        self.X = X
        self.input_size = get_input_size(self.X, feature_size_fixed)
        self.y = [torch.tensor(y)] * len(X) if isinstance(y, int) else torch.tensor(y)
        self._output_size = len(set(self.y))


    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        return self.X[item], self.y[item]

    @property
    def output_size(self):
        return self._output_size

    @property
    def input_size(self):
        return self._input_size

    @input_size.setter
    def input_size(self, value):
        self._input_size = value




if __name__ == "__main__":
    tree1 = uproot.open("./data/test_qcd_1000.root")["Delphes"]
    tree2 = uproot.open("./data/test_ttbar_1000.root")["Delphes"]
    deepset_fields = [['Jet.fUniqueID'], ['Jet.PT', 'Jet.Flavor']]
    rnn_fields = [['Jet.Eta', 'Jet.Tau[5]'], ['Jet.Mass', 'Jet.TrimmedP4[5]']]
    #fields = ['Jet.PT', 'Jet.Mass', 'Jet.Tau[5]']
    fields = ['Jet.PT', 'Jet.Flavor', 'Jet.TrimmedP4[5]', 'Jet.Tau[5]']
    #fields = ["Particle*"]
    #fields = ["Jet*"]
    X1 = get_events_from_tree(tree1, fields)
    X2 = get_events_from_tree(tree2, fields)
    y1 = [1] * len(X1)
    y2 = [0] * len(X2)
    print()
    X = X1 + X2
    y = y1 + y2

    print()
    for i in X:
        print(i)
    print()



