import uproot
import uproot_methods
import awkward
import numpy as np
import torch
from torch.utils.data import Dataset

from awkwardNN.utils.root_utils import get_roottree
from awkwardNN.utils.yaml_utils import get_yaml_dict_list

TYPES = [np.ndarray, awkward.array.jagged.JaggedArray]

# fields that don't work:
# b'Particle.fBits', b'Track.fBits', b'Tower.fBits'
# b'EFlowTrack.fBits', b'EFlowPhoton.fBits', b'EFlowNeutralHadron.fBits'


def get_events_from_tree(roottree, col_names=None):
    '''
    :param tree: _TTree_
    :param col_names: _list_ of _str_
        list of column names from which to convert into rows of data
    :param batchsize: _int_
    :return:
    '''
    data = []
    # steps = 2
    # for col_batch in roottree.iterate(branches=col_names, entrysteps=steps, namedecode='ascii'):
    #     data.extend(columns2rows(col_batch))
    #     break
    for col_batch in roottree.iterate(branches=col_names, namedecode='ascii'):
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
    :param field: list
    :return: float
    '''
    try:
        if isinstance(field[0][0], uproot_methods.classes.TLorentzVector.TLorentzVector):
            return field.E
    except:
        pass
    return field


def get_input_size(data):
    '''
    Determines input size for Awkward-NN. Assume data is triple-nested.
    :param data: list of list
    :return: int
    '''
    return len(data[0][0])


def wrap_fields_in_list(event_list):
    # if field is fixed size
    for event_i in range(len(event_list)):
        for particle_j in range(len(event_list[event_i])):
            event_list[event_i][particle_j] = [event_list[event_i][particle_j]]
    return [torch.tensor(i) for i in event_list]
    #return event_list


class AwkwardDataset(Dataset):
    def __init__(self, X, y):
        """
        :param X: list
        :param y: int
        :param feature_size_fixed: bool
            specifies whether the final nested list (e.g. the list
            of features for a particle) has a fixed size or not
        """
        self.y = [torch.tensor(y)]*len(X) if isinstance(y, int) else torch.tensor(y)
        self._output_size = len(set(y))
        # self.X = wrap_fields_in_list(X) if feature_size_fixed else X
        self.X = X
        self.input_size = get_input_size(self.X)

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
    def __init__(self, roottree_dict_list, yaml_dict):
        """"""
        self._init_fixed_data(roottree_dict_list, yaml_dict['fixed_fields'])
        self._init_jagged_data(roottree_dict_list, yaml_dict['jagged_fields'])
        self._init_nested_data(roottree_dict_list, yaml_dict['jagged_fields'])

        self._fixed_output = yaml_dict['fixed_fields']['embed_dim']
        self._jagged_output = yaml_dict['jagged_fields']['embed_dim']

    def __len__(self):
        # total number of events is same as total number in one of the datasets
        return len(self._fixed_dataset[0])

    def __getitem__(self, item):
        X_fixed, y_fixed = self._get_fixed_data(item)
        X_jagged, y_jagged = self._get_jagged_data(item)
        X_nested, y_nested = self._get_nested_data(item)

        # in case on the y's is an empty list because no fields were listed in yaml
        for i in [y_fixed, y_jagged, y_nested]:
            if isinstance(i, int):
                y = i
        return (X_fixed, X_jagged, X_nested), y

    def _init_fixed_data(self, roottree_dict_list, fixed_dict):
        X_fixed, y_fixed = [], []
        for roottree_dict in roottree_dict_list:
            fixed_events = get_events_from_tree(roottree_dict['roottree'], fixed_dict['fields'])
            X_fixed += fixed_events
            y_fixed += len(fixed_events) * [roottree_dict['target']]
        self._fixed_dataset = AwkwardDataset(X_fixed, y_fixed)

    def _init_jagged_data(self, roottree_dict_list, jagged_dict):
        jagged_fields = [field for field in jagged_dict['fields'] if isinstance(field, str)]
        X_jagged, y_jagged = [], []
        for roottree_dict in roottree_dict_list:
            jagged_events = get_events_from_tree(roottree_dict['roottree'], jagged_fields)
            X_jagged += jagged_events
            y_jagged += len(jagged_events) * [roottree_dict['target']]
        self._jagged_dataset = AwkwardDataset(X_jagged, y_jagged)

    def _init_nested_data(self, roottree_dict_list, jagged_dict):
        nested_fields = [list(field.values())[0] for field in jagged_dict['fields'] if isinstance(field, dict)]
        self._jagged_input_size = sum([field['embed_dim'] for field in nested_fields])
        self.nested_datasets = []
        for nest in nested_fields:
            dataset = AwkwardDatasetFromYaml(roottree_dict_list, nest)
            self.nested_datasets.append(dataset)

    def _get_fixed_data(self, item):
        if self._fixed_dataset != []:
            X_fixed, y_fixed = self._fixed_dataset[item]
        else:
            X_fixed, y_fixed = [], []
        return X_fixed, y_fixed

    def _get_jagged_data(self, item):
        if self._jagged_dataset != []:
            X_jagged, y_jagged = self._fixed_dataset[item]
        else:
            X_jagged, y_jagged = [], []
        return X_jagged, y_jagged

    def _get_nested_data(self, item):
        X_nested = []
        for dataset in self._nested_datasets:
            nested_event, y_nested = dataset[item]
            X_nested.append(nested_event)
        return X_nested, y_nested

    @property
    def output_size(self):
        return self._fixed_output, self._jagged_output

    @property
    def fixed_input_size(self):
        return self._fixed_dataset.input_size

    @property
    def jagged_input_size(self):
        # returns the size of the all the subnests being fed into the jagged network.
        # works if all the fields in jagged_fields have subkeys.
        # not good if one of those fields is jagged with no subkeys
        return self._jagged_input_size




