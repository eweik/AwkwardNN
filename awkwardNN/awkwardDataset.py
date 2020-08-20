import torch
from torch.utils.data import Dataset

from awkwardNN.utils.yaml_utils import get_nested_yaml
from awkwardNN.utils.dataset_utils import get_data_from_tree_dict_list


####################################################################
#  HELPER FUNCTIONS FOR AwkwardDataset and AwkwardDatasetFromYaml  #
####################################################################

def _get_input_size(dataset):
    if len(dataset[0]) > 0:
        return len(dataset[0][0])
    else:
        for i in dataset:
            if len(i) != 0:
                return len(i[0])
    return 0


def _all_are_false(A, B, C):
    if A is False and B is False and C is False:
        return True
    return False


def _get_correct_y(y1, y2, y3):
    # in case on the y's is an empty list because no fields were listed in yaml
    for i in [y1, y2, y3]:
        if i != []:
            return i


def _use_field_true(yaml_dict):
    if 'use' in yaml_dict and yaml_dict['use'] == False:
        return False
    return True

#####################################################################

class AwkwardDataset(Dataset):
    def __init__(self, X, y):
        self.y = [torch.tensor(y)]*len(X) if isinstance(y, int) else torch.tensor(y)
        self._output_size = len(set(y))
        self.X = X
        self._input_size = _get_input_size(X)

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


class AwkwardDatasetFromYaml(Dataset):
    def __init__(self, roottree_dict_list, yaml_dict):
        """"""
        # check if first/top yaml_dict info should be used,
        # n/a for lower nested fields b/c it never reaches them
        self._use_dataset = True
        if 'use' in yaml_dict and yaml_dict['use'] is False:
            self._use_dataset = False
            return

        self._init_fixed_data(roottree_dict_list, yaml_dict['fixed_fields'])
        self._init_nested_data(roottree_dict_list, yaml_dict['jagged_fields'])
        self._init_jagged_data(roottree_dict_list, yaml_dict['jagged_fields'])

        if _all_are_false(self._use_fixed_data, self._use_jagged_data, self._use_nested_data):
            self._use_dataset = False
            return

        self._length = sum([len(tree['roottree']) for tree in roottree_dict_list])
        self._target_size = len(set([tree['target'] for tree in roottree_dict_list]))

    def __len__(self):
        return self._length

    def __getitem__(self, item):
        if self.use_dataset is False:
            return ([], [], []), []

        X_fixed, y_fixed = self._get_fixed_item(item)
        X_jagged, y_jagged = self._get_jagged_item(item)
        X_nested, y_nested = self._get_nested_item(item)
        y = _get_correct_y(y_fixed, y_jagged, y_nested)
        return (X_fixed, X_jagged, X_nested), y

    ########################################################################
    #                     helper functions for init                        #
    ########################################################################

    def _init_fixed_data(self, roottree_dict_list, yaml_fixed_dict):
        X_fixed, y_fixed = get_data_from_tree_dict_list(roottree_dict_list, yaml_fixed_dict)
        self._use_fixed_data = len(X_fixed) != 0 and _use_field_true(yaml_fixed_dict)
        if self._use_fixed_data:
            self._fixed_dataset = AwkwardDataset(X_fixed, y_fixed)

    def _init_nested_data(self, roottree_dict_list, yaml_jagged_dict):
        nested_fields, _ = get_nested_yaml(yaml_jagged_dict)
        self._jagged_input_size = sum([field['embed_dim'] for field in nested_fields])
        self.nested_datasets = []
        for nest in nested_fields:
            dataset = AwkwardDatasetFromYaml(roottree_dict_list, nest)
            self.nested_datasets.append(dataset)
        self._use_nested_data = len(self.nested_datasets) != 0

    def _init_jagged_data(self, roottree_dict_list, yaml_jagged_dict):
        X_jagged, y_jagged = get_data_from_tree_dict_list(roottree_dict_list, yaml_jagged_dict)
        self._use_jagged_data = len(X_jagged) != 0 and _use_field_true(yaml_jagged_dict)
        if self._use_jagged_data:
            self._jagged_dataset = AwkwardDataset(X_jagged, y_jagged)

    ########################################################################
    #                   helper functions for getitem                       #
    ########################################################################

    def _get_fixed_item(self, item):
        if self._use_fixed_data:
            return self._fixed_dataset[item]
        return [], []

    def _get_jagged_item(self, item):
        if self._use_jagged_data:
            return self._jagged_dataset[item]
        return [], []

    def _get_nested_item(self, item):
        X_nested, y_nested = [], []
        for dataset in self.nested_datasets:
            nested_event, y_nested = dataset[item]
            X_nested.append(nested_event)
        return X_nested, y_nested

    ########################################################################
    #                        misc. helper functions                        #
    ########################################################################

    @property
    def target_size(self):
        return self._target_size

    @property
    def fixed_input_size(self):
        return self._fixed_dataset.input_size

    @property
    def jagged_input_size(self):
        # returns the size of the all the subnests being fed into the jagged network.
        # good if all the fields in jagged_fields have subkeys, so can use with an MLP
        # not good if one of those fields is jagged with no subkeys
        # note: will only be called by MLP, will not be called by other double jagged type nets
        return self._jagged_input_size

    @property
    def use_dataset(self):
        return self._use_dataset

    @property
    def use_fixed_data(self):
        return self._use_fixed_data

    @property
    def use_jagged_data(self):
        return self._use_jagged_data

    @property
    def use_nested_data(self):
        return self._use_nested_data

