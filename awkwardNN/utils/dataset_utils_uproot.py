import numpy as np
import torch
import uproot_methods
import awkwardNN.utils.root_utils_uproot as root_utils
import awkward


def get_data_from_tree_dict_list(roottree_dict_list, sub_yaml_dict):
    X, y = [], []
    for roottree_dict in roottree_dict_list:
        fields = [field for field in sub_yaml_dict['fields'] if isinstance(field, str) and field is not None]
        events = get_events_from_tree(roottree_dict['roottree'], fields)
        X += events
        y += len(events) * [roottree_dict['target']]
    return X, y


######################################################################
#                  get data from rootfiles in uproot                 #
######################################################################

def get_events_from_tree(roottree, col_names):
    '''
    :param roottree: TTree
    :param col_names: list of str
    :return:
    '''
    if len(col_names) == 0:
        return []
    _check_fields_interpretation(roottree, col_names)
    data = []
    # steps = 3
    # for col_dict in roottree.iterate(branches=col_names, entrysteps=steps, namedecode='ascii'):
    #     data.extend(_columns2rows(roottree, col_dict))
    #     break
    for col_dict in roottree.iterate(branches=col_names, namedecode='ascii'):
        data.extend(_columns2rows(roottree, col_dict))
    return data


######################################################################
#    helper functions for organizing data from rootfiles in uproot   #
######################################################################

def _columns2rows(roottree, column_dict):
    key = list(column_dict.keys())[0]
    if root_utils.is_fixed(roottree, key):
        return _columns2rows_fixed(column_dict)
    elif root_utils.is_jagged(roottree, key):
        return _columns2rows_jagged(roottree, column_dict)
    elif root_utils.is_object(roottree, key):
        return _columns2rows_object(roottree, column_dict)
    raise ValueError("Could not identify uproot interpretation of the fields {} "
                     "in roottree {}".format(list(column_dict.keys()), roottree))


def _columns2rows_fixed(column_dict):
    data = np.asarray(list(column_dict.values()), dtype=np.float32)
    event_list = torch.unsqueeze(torch.tensor(data[0]), dim=1)
    for field in data[1:]:
        field = torch.unsqueeze(torch.tensor(field), dim=1)
        event_list = torch.cat((event_list, field), dim=1)
    return event_list.unsqueeze(1)


def _columns2rows_jagged(roottree, column_dict):
    keys = list(column_dict.keys())
    data = list(column_dict.values())
    event_list = [np.expand_dims(event, axis=1) for event in data[0]]
    for field_name, field_data in zip(keys[1:], data[1:]):
        field_data = _check_lorentz_vector(roottree, field_name, field_data)
        field_data = _check_tref_vector(roottree, field_name, field_data)
        event_list = _append_jagged(event_list, field_data)
    return _convert_to_float32(event_list)


def _append_jagged(event_list, field):
    for i, field_i in enumerate(field):
        if len(field_i) != 0 and len(event_list[i]) != 0:
            if len(field.content.shape) == 1:
                field_i = np.split(field_i, len(event_list[i]))
            event_list[i] = np.append(event_list[i], field_i, axis=1)
        elif len(field_i) != 0:
            event_list[i] = np.expand_dims(field_i, axis=1)
    return event_list


def _convert_to_float32(jagged_events):
    new_jagged_events = []
    for event in jagged_events:
        if event.size == 0:
            new_jagged_events.append(torch.tensor([]))
        else:
            event = torch.tensor(event, dtype=torch.float32)
            event = event.unsqueeze(1)
            new_jagged_events.append(event)
    return new_jagged_events


def _columns2rows_object(roottree, column_dict):
    keys = list(column_dict.keys())
    data = list(column_dict.values())
    event_list = data[0]
    for field_name, field_data in zip(keys[1:], data[1:]):
        # field_data = _check_lorentz_vector(roottree, field_name, field_data)
        event_list = _append_object(event_list, field_data)
    return _expand_object_dim(event_list)


def _append_object(event_list, field):
    new_event_list = []
    for i in range(len(event_list)):
        event_list_i = awkward.fromiter(event_list[i])
        field_i = awkward.fromiter(field[i])
        new_event_list.append(awkward.concatenate([event_list_i, field_i], axis=1).tolist())
    return new_event_list


def _expand_object_dim(event_list):
    new_event_list = []
    for event in event_list:
        event = [torch.tensor(i, dtype=torch.float32) for i in event]
        new_event = [i.view(-1, 1, 1) for i in event]
        new_event_list.append(new_event)
    return new_event_list


# TODO: What do I do about Lorentz vector objects in rootfile?
def _check_lorentz_vector(roottree, field_name, field_data):
    if root_utils.is_tlorentzvector(roottree, field_name):
        return field_data.E
    return field_data


def _check_tref_vector(roottree, field_name, field_data):
    if root_utils.is_tref(roottree, field_name):
        return field_data.id
    return field_data


def _check_uint32(event):
    # torch has issues with np.uint32 type
    if _iterable(event) and len(event) == 0:
        return event
    sample_event = event[0] if _iterable(event) else event
    if isinstance(sample_event, np.uint32):
        event = event.astype(np.int32)
    return event


def _iterable(obj):
    try:
        iter(obj)
    except Exception:
        return False
    else:
        return True


def _check_fields_interpretation(roottree, field_list):
    interps = set([type(roottree[key].interpretation) for key in field_list])
    if len(interps) > 1:
        raise ValueError("All fields in same chunk must have the same uproot interpretation.\n"
                         "The fields {} have interpretations {}".format(field_list, interps))
