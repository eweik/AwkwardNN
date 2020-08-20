import numpy as np
import uproot
import uproot_methods


def get_data_from_tree_dict_list(roottree_dict_list, sub_yaml_dict):
    X, y = [], []
    for roottree_dict in roottree_dict_list:
        fields = [field for field in sub_yaml_dict['fields'] if isinstance(field, str) and field is not None]
        events = get_events_from_tree(roottree_dict['roottree'], fields)
        X += events
        y += len(events) * [roottree_dict['target']]
    return X, y


def get_events_from_tree(roottree, col_names=None):
    '''
    :param roottree: TTree
    :param col_names: list of str
    :return:
    '''
    data = []
    # steps = 2
    # for col_batch in roottree.iterate(branches=col_names, entrysteps=steps, namedecode='ascii'):
    #     data.extend(_columns2rows(col_batch))
    #     break
    for col_batch in roottree.iterate(branches=col_names, namedecode='ascii'):
        data.extend(_columns2rows(col_batch))
    return data


###############################################################
#     helper functions for organizing data from rootfiles     #
###############################################################

def _columns2rows(column_dict):
    event_list = []
    for key, col in column_dict.items():
        col = _check_lorentz_vector(col)
        event_list = _list_elements(col) if event_list == [] else _append_axis2(event_list, col)
    return event_list


def _list_elements(field_list):
    """
    :param field_list: list of list of numbers
    :return: list of list of list of numbers

    e.g.
        field_list = [[11 12 13 14 15] [51 52]]
        return b = [[[11] [12] [13] [14] [15]] [[51] [52]]]
    """
    new_field_list = []
    for event in field_list:
        if not _iterable(event):
            new_field_list.append([[event]])
        elif len(event) == 0:
            new_field_list.append([])
        elif _iterable(event[0]):
            new_field_list.append(event)
        else:
            if isinstance(event[0], np.uint32):
                event = event.astype(np.int32)
            elements2lists = [[el] for el in event]
            new_field_list.append(elements2lists)
    return new_field_list


def _append_axis2(a, b):
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
    for i, event in enumerate(b):
        event = _check_uint32(event)
        if not _iterable(event):
            event = [event]
        if len(event) != 0:
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


# TODO: What do I do about Lorentz vector objects in rootfile?
def _check_lorentz_vector(field):
    try:
        if isinstance(field[0][0], uproot_methods.classes.TLorentzVector.TLorentzVector):
            return field.E
    except:
        pass
    return field


def _iterable(obj):
    try:
        iter(obj)
    except Exception:
        return False
    else:
        return True


def _check_uint32(event):
    # torch has issues with np.uint32 type
    if _iterable(event) and len(event) == 0:
        return event
    sample_event = event[0] if _iterable(event) else event
    if isinstance(sample_event, np.uint32):
        event = event.astype(np.int32)
    return event