import numpy as np
import uproot4
import awkward1
import uproot_methods
import awkwardNN.utils.root_utils_uproot4 as root_utils


def get_data_from_tree_dict_list(roottree_dict_list, sub_yaml_dict):
    X, y = [], []
    for roottree_dict in roottree_dict_list:
        fields = [field for field in sub_yaml_dict['fields'] if isinstance(field, str) and field is not None]
        events = get_events_from_tree(roottree_dict['roottree'], fields)
        X += events
        y += len(events) * [roottree_dict['target']]
    return X, y

######################################################################
#                   get data from rootfiles in uproot4               #
######################################################################

def get_events_from_tree(roottree, col_names):
    '''
    '''
    _check_fields_interpretation(roottree, col_names)
    data = []
    for col_batch in roottree.iterate(expressions=col_names, entry_stop=2):
        data.extend(_columns2rows(col_batch, col_names, roottree))
        break
    # for col_batch in roottree.iterate(expressions=col_names):
    #     data.extend(_columns2rowsd(col_batch, col_names))
    return data


######################################################################
#    helper functions for organizing data from rootfiles in uproot4  #
######################################################################

def _columns2rows(data, fields, roottree):
    key = fields[0]
    if root_utils.is_fixed(roottree, key):
        return _columns2rows_fixed(data, fields)
    if root_utils.is_jagged(roottree, key):
        return _columns2rows_jagged(data, fields)
    elif root_utils.is_object(roottree, key):
        return _columns2rows_object(data, fields)
    raise ValueError("Could not identify uproot interpretation of the fields {} "
                     "in roottree {}".format(fields, roottree))


def _columns2rows_fixed(event_dict_list, fields):
    event_list = []
    for event_dict in event_dict_list:
        event = [event_dict[i] for i in fields]
        event_list.append(event)
    return event_list


def _columns2rows_jagged(event_dict_list, fields):
    event_list = []
    for event_dict in event_dict_list:
        events = _transpose_jagged(event_dict, fields)
        event_list.append(events)
    return event_list


def _transpose_jagged(event_dict, fields):
    event = np.expand_dims(event_dict[fields[0]].tolist(), axis=1)
    for f in fields[1:]:
        field_i = awkward1.to_numpy(event_dict[f])
        event = _append_jagged(event, field_i)
    return event


def _append_jagged(a, b):
    if len(b) != 0 and len(a) != 0:
        if len(b.shape) == 1:
            b = np.split(b, len(a))
        a = np.append(a, b, axis=1)
    elif len(b) != 0:
        a = np.expand_dims(b, axis=1)
    return a


# TODO: this seems to return something different in uproot4 than in uproot, think the latter correct
def _columns2rows_object(event_dict_list, fields):
    event_list = []
    for event_dict in event_dict_list:
        # print(event_dict.layout)
        for i in event_dict[fields[0]][0].tolist():
            print("{} {}".format(i, event_dict[fields[0]][0].tolist()[i]))
        event = event_dict[fields[0]][0].tolist()['refs']
        for f in fields[1:]:
            event.extend(event_dict[f][0].tolist()['refs'])
        event_list.append(event)
    return event_list



def _append_object(event_list, field):
    for i in range(len(event_list)):
        for j in range(len(event_list[i])):
            event_list[i][j] = np.append(event_list[i][j], field[i][j])
    return event_list



def _transpose_column(data, fields):
    first_field = awkward1.to_numpy(data[fields[0]])
    event_T = np.expand_dims(first_field, axis=1)
    for f in fields[1:]:
        column = _check_lorentz_vector(data[f])
        column = awkward1.to_numpy(column)
        event_T = _append_column(event_T, column)
    return event_T


def _append_column(event_T, column):
    assert column.size % len(event_T) == 0
    column = _check_uint32(column)
    split_col = np.split(column, len(event_T))
    return np.append(event_T, split_col, axis=1)


def _check_uint32(data):
    if data.dtype == 'uint32':
        return data.astype(np.int32)
    return data


# TODO: What do I do about Lorentz vector objects in rootfile?
def _check_lorentz_vector(data):
    if '__record__' in awkward1.type(data).type.parameters and \
            awkward1.type(data).type.parameters['__record__'] == "TLorentzVector":
        return data.fE
    return data


def _iterable(obj):
    try:
        iter(obj)
    except Exception:
        return False
    else:
        return True


def _check_fields_interpretation(roottree, field_list):
    interps = set([type(roottree[key].interpretation) for key in field_list])
    if len(interps) != 1:
        raise ValueError("All fields in same chunk must have the same uproot interpretation.\n"
                         "The fields {} have interpretations {}".format(field_list, interps))
