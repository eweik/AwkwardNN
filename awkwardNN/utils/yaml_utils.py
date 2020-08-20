import yaml
from awkwardNN.utils.root_utils import get_roottree


BADKEYS = ['Particle.fBits', 'Track.fBits', 'Tower.fBits',
           'EFlowTrack.fBits', 'EFlowPhoton.fBits', 'EFlowNeutralHadron.fBits']


#########################################################################
# Functions to be used for generating default yaml file from rootfile
#########################################################################
def get_default_yaml_dict_from_rootfile(rootfile, embed_dim, mode, hidden_sizes, nonlinearity):
    roottree = get_roottree(rootfile)
    yaml_block = _init_awkwardNN_yaml_block(embed_dim, mode, hidden_sizes, nonlinearity)
    for k in roottree.keys():
        yaml_block = _add_key_to_yaml(yaml_block, k.decode("utf-8"), roottree,
                                      embed_dim, mode, hidden_sizes, nonlinearity)
    yaml_dict = {'event': yaml_block}
    return yaml_dict


#########################################################################
# Helper functions to be used for generating default yaml file from rootfile
#########################################################################


def _add_key_to_yaml(yaml_dict, key, roottree,
                     embed_dim, mode, hidden_sizes, nonlinearity):
    if _has_subkeys(roottree, key):
        new_yaml_dict = _init_awkwardNN_yaml_block(embed_dim, mode, hidden_sizes, nonlinearity)
        for k in roottree[key].keys():
            if k.decode("utf-8") in BADKEYS:
                continue
            new_yaml_dict = _add_key_to_yaml(new_yaml_dict, k.decode("utf-8"), roottree,
                                             embed_dim, mode, hidden_sizes, nonlinearity)
        yaml_dict['jagged_fields']['fields'].append({key: new_yaml_dict})

    elif _is_nested_and_jagged(roottree, key):
        # empty dict value for new key indicates it can be used, as opposed to
        # non-empty dict with key-value pair: {'use': False}, which user can specify
        yaml_dict['jagged_fields']['fields'].append(key)

    else:
        yaml_dict['fixed_fields']['fields'].append(key)

    return yaml_dict


def _has_subkeys(roottree, key):
    return roottree[key].keys() != []


def _is_nested_and_jagged(roottree, key):
    for event in roottree[key].array():
        try:
            sizes = set([len(i) for i in event])
            if len(sizes) > 1:
                return True
        except: # not nested
            return False
    return False


def _init_awkwardNN_yaml_block(embed_dim, mode, hidden_sizes, nonlinearity):
    subfield_block_a = {'embed_dim': embed_dim, 'mode': mode, 'fields': []}
    subfield_block_b = {'embed_dim': embed_dim, 'mode': mode, 'fields': []}
    yaml_block = {'embed_dim': embed_dim,
                   'mode': mode,
                   'fixed_fields': subfield_block_a,
                  'jagged_fields': subfield_block_b}
    if mode == 'mlp':
        yaml_block.update({'hidden_sizes': hidden_sizes, 'nonlinearity': nonlinearity})
    return yaml_block


############################################################
# Functions for opening and saving yaml files from filename
############################################################
def get_yaml_dict_list(yaml_filename):
    """"""
    with open(yaml_filename, 'r') as file:
        awkwardNN_list = yaml.load(file, Loader=yaml.FullLoader)
    return awkwardNN_list


def save_yaml_model(yaml_dict_list, saveto):
    with open(saveto, 'w') as file:
        yaml.dump(yaml_dict_list, file, sort_keys=False)


############################################################
# Helper functions related to yaml (called by other methods)
############################################################

def get_nested_yaml(jagged_yaml_dict):
    nested_yaml, nested_name = [], []
    for field in jagged_yaml_dict['fields']:
        # len()==1 to ignore nested fields that are commented out
        if isinstance(field, dict) and len(field) == 1:
            name = list(field.keys())[0]
            yaml = list(field.values())[0]
            if 'use' in yaml and yaml['use'] is False:
                continue
            nested_yaml.append(yaml)
            nested_name.append(name)
    return nested_yaml, nested_name


def parse_hidden_size_string(hidden_size_string):
    hidden_size_string = hidden_size_string[1:-1]
    hidden_size_string = hidden_size_string.replace(" ", "")
    hidden_size_string_list = hidden_size_string.split(",")
    hidden_size = [int(x) for x in hidden_size_string_list]
    return hidden_size


