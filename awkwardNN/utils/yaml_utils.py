import yaml
from awkwardNN.utils.root_utils import get_roottree
import collections

# b'Particle.fBits', b'Track.fBits', b'Tower.fBits'
# b'EFlowTrack.fBits', b'EFlowPhoton.fBits', b'EFlowNeutralHadron.fBits'
BADKEYS = ['Particle.fBits', 'Track.fBits', 'Tower.fBits',
           'EFlowTrack.fBits', 'EFlowPhoton.fBits', 'EFlowNeutralHadron.fBits']


def get_yaml_dict_list(yaml_filename):
    """"""
    with open(yaml_filename, 'r') as file:
        awkwardNN_list = yaml.load(file, Loader=yaml.FullLoader)
    return awkwardNN_list


def get_default_yaml_dict_from_rootfile(rootfile):
    roottree = get_roottree(rootfile)
    yaml_block = init_awkwardNN_yaml_block()
    for k in roottree.keys():
        yaml_block = add_key_to_yaml(yaml_block, k.decode("utf-8"), roottree)
    yaml_dict = {'event': yaml_block}
    return yaml_dict


def add_key_to_yaml(yaml_dict, key, roottree):
    if has_subkeys(roottree, key):
        new_yaml_dict = init_awkwardNN_yaml_block()
        for k in roottree[key].keys():
            if k.decode("utf-8") in BADKEYS:
                continue
            new_yaml_dict = add_key_to_yaml(new_yaml_dict, k.decode("utf-8"), roottree)
        yaml_dict['jagged_fields']['fields'].append({key: new_yaml_dict})

    elif is_nested_and_jagged(roottree, key):
        # empty dict value for new key indicates it can be used, as opposed to
        # non-empty dict with key-value pair: {'use': False}, which user can specify
        yaml_dict['jagged_fields']['fields'].append(key)

    else:
        yaml_dict['fixed_fields']['fields'].append(key)

    return yaml_dict


def has_subkeys(roottree, key):
    return roottree[key].keys() != []


def is_nested_and_jagged(roottree, key):
    for event in roottree[key].array():
        try:
            sizes = set([len(i) for i in event])
            if len(sizes) > 1:
                return True
        except: # not nested
            return False
    return False


def init_awkwardNN_yaml_block():
    subfield_block_a = {'embed_dim': 32, 'mode': 'vanilla_rnn', 'fields': []}
    subfield_block_b = {'embed_dim': 32, 'mode': 'vanilla_rnn', 'fields': []}
    return {'embed_dim': 32,
            'mode': 'mlp',
            'hidden_sizes': "(30, 30)",
            'nonlinearity': 'relu',
            'fixed_fields': subfield_block_a,
            'jagged_fields': subfield_block_b}


def save_yaml_model(yaml_dict_list, saveto):
    with open(saveto, 'w') as file:
        yaml.dump(yaml_dict_list, file, sort_keys=False)


if __name__ == "__main__":
    rootfile = "./data/test_qcd_1000.root"
    yaml_dict = get_default_yaml_dict_from_rootfile(rootfile)
    saveto = "test_qcd_1000_default.yaml"
    save_yaml_model([yaml_dict], saveto)


