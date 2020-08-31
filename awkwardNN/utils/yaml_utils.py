import yaml


#########################################################################
# Helper functions to be used for generating default yaml file from rootfile
#########################################################################


def add_bad_key(yaml_dict, key):
    if 'bad_keys' in yaml_dict:
        yaml_dict['bad_keys'].append(key)
    else:
        yaml_dict.update({'bad_keys': [key]})
    return yaml_dict


def remove_empty_subblocks(yaml_dict):
    if yaml_dict['fixed_fields']['fields'] == []:
        del yaml_dict['fixed_fields']
    if yaml_dict['jagged_fields']['fields'] == []:
        del yaml_dict['jagged_fields']
    if yaml_dict['object_fields']['fields'] == []:
        del yaml_dict['object_fields']
    if yaml_dict['nested_fields']['fields'] == []:
        del yaml_dict['nested_fields']
    return yaml_dict


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
# Helper functions related to yaml (called by other modules)
############################################################

def get_nested_yaml(yaml_dict):
    nested_yaml, nested_name = [], []
    for field in yaml_dict['fields']:
        # len()==1 to ignore nested fields that are commented out
        if not isinstance(field, dict):
            raise ValueError("nested field {} is not nested".format(field))
        if len(field) == 1:
            name = list(field.keys())[0]
            yaml = list(field.values())[0]
            nested_yaml.append(yaml)
            nested_name.append(name)
    return nested_yaml, nested_name


def parse_hidden_size_string(hidden_size_string):
    hidden_size_string = hidden_size_string.strip()[1:-1]
    hidden_size_string = hidden_size_string.replace(" ", "")
    hidden_size_string_list = hidden_size_string.split(",")
    hidden_size = [int(x) for x in hidden_size_string_list]
    return hidden_size


