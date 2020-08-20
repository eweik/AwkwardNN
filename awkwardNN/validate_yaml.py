from awkwardNN.utils.yaml_utils import parse_hidden_size_string

MODES = ['gru', 'lstm', 'vanilla_rnn', 'deepset', 'mlp', 'identity']
ACTIVATIONS = ['tanh', 'relu']


def validate_use_in_first_yaml(yaml_dict):
    _validate_use(yaml_dict)
    if 'use' in yaml_dict and yaml_dict['use'] is False:
        raise ValueError("\'use\' field in outermost network may not be False. C'mon, what's "
                         "the point of doing this if the topmost \'use\' is False smh")


def validate_yaml(yaml_dict):
    _validate_embed_dim(yaml_dict)
    _validate_mode(yaml_dict)
    _validate_nonlinearity(yaml_dict)
    _validate_hidden_size(yaml_dict)
    _validate_use(yaml_dict)
    _validate_fixed_fields(yaml_dict)
    _validate_jagged_fields(yaml_dict)
    for field in yaml_dict['jagged_fields']:
        if isinstance(field, dict):
            validate_yaml(field)


def _validate_fixed_fields(yaml_dict):
    if 'fixed_fields' not in yaml_dict:
        raise ValueError("Yaml file is missing \'fixed_fields\', it is a required field.")
    _validate_embed_dim(yaml_dict['fixed_fields'])
    _validate_mode(yaml_dict['fixed_fields'])
    _validate_nonlinearity(yaml_dict['fixed_fields'])
    _validate_hidden_size(yaml_dict['fixed_fields'])
    _validate_use(yaml_dict['fixed_fields'])
    _validate_fields(yaml_dict['fixed_fields'])


def _validate_jagged_fields(yaml_dict):
    if 'jagged_fields' not in yaml_dict:
        raise ValueError("Yaml file is missing \'jagged_fields\', it is a required field.")
    _validate_embed_dim(yaml_dict['jagged_fields'])
    _validate_mode(yaml_dict['jagged_fields'])
    _validate_nonlinearity(yaml_dict['jagged_fields'])
    _validate_hidden_size(yaml_dict['jagged_fields'])
    _validate_use(yaml_dict['jagged_fields'])
    _validate_fields(yaml_dict['jagged_fields'])


################################################################
#           functions to validate fields individually          #
################################################################

def _validate_fields(yaml_dict):
    if 'fields' not in yaml_dict:
        raise ValueError("Yaml file is missing \'fields\', it is a required field.")


def _validate_embed_dim(yaml_dict):
    if 'embed_dim' not in yaml_dict:
        raise ValueError("Yaml file is missing \'embed_dim\', it is a required field.")
    if not isinstance(yaml_dict['embed_dim'], int) or yaml_dict['embed_dim'] <= 0:
        raise ValueError("embed_dim must be a int > 0, got {}".format(yaml_dict['embed_dim']))


def _validate_mode(yaml_dict):
    if 'mode' not in yaml_dict:
        raise ValueError("Yaml file is missing \'mode\', it is a required field.")
    if not isinstance(yaml_dict['mode'], str) or yaml_dict['mode'].strip().lower() not in MODES:
        raise ValueError("The mode '%s' is not supported. Supported modes"
                         " are %s." % (yaml_dict['mode'], MODES))


def _validate_hidden_size(yaml_dict):
    if 'hidden_size' in yaml_dict:
        if not isinstance(yaml_dict['hidden_size'], str) or \
                _hidden_size_positiveness(yaml_dict['hidden_size'].strip()):
            raise ValueError("all hidden sizes must be > 0, got %s." % yaml_dict['hidden_size'])
        pass


def _validate_nonlinearity(yaml_dict):
    if 'nonlinearity' in yaml_dict:
        if not isinstance(yaml_dict['nonlinearity'], str) or \
                yaml_dict['nonlinearity'].strip().lower() not in ACTIVATIONS:
            raise ValueError("The nonlinearity '%s' is not supported. Supported "
                             "nonlinearities are %s."
                             % (yaml_dict['nonlinearity'], ACTIVATIONS))


def _validate_use(yaml_dict):
    if 'use' in yaml_dict and not isinstance(yaml_dict['use'], bool):
        raise ValueError("use must be either True or False, got %s." % yaml_dict['use'])


################################################################
#            helper functions for validate_yaml.py             #
################################################################

def _hidden_size_positiveness(hidden_size_string):
    hidden_size_list = parse_hidden_size_string(hidden_size_string)
    for hidden_size in hidden_size_list:
        if hidden_size <= 0:
            return False
    return True

