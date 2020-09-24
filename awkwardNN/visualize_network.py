from graphviz import Digraph


def visualize_network(yaml_dict, fontsize=12):
    g = Digraph("AwkwardNN", format="png",
                node_attr={'shape': 'box', 'fontsize': str(fontsize)},
                graph_attr={'rankdir': 'BT'})
    g, _ = _fill_graph_body(g, yaml_dict, 0)
    return g


########################################################################
#                        get network blocks                            #
########################################################################


def _fill_graph_body(graph, yaml_dict, id, parent_id=-1):
    # fill top of awkward block
    name = list(yaml_dict.keys())[0]
    yaml_dict = yaml_dict[name]
    label = _get_network_block_label(yaml_dict, name)
    graph.node(str(id), label)
    if parent_id > 0:
        graph.edge(str(id), str(parent_id))

    # fill out sub blocks of awkward block
    parent_id = id
    graph, id = _fill_block(graph, yaml_dict, parent_id, id, 'fixed_fields', 'Fixed Block')
    graph, id = _fill_block(graph, yaml_dict, parent_id, id, 'jagged_fields', 'Jagged Block')
    graph, id = _fill_block(graph, yaml_dict, parent_id, id, 'object_fields', 'Object Block')
    graph, id = _fill_nested_block(graph, yaml_dict, parent_id, id)
    return graph, id


def _fill_block(graph, yaml_dict, parent_id, id, block_key, block_name):
    if block_key not in yaml_dict:
        return graph, id
    id += 1
    label = _get_network_block_label(yaml_dict[block_key], block_name)
    graph.node(str(id), label)
    graph.edge(str(id), str(parent_id))
    graph, id = _fill_field_block(graph, yaml_dict[block_key], id)
    return graph, id


def _fill_nested_block(graph, yaml_dict, parent_id, id):
    if 'nested_fields' not in yaml_dict:
        return graph, id
    id += 1
    label = _get_network_block_label(yaml_dict, 'Nested Block')
    graph.node(str(id), label)
    graph.edge(str(id), str(parent_id))

    # fill sub blocks of nested block
    node_id = id
    for f in yaml_dict['nested_fields']['fields']:
        graph, id = _fill_graph_body(graph, f, id+1, node_id)
    return graph, id


def _fill_field_block(graph, yaml_dict, id):
    label = _get_fields_label(yaml_dict)
    graph.node(str(id+1), label)
    graph.edge(str(id+1), str(id))
    return graph, id+1


########################################################################
#                              get labels                              #
########################################################################

def _get_network_block_label(yaml_dict, name):
    label = name
    label += '\nembed_dim: ' + str(yaml_dict['embed_dim'])
    label += '\nmode: ' + yaml_dict['mode']
    if yaml_dict['mode'] == 'mlp':
        label = _add_mlp_block_label(yaml_dict, label)
    elif yaml_dict['mode'] == 'deepset':
        label = _add_deepset_block_label(yaml_dict, label)
    elif yaml_dict['mode'] == 'vanilla_rnn':
        label = _add_vanilla_rnn_block_label(yaml_dict, label)
    return label


def _add_mlp_block_label(yaml_dict, label):
    label = _check_and_add_attr_to_label(yaml_dict, label, 'hidden_sizes')
    label = _check_and_add_attr_to_label(yaml_dict, label, 'nonlinearity')
    return label


def _add_deepset_block_label(yaml_dict, label):
    label = _check_and_add_attr_to_label(yaml_dict, label, 'phi_sizes')
    label = _check_and_add_attr_to_label(yaml_dict, label, 'rho_sizes')
    label = _check_and_add_attr_to_label(yaml_dict, label, 'nonlinearity')
    return label


def _add_vanilla_rnn_block_label(yaml_dict, label):
    label = _check_and_add_attr_to_label(yaml_dict, label, 'nonlinearity')
    return label


def _check_and_add_attr_to_label(yaml_dict, label, key):
    if key in yaml_dict:
        label += '\n{}: {}'.format(key, yaml_dict[key])
    return label


def _get_fields_label(yaml_dict):
    label = ''
    for field in yaml_dict['fields']:
        label += field + '\n'
    return label.rstrip()


