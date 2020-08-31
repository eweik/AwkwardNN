import awkwardNN.utils.root_utils_uproot as root_utils
import awkwardNN.utils.yaml_utils as yaml_utils

class YamlTree():
    def __init__(self, roottree, *,
                 embed_dim=32, mode='mlp',
                 fixed_mode='mlp', jagged_mode='vanilla_rnn',
                 object_mode = 'vanilla_rnn', nested_mode='mlp',
                 hidden_sizes='(32, 32)', nonlinearity='relu',
                 phi_sizes='(32, 32)', rho_sizes='(32, 32)'):
        self.roottree = roottree
        self.embed_dim = embed_dim
        self.mode = mode
        self.fixed_mode = fixed_mode
        self.jagged_mode = jagged_mode
        self.object_mode = object_mode
        self.nested_mode = nested_mode
        self.hidden_sizes = hidden_sizes
        self.nonlinearity = nonlinearity
        self.phi_sizes = phi_sizes
        self.rho_sizes = rho_sizes
        self.kwargs = {'embed_dim': embed_dim, 'mode': mode,
                       'fixed_mode': fixed_mode, 'jagged_mode': jagged_mode,
                       'object_mode': object_mode, 'nested_mode': nested_mode,
                       'hidden_sizes': hidden_sizes, 'nonlinearity': nonlinearity,
                       'phi_sizes': phi_sizes, 'rho_sizes': rho_sizes}
        self._init_field_blocks()
        self._init_roottree(roottree)

    def _init_field_blocks(self):
        fixed_sublock = self._init_subblock(self.fixed_mode)
        jagged_sublock = self._init_subblock(self.jagged_mode)
        object_subblock = self._init_subblock(self.object_mode)
        nested_subblock = self._init_subblock(self.nested_mode)
        self._yaml_block = self._init_subblock(self.mode)
        del self._yaml_block['fields']
        self._yaml_block.update({'fixed_fields': fixed_sublock,
                                 'jagged_fields': jagged_sublock,
                                 'object_fields': object_subblock,
                                 'nested_fields': nested_subblock})

    def _init_subblock(self, sub_mode):
        subblock = {'embed_dim': self.embed_dim, 'mode': sub_mode}
        if sub_mode == 'mlp':
            subblock.update({'hidden_sizes': self.hidden_sizes, 'nonlinearity': self.nonlinearity})
        elif sub_mode == 'deepset':
            update = {'phi_sizes': self.phi_sizes, 'rho_sizes': self.rho_sizes, 'nonlinearity': self.nonlinearity}
            subblock.update(update)
        elif sub_mode == 'vanilla_rnn':
            subblock.update({'nonlinearity': self.nonlinearity})
        subblock.update({'fields': []})
        return subblock

    def _init_roottree(self, roottree):
        for k in roottree.keys():
            self._add_key(roottree, k.decode('ascii'))
        self._yaml_block = yaml_utils.remove_empty_subblocks(self._yaml_block)

    def _add_key(self, roottree, key):
        try:
            roottree[key].array()
            if root_utils.is_nested(roottree, key):
                new_yaml_block = YamlTree(roottree[key], **self.kwargs).dictionary['event']
                self._yaml_block['nested_fields']['fields'].append({key: new_yaml_block})
            elif root_utils.is_fixed(roottree, key):
                self._yaml_block['fixed_fields']['fields'].append(key)
            elif root_utils.is_jagged(roottree, key):
                self._yaml_block['jagged_fields']['fields'].append(key)
            else:
                self._yaml_block['object_fields']['fields'].append(key)
        except:
            self._yaml_dict = yaml_utils.add_bad_key(self._yaml_block, key)

    @property
    def dictionary(self):
        return {'event': self._yaml_block}
