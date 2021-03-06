import uproot4


def get_roottree_dict_list(rootfile_dict_list):
    roottree_dict_list = []
    for data_dict in rootfile_dict_list:
        roottree = get_roottree(data_dict['rootfile'])
        roottree_dict = {'roottree': roottree, 'target': data_dict['target']}
        roottree_dict_list.append(roottree_dict)
    return roottree_dict_list

###############################################################
#   helper functions for direct interaction with rootfiles    #
###############################################################
def get_roottree(rootfile):
    """
    Get the subtree in rootfile that doesn't return an error when you
    try and access its `keys`.
    Possible error: what if rootfile has multiple such trees
    :param rootfile:i
    :return:
    """
    for tree in uproot4.open(rootfile).values():
        try:
            tree.keys()  # try to access keys
            return tree
        except:
            pass
    raise AssertionError("no uproot.rootio.TTree found in {}".format(rootfile))


###########################################################################
#      helper functions deal with the interpretation of root fields       #
###########################################################################
def is_nested(roottree, key):
    return isinstance(roottree[key].interpretation, uproot4.interpretation.grouped.AsGrouped)


def is_fixed(roottree, key):
    return isinstance(roottree[key].interpretation, uproot4.interpretation.numerical.Numerical)


def is_jagged(roottree, key):
    return isinstance(roottree[key].interpretation, uproot4.interpretation.jagged.AsJagged)


def is_group(roottree, key):
    return isinstance(roottree[key].interpretation, uproot4.interpretation.grouped.AsGrouped)


def is_object(roottree, key):
    return isinstance(roottree[key].interpretation, uproot4.interpretation.objects.AsObjects)

