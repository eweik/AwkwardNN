import uproot


def get_roottree(rootfile):
    """
    Get the subtree in rootfile that doesn't return an error when you
    try and access its `keys`.
    Possible error: what if rootfile has multiple such trees
    :param rootfile:i
    :return:
    """
    for tree in uproot.open(rootfile).values():
        try:
            tree.keys()  # try to access keys
            return tree
        except:
            pass
    raise AssertionError("no uproot.rootio.TTree found in {}".format(rootfile))

