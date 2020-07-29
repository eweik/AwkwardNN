import yaml
import uproot


def read_awkward_yaml(yamlfile):
    with open(yamlfile, 'r') as file:
        awkwardNN_list = yaml.load(file, Loader=yaml.FullLoader)
    return awkwardNN_list


def get_yaml_model_from_root(rootfile):
    dict_file = get_awkward_yaml_dict(rootfile)
    yamlfile = rootfile.rsplit('/', 1)[-1][:-5] + "_awkwardNN.yaml"
    with open(yamlfile, 'w') as file:
        yaml.dump(dict_file, file, sort_keys=False)


def get_awkward_yaml_dict(rootfile):
    roottree = get_roottree(rootfile)
    fields = [i.decode("ascii") for i in roottree.allkeys()]
    dict_file = [{'mode': 'rnn', 'fields': fields}]
    return dict_file


def get_roottree(rootfile):
    """
    Get the subtree in rootfile that doesn't return an error when you
    try and access its `keys`.
    Possible error: what if rootfile has multiple such trees
    :param rootfile:
    :return:
    """
    for tree in uproot.open(rootfile).values():
        try:
            tree.keys()  # try to access keys
            return tree
        except:
            pass
    raise AssertionError("no uproot.rootio.TTree found in {}".format(rootfile))


if __name__ == "__main__":
    # rootfile = "./data/test_qcd_1000.root"
    # trees = get_roottree(rootfile)
    # get_yaml_model_from_root(rootfile)
    # print(rootfile.rsplit('/', 1)[-1][:-5])
    x = read_awkward_yaml("./test_qcd_1000_awkwardNN.yaml")
    for i in x:
        for j, k in i.items():
            print("{} {}".format(j, k))

