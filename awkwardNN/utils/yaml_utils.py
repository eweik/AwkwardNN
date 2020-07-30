import yaml
from awkwardNN.utils.root_utils import get_roottree


def read_awkward_yaml(yamlfile):
    with open(yamlfile, 'r') as file:
        awkwardNN_list = yaml.load(file, Loader=yaml.FullLoader)
    return awkwardNN_list


def get_yaml_model_from_root(rootfile):
    dict_file = get_default_awkward_yaml_dict_from_rootfile(rootfile)
    yamlfile = rootfile.rsplit('/', 1)[-1][:-5] + "_awkwardNN.yaml"
    with open(yamlfile, 'w') as file:
        yaml.dump(dict_file, file, sort_keys=False)


def get_default_awkward_yaml_dict_from_rootfile(rootfile):
    roottree = get_roottree(rootfile)
    fields = [i.decode("ascii") for i in roottree.allkeys()]
    dict_file = [{'mode': 'rnn', 'fields': fields}]
    return dict_file


if __name__ == "__main__":
    # rootfile = "./data/test_qcd_1000.root"
    # trees = get_roottree(rootfile)
    # get_yaml_model_from_root(rootfile)
    # print(rootfile.rsplit('/', 1)[-1][:-5])
    x = read_awkward_yaml("./test_qcd_1000_awkwardNN.yaml")
    for i in x:
        for j, k in i.items():
            print("{} {}".format(j, k))

