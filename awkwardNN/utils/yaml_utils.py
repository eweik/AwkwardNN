import yaml
from awkwardNN.utils.root_utils import get_roottree


def get_yaml_dict_list(yamlfile):
    with open(yamlfile, 'r') as file:
        awkwardNN_list = yaml.load(file, Loader=yaml.FullLoader)
    return awkwardNN_list


def get_default_yaml_dict_from_rootfile(rootfile):
    roottree = get_roottree(rootfile)
    fields = [i.decode("ascii") for i in roottree.allkeys()]
    dict_list = [{'mode': 'rnn', 'fields': fields}]
    return dict_list


if __name__ == "__main__":
    # rootfile = "./data/test_qcd_1000.root"
    # trees = get_roottree(rootfile)
    # get_yaml_model_from_root(rootfile)
    # print(rootfile.rsplit('/', 1)[-1][:-5])
    x = get_yaml_dict_list("./test_qcd_1000_awkwardNN.yaml")
    print(x)
    print()
    for i in x:
        for j, k in i.items():
            print("{}: {}".format(j, k))

