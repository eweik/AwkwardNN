import torch
from awkwardNN.nets.awkwardNN import awkwardNN_fromYaml

# import awkwardNN.utils.root_utils_uproot4 as root_utils
# import awkwardNN.utils.dataset_utils_uproot4 as dataset_utils
import awkwardNN.utils.root_utils_uproot as root_utils
import awkwardNN.utils.dataset_utils_uproot as dataset_utils


# TODO: Add jupyter notebooks
# TODO: Setup binder for repo
# TODO: create torchvision to visualize network
# TODO: Add documenation: sphinx or pdoc
# TODO: Clean up github repo
# TODO: Add to pip
# TODO: Revisit issues with uproot4 and Object fields and '[]' in field names in uproot4


if __name__ == "__main__":
    yaml_filename = 'test_qcd_1000_test2.yaml'
    data_info = [{'rootfile': './data/test_qcd_1000.root', 'target': 0},
                 {'rootfile': './data/test_ttbar_1000.root', 'target': 1}]
    model = awkwardNN_fromYaml(yaml_filename, max_iter=3, verbose=True)
    model.train(data_info)

    # yamlfile = 'test_qcd_1000_test2.yaml'
    # kwargs = {'embed_dim': 64, 'fixed_mode': 'deepset', 'jagged_mode': 'lstm'}
    # rootfile = './data/test_ttbar_1000.root'
    # awkwardNN_fromYaml.create_yaml_file_from_rootfile(rootfile, yamlfile, **kwargs)
    # for i in range(15):
    #     yamlfile = 'test_qcd_1000_test{}.yaml'.format(i+1)
    #     awkwardNN_fromYaml.create_yaml_file_from_rootfile(rootfile, yamlfile)

    # fields1 = ['Jet_size', 'Muon_size']  # pass
    # fields2 = ['Jet.Eta', 'Jet.Phi']  # pass
    # fields3 = ['Jet.Eta', "Jet.Tau[5]", 'Jet.Phi', 'Jet.PT']  # uproot4 can't read in Jet.Tau[5]
    # fields4 = ['Jet.Eta', "Jet.PrunedP4[5]", 'Jet.Phi', 'Jet.PT']   # uproot4 can't read in Jet.PrunedP4[5]
    # fields5 = ['Jet.Constituents', 'Jet.Particles']
    # fields6 = ['Muon.Eta', 'Muon.Phi']  # pass
    # fields7 = ['Jet_size', 'Jet.Eta']  # should return error  # pass
    # fields8 = ['Jet.Constituents', 'Jet_size', 'Jet.Eta']  # should return error  # pass
    #
    # rootfile = './data/test_ttbar_1000.root'
    # roottree = root_utils.get_roottree(rootfile)
    # data1 = dataset_utils.get_events_from_tree(roottree, fields1)
    # data2 = dataset_utils.get_events_from_tree(roottree, fields6)
    # data3 = dataset_utils.get_events_from_tree(roottree, fields5)
    # print(data1[0])
    # print(data2[0])
    # print(data3[0])
    # print()
    # print(len(data1[0]))
    # print(len(data1[0][0]))
    # print(len(data2[0]))
    # print(len(data2[0][0]))
    # print(len(data3[0]))
    # print(len(data3[0][0]))

