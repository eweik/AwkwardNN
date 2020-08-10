from awkwardNN.nets.awkwardNN import awkwardNN_fromYaml


if __name__ == "__main__":

    root_filename = "./data/test_qcd_1000.root"
    yaml_filename = './test_qcd_1000_default.yaml'
    yaml_dict_list = awkwardNN_fromYaml.get_yaml_model_from_rootfile(root_filename, yaml_filename)
    data_info = [{'rootfile': './data/test_qcd_1000.root', 'target': 0},
                 {'rootfile': './data/test_ttbar_1000.root', 'target': 1}]

    # modeled a lot of the parameters and methods from scikit-learn
    model = awkwardNN_fromYaml(yaml_filename, max_iter=3, verbose=True)
    model.train(data_info)