import torch
from awkwardNN.nets.awkwardNN import awkwardNN_fromYaml
from awkwardNN.utils.dataset_utils import get_events_from_tree
import uproot


# TODO: Finish identity networks
# TODO: finish proba, log_proba, predict
# TODO: Another jupyter notebook
# TODO: Add documenation, sphinx or pdoc
# TODO: Setup binder for repo
# TODO: Clean up github repo


if __name__ == "__main__":
    yaml_filename = 'test_qcd_1000_test5.yaml'
    data_info = [{'rootfile': './data/test_qcd_1000.root', 'target': 0},
                 {'rootfile': './data/test_ttbar_1000.root', 'target': 1}]

    # modeled a lot of the parameters and methods from scikit-learn
    model = awkwardNN_fromYaml(yaml_filename, max_iter=3, verbose=True)
    model.train(data_info)


