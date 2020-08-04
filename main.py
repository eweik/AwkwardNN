#

import uproot
from sklearn.model_selection import train_test_split
from awkwardNN.awkwardNN import awkwardNN, awkwardNN_fromYaml
from awkwardNN.awkwardDataset import AwkwardDatasetFromYaml
from awkwardNN.utils.yaml_utils import get_yaml_dict_list
from awkwardNN.awkwardDataset import get_events_from_tree


if __name__ == "__main__":
    editted_yaml = './test_qcd_1000.yaml'
    model = awkwardNN_fromYaml(editted_yaml, max_iter=3, verbose=True)
    data_info = [{'rootfile': './data/test_qcd_1000.root', 'target': 0},
                 {'rootfile': './data/test_ttbar_1000.root', 'target': 1}]
    model.train(data_info)

    # tree1 = uproot.open("./data/test_qcd_1000.root")["Delphes"]
    # tree2 = uproot.open("./data/test_ttbar_1000.root")["Delphes"]
    # varying_fields = ["Jet*"]
    # X1 = get_events_from_tree(tree1, varying_fields)
    # X2 = get_events_from_tree(tree2, varying_fields)
    # y1 = [1] * len(X1)
    # y2 = [0] * len(X2)
    # X = X1 + X2
    # y = y1 + y2
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    # num_epochs = 3

    # rnn_double_jagged = awkwardNN(mode='deepset', max_iter=num_epochs, verbose=True)
    # rnn_double_jagged.train(X_train, y_train)
    # rnn_double_jagged.test(X_test, y_test)