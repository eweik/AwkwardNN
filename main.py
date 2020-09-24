import torch
from awkwardNN.nets.awkwardNN import awkwardNN_fromYaml
from awkwardNN.awkwardDataset import AwkwardDatasetFromYaml
import awkwardNN.utils.yaml_utils as yaml_utils
from awkwardNN.nets.awkwardYaml import AwkwardYaml

# import awkwardNN.utils.root_utils_uproot4 as root_utils
# import awkwardNN.utils.dataset_utils_uproot4 as dataset_utils
import awkwardNN.utils.root_utils_uproot as root_utils
import awkwardNN.utils.dataset_utils_uproot as dataset_utils


# TODO: Add jupyter notebooks
# TODO: Train 3 different models for paper
# TODO: edit github README
# TODO: Setup binder for repo
# TODO: Add documenation: sphinx or pdoc
# TODO: Clean up github repo
# TODO: Clean up code
# TODO: figure out way to deal with method types like TLorentzVector etc.
# TODO: Add to pip
# TODO: Revisit issues with uproot4 and Object fields and '[]' in field names in uproot4




if __name__ == "__main__":
    # root_filename = "./data/test_ttbar_1000.root"
    # yaml_filename = "./test_qcd_1000_default.yaml"
    # awkwardNN_fromYaml.create_yaml_file_from_rootfile(root_filename, yaml_filename)

    # yaml_filename = 'test_qcd_1000_demo.yaml'
    # data_info = [{'rootfile': './data/test_qcd_1000.root', 'target': 0},
    #              {'rootfile': './data/test_ttbar_1000.root', 'target': 1}]
    # model = awkwardNN_fromYaml(yaml_filename, max_iter=3, verbose=True)
    # model.train(data_info)

    # to save to dot file
    # from awkwardNN.visualize_network import visualize_network
    # import awkwardNN.utils.yaml_utils as yaml_utils
    # yaml_filename = './test_qcd_1000_default.yaml'
    # yaml_dict = yaml_utils.get_yaml_dict_list(yaml_filename)
    # graph = visualize_network(yaml_dict, fontsize=6)
    # graph.save("default.dot")
    # dot -Tpng default.dot -o default.png

    import numpy as np
    from sklearn import metrics
    import matplotlib.pyplot as plt

    yaml_filename = './test_qcd_1000_demo.yaml'
    model1 = awkwardNN_fromYaml(yaml_filename,
                                max_iter=2,
                                verbose=True,
                                model_name='awkwardNN_demo')
    scores1 = model1.predict_proba('./data/test_qcd_1000.root')[:, :, 1]
    scores2 = model1.predict_proba('./data/test_ttbar_1000.root')[:, :, 1]
    scores = np.concatenate((scores1, scores2))
    y1 = np.zeros(1000)
    y2 = np.ones(1000)
    y = np.concatenate((y1, y2))

    print(scores)
    print(y)

    fpr, tpr, _ = metrics.roc_curve(y, scores)
    auc = metrics.auc(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='ROC curve (area = {:.2f})'.format(auc))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()

