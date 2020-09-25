import numpy as np
import matplotlib.pyplot as plt
from awkwardNN.nets.awkwardNN import awkwardNN_fromYaml

from awkwardNN.utils.utils import plot_loss_acc, plot_roc_curve
from awkwardNN.visualize_network import visualize_network


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
    data_info = [{'rootfile': './data/test_qcd_1000.root', 'target': 0},
                 {'rootfile': './data/test_ttbar_1000.root', 'target': 1}]
    y1 = np.zeros(1000)
    y2 = np.ones(1000)
    y = np.concatenate((y1, y2))

    for i in range(3):
        yamlfile = './yaml/test{}.yaml'.format(i+1)
        model = awkwardNN_fromYaml(yamlfile, max_iter=100, verbose=True,
                                   resume_training=False,
                                   model_name='test_case{}'.format(i+1))
        model.train(data_info)
        plot_loss_acc(model._train_losses, model._train_accs,
                      model._valid_losses, model._valid_accs,
                      "Test case {}".format(i+1), "./plots")
        graph = visualize_network(yamlfile, fontsize=10)
        graph.save("test_case{}.dot".format(i+1))
        # dot -Tpng test_case1.dot -o ./plots/test_case1.png


