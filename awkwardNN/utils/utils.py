import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn import metrics

############################################################
# helper functions in training the neural network
############################################################

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.history = []
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.history.append(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_loss(y, log_class_prob):
    # loss = F.nll_loss(log_class_prob, y)
    loss = F.cross_entropy(log_class_prob, y)
    return loss


def get_accuracy(y, log_class_prob):
    _, prediction = torch.max(log_class_prob, 1)
    correct = (prediction == y).float()
    acc = 100. * (correct.sum() / len(y))
    return acc


def plot_loss_acc(train_loss, train_acc, valid_loss, valid_acc, title, plot_dir):
    plt.plot(train_loss, label='Train')
    plt.plot(valid_loss, label='Valid')
    plt.ylim([0, 1])
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.title(title)
    plt.legend()
    plt.savefig(plot_dir + "/loss_{}.pdf".format(title.replace(" ", "").lower()))
    plt.close()

    plt.plot(train_acc, label='Train')
    plt.plot(valid_acc, label='Valid')
    plt.ylim([0, 100])
    plt.xlabel("iterations")
    plt.ylabel("accuracy")
    plt.title(title)
    plt.legend()
    plt.savefig(plot_dir + "/accuracy_{}.pdf".format(title.replace(" ", "").lower()))


def plot_roc_curve(y_true, y_pred, curve_label):
    # create roc curve
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)
    auc = metrics.auc(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label=curve_label + ' (area = {:.2f})'.format(auc))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')


