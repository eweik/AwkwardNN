import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

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


def plot_loss_acc(train_loss, train_acc, valid_loss, valid_acc, plot_dir):
    plt.plot(train_loss, label='Train')
    plt.plot(valid_loss, label='Valid')
    plt.ylim([0, 1])
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(plot_dir + "/loss.pdf")
    plt.close()

    plt.plot(train_acc, label='Train')
    plt.plot(valid_acc, label='Valid')
    plt.ylim([0, 100])
    plt.xlabel("iterations")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig(plot_dir + "/accuracy.pdf")

