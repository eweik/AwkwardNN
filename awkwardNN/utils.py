# utils.py
# Helper functions for AwkwardNNpractice

import os
import torch
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt


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
    #loss = F.nll_loss(log_class_prob, y)
    loss = F.cross_entropy(log_class_prob, y)
    return loss


def get_accuracy(y, log_class_prob):
    _, prediction = torch.max(log_class_prob, 1)
    correct = (prediction == y).float()
    acc = 100. * (correct.sum() / len(y))
    return acc


def print_time(start):
    time_elapsed = time.time() - start
    print('\nComplete in {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))
    return


def print_train_stat(epoch, batch_idx, data, trainsize, loss, acc):
    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.0f}%'.format(
        epoch,
        batch_idx * len(data),
        trainsize,
        100. * batch_idx * len(data) / trainsize,
        loss.item(),
        acc.item()))
    return


def print_valid_stat(valid_loss, valid_acc, validsize, best_valid_acc):
    correct = int(valid_acc * validsize / 100)
    print('\nValid set:\n    ', end="")
    end = " [*]\n" if valid_acc > best_valid_acc else "\n"
    print('Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        valid_loss,
        correct,
        validsize,
        valid_acc), end=end)
    return


def print_test_set(test_loss, correct, acc, testsize):
    print('\n[*] Test set:\n    ', end="")
    print('Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss,
                                                                  correct,
                                                                  testsize,
                                                                  acc))
    return


def plot_loss_acc(train_loss, train_acc, valid_loss, valid_acc, plot_dir):
    plt.plot(train_loss)
    plt.plot(valid_loss)
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.savefig(plot_dir + "/loss.pdf")
    plt.close()

    plt.plot(train_acc)
    plt.plot(valid_acc)
    plt.xlabel("iterations")
    plt.ylabel("accuracy")
    plt.savefig(plot_dir + "/acc.pdf")

