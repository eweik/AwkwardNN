# trainAwkwardNet.py
#
#

import os
import torch.nn.functional as F
from AwkwardNN.awkwardNet import AwkwardNN
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from AwkwardNN.utils import *
import shutil


# Training class
class AwkwardNNTrainer(object):
    def __init__(self, config, max_depth, input_size, output_size, dataloader):
        self.config = config

        # Network parameters
        self.input_size = input_size
        self.hidden_size = config.hidden_size
        self.output_size = output_size

        # Training parameters
        self.lr = config.learning_rate
        self.start_epoch = 0
        self.epochs = config.epochs
        self.momentum = config.momentum
        self.train_patience = config.train_patience
        self.lr_decay_step = config.lr_decay_step
        self.lr_decay_factor = config.lr_decay_factor

        # Misc parameters
        self.resume_training = config.resume_training
        self.load_best = config.load_best
        self.plot_dir = config.plot_dir
        self.print_freq = config.print_freq
        self.best_valid_acc = 0.
        self.counter = 0
        self.model_name = 'awkwardNN'

        # Keep track of loss, accuracy during training
        self.total_train_loss = []
        self.total_train_acc = []
        self.total_valid_loss = []
        self.total_valid_acc = []

        # Data parameters
        self.batch_size = config.batch_size
        if config.train:
            self.trainloader, self.validloader = dataloader
            self.trainsize, self.validsize = config.train_size, config.valid_size
        else:
            self.testloader = dataloader
            self.testsize = config.test_size

        self.ckpt_dir = './ckpt/' + self.model_name + '/'
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)

        self.model = AwkwardNN(max_depth, self.input_size,
                               self.hidden_size, self.output_size)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        #self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr,
        #                           momentum=self.momentum, nesterov=True)
        # Decay LR by a factor of 0.1 every 20 epochs
        self.scheduler = lr_scheduler.StepLR(self.optimizer,
                                             step_size=self.lr_decay_step,
                                             gamma=self.lr_decay_factor)

    def train(self):
        if self.resume_training:
            self.load_checkpoint(best=True)

        for epoch in range(self.start_epoch, self.epochs):
            print('\nEpoch: {}/{}'.format(epoch+1, self.epochs))
            train_loss, train_acc = self.train_one_epoch(epoch)
            valid_loss, valid_acc = self.validate_one_epoch(epoch)
            print_valid_stat(valid_loss, valid_acc, self.validsize, self.best_valid_acc)
            if self.stop_training(valid_acc):
                return
            self.check_progress(epoch, valid_acc)
            self.update_loss_acc(train_loss, train_acc, valid_loss, valid_acc)
        return

    def train_one_epoch(self, epoch):
        losses, accs = AverageMeter(), AverageMeter()
        self.model.train()
        for i, (x, marker, y) in enumerate(self.trainloader):
            x, marker, y = x.to(self.device), marker.to(self.device), y.to(self.device)
            hidden = self.reset(self.batch_size)
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                y_hat, hidden = self.model(x, marker, hidden)
                loss = get_loss(y, y_hat)
                acc = get_accuracy(y, y_hat)
                loss.backward()
                self.optimizer.step()
                losses.update(loss.item(), x.size(0))
                accs.update(acc.item(), x.size(0))

            if i % self.print_freq == 0:
                print_train_stat(epoch+1, i+self.print_freq, x, self.trainsize, loss, acc)
        self.scheduler.step()
        return losses.avg, accs.avg

    def validate_one_epoch(self, epoch):
        losses, accs = AverageMeter(), AverageMeter()
        self.model.eval()
        for i, (x, marker, y) in enumerate(self.validloader):
            x, marker, y = x.to(self.device), marker.to(self.device), y.to(self.device)
            hidden = self.reset(self.batch_size)
            with torch.no_grad():
                y_hat, _ = self.model(x, marker, hidden)
                loss = get_loss(y, y_hat)
                acc = get_accuracy(y, y_hat)
                losses.update(loss.item(), x.size(0))
                accs.update(acc.item(), x.size(0))
        return losses.avg, accs.avg

    def test(self):
        correct = 0
        losses = AverageMeter()
        self.load_checkpoint(best=True)
        self.model.eval()
        for i, (x, marker, y) in enumerate(self.testloader):
            x, marker, y = x.to(self.device), marker.to(self.device), y.to(self.device)
            hidden = self.reset(self.batch_size)
            with torch.no_grad():
                y_hat, _ = self.model(x, marker, hidden)
                loss = get_loss(y, y_hat)
                _, prediction = torch.max(y_hat, 1)
                correct += prediction.eq( y.data.view_as(prediction) ).sum()
                losses.update(loss.item(), x.size(0))
        acc = 100. * correct / self.testsize
        print_test_set(losses.avg, correct, acc, self.testsize)
        return losses.avg, acc

    def reset(self, batch_size):
        hidden = torch.zeros(batch_size, self.hidden_size)
        hidden = hidden.to(self.device)
        return hidden

    def stop_training(self, valid_acc):
        if (valid_acc > self.best_valid_acc):
            self.counter = 0
        else:
            self.counter += 1
        if self.counter > self.train_patience:
            print("[!] No improvement in a while, stopping training.")
            return True
        return False

    def update_loss_acc(self, train_loss, train_acc, valid_loss, valid_acc):
        self.total_train_loss.append(train_loss)
        self.total_train_acc.append(train_acc)
        self.total_valid_loss.append(valid_loss)
        self.total_valid_acc.append(valid_acc)

    def check_progress(self, epoch, valid_acc):
        is_best = valid_acc > self.best_valid_acc
        self.best_valid_acc = max(valid_acc, self.best_valid_acc)
        self.save_checkpoint(
            {'epoch': epoch+1,
             'model_state': self.model.state_dict(),
             'optim_state': self.optimizer.state_dict(),
             'sched_state': self.scheduler.state_dict(),
             'best_valid_acc': self.best_valid_acc}, is_best
        )
        return

    def save_checkpoint(self, state, is_best):
        filename = self.model_name + '_ckpt.pth'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        torch.save(state, ckpt_path)
        if is_best:
            filename = self.model_name + '_model_best.pth'
            shutil.copyfile(ckpt_path, os.path.join(self.ckpt_dir, filename))
        return

    def load_checkpoint(self, best=False):
        print("[*] Loading model from {}".format(self.ckpt_dir))
        filename = self.model_name + '_ckpt.pth'
        if best:
            filename = self.model_name + '_model_best.pth'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        ckpt = torch.load(ckpt_path)
        self.load_variables(filename, ckpt, best)
        return

    def load_variables(self, filename, checkpoint, best):
        self.start_epoch = checkpoint['epoch']
        self.best_valid_acc = checkpoint['best_valid_acc']
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optim_state'])
        self.scheduler.load_state_dict(checkpoint['sched_state'])
        msg = "[*] Loaded {} checkpoint @ epoch {}".format(filename, self.start_epoch)
        if best:
            msg += " with best valid acc of {:.3f}".format(self.best_valid_acc)
        print(msg)
        return
