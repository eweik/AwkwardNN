import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split
import shutil
import os

import awkwardNN.utils as utils
from awkwardNN.preprocessRoot import AwkwardDataset
from awkwardNN.validate_hyperparameters import _validate_hyperparameters
from awkwardNN.awkwardRNN import AwkwardRNN
from awkwardNN.deepset import DeepSetNetwork, AwkwardDeepSet


class awkwardNN(object):
    def __init__(self, mode='rnn', *,
                 hidden_size=64, num_layers=2,
                 phi_sizes=(64, 64), rho_sizes=(64, 64),
                 activation='relu', solver='adam',
                 batch_size=1, learning_rate='constant',
                 learning_rate_init=0.001, max_iter=200,
                 shuffle=True, tol=0.0001, verbose=False,
                 resume_training=False, load_best=True,
                 momentum=0.9, nesterovs_momentum=True,
                 early_stopping=True, validation_fraction=0.1,
                 beta_1=0.9, beta_2=0.999, epsilon=1e-08,
                 n_iter_no_change=10, lr_decay_step=30,
                 lr_decay_factor=0.1, l2=0, dropout=0,
                 ckpt_dir="./ckpt", model_name="awkwardNN"):

        self.mode = mode
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.phi_sizes = phi_sizes
        self.rho_sizes = rho_sizes
        self.activation = activation
        self.solver = solver
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.start_epoch = 0
        self.epochs = max_iter
        self.shuffle = shuffle
        self.tol = tol
        self.verbose = verbose
        self.resume_training = resume_training
        self.load_best = load_best
        self.momentum = momentum
        self.nesterovs_momentum = nesterovs_momentum
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.n_iter_no_change = n_iter_no_change
        self.lr_decay_step = lr_decay_step
        self.lr_decay_factor = lr_decay_factor
        self.l2 = l2
        self.dropout = dropout
        self.best_valid_acc = 0.
        self.best_train_loss = 0.
        self._no_improvement_counter = 0
        self.model_name = model_name

        _validate_hyperparameters(self)

        # Keep track of loss, accuracy during training
        self._train_losses = []
        self._train_accs = []
        self._valid_losses = []
        self._valid_accs = []

        self.ckpt_dir = ckpt_dir + '/' + self.model_name + '/'
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)

    def train(self, X, y):
        self._init_training(X, y)

        for epoch in range(self.start_epoch, self.epochs):
            if self.verbose:
                print('\nEpoch: {}/{}'.format(epoch+1, self.epochs))

            train_loss, train_acc = self._train_one_epoch(epoch)
            valid_loss, valid_acc = self._validate_one_epoch(epoch)
            if self.early_stopping: # and self.verbose:
                utils.print_valid_stat(epoch+1, valid_loss, valid_acc,
                                       self.validsize, self.best_valid_acc)

            if self._stop_training(valid_acc, train_loss):
                return
            self._check_progress(epoch, valid_acc, train_loss)
            self._update_loss_acc(train_loss, train_acc, valid_loss, valid_acc)
        return

    def _init_training(self, X, y):
        # Get dataloaders
        dataset = AwkwardDataset(X, y)
        if self.early_stopping:
            self.validsize = int(len(dataset) * self.validation_fraction)
            self.trainsize = len(dataset) - self.validsize
            trainset, validset = random_split(dataset, [self.trainsize, self.validsize])
            self.trainloader = DataLoader(trainset, batch_size=self.batch_size,
                                          shuffle=self.shuffle)
            self.validloader = DataLoader(validset, batch_size=self.batch_size)
        else:
            self.trainsize = len(dataset)
            self.trainloader = DataLoader(dataset, batch_size=self.batch_size,
                                          shuffle=self.shuffle)

        # initialize pytorch model
        if self.mode == 'deepset':
            self.model = AwkwardDeepSet(dataset.input_size, self.phi_sizes, self.rho_sizes,
                                        self.activation, self.dropout)
        else:
            self.model = AwkwardRNN(self.mode, dataset.input_size, self.hidden_size,
                                    self.num_layers, self.activation, self.dropout)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #self.model = self.model.to(self.device)
        if self.solver == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(),
                                        lr=self.learning_rate_init,
                                        betas=(self.beta_1, self.beta_2),
                                        eps=self.epsilon,
                                        weight_decay=self.l2)
        elif self.solver == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(),
                                       lr=self.learning_rate_init,
                                       momentum=self.momentum,
                                       nesterov=self.nesterovs_momentum,
                                       weight_decay=self.l2)

        if self.learning_rate == 'adaptive':
            self.scheduler = lr_scheduler.StepLR(self.optimizer,
                                                step_size=self.lr_decay_step,
                                                gamma=self.lr_decay_factor)

        if self.resume_training:
            self._load_checkpoint(best=True)
        self.print_freq = int(len(self.trainloader) / 10)
        return

    def _train_one_epoch(self, epoch):
        losses, accs = utils.AverageMeter(), utils.AverageMeter()
        self.model.train()
        for i, (x, y) in enumerate(self.trainloader):
            #x, y = x.to(self.device), y.to(self.device)
            #y = y.to(self.device)
            #x = torch.tensor(x, device=self.device)
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                y_hat = self.model(x)
                loss = utils.get_loss(y, y_hat)
                acc = utils.get_accuracy(y, y_hat)
                loss.backward()
                self.optimizer.step()
                losses.update(loss.item(), len(x))
                accs.update(acc.item(), len(x))

            if self.verbose and (i % self.print_freq == 0):
                utils.print_train_stat(epoch+1, i+self.print_freq,
                                       self.trainsize, loss, acc)
        if self.learning_rate == 'adaptive':
            self.scheduler.step()
        return losses.avg, accs.avg

    def _validate_one_epoch(self, epoch):
        if not self.early_stopping:
            return 0, 0

        losses, accs = utils.AverageMeter(), utils.AverageMeter()
        self.model.eval()
        for i, (x, y) in enumerate(self.validloader):
            #x, y = x.to(self.device), y.to(self.device)
            with torch.no_grad():
                y_hat = self.model(x)
                loss = utils.get_loss(y, y_hat)
                acc = utils.get_accuracy(y, y_hat)
                losses.update(loss.item(), len(x))
                accs.update(acc.item(), len(x))
        return losses.avg, accs.avg

    def test(self, X, y):
        self._init_test(X, y)
        correct = 0
        losses = utils.AverageMeter()
        self.model.eval()
        for i, (x, y) in enumerate(self.testloader):
            #x, y = x.to(self.device), y.to(self.device)
            with torch.no_grad():
                y_hat = self.model(x)
                loss = utils.get_loss(y, y_hat)
                _, prediction = torch.max(y_hat, 1)
                correct += prediction.eq(y.data.view_as(prediction)).sum()
                losses.update(loss.item(), len(x))
        acc = 100. * correct / self.testsize
        utils.print_test_set(losses.avg, correct, acc, self.testsize)
        return losses.avg, acc

    def _init_test(self, X, y):
        dataset = AwkwardDataset(X, y)
        self.testsize = len(dataset)
        self.testloader = DataLoader(dataset, batch_size=self.batch_size,
                                     shuffle=False)
        self._load_checkpoint(best=True)

    def predict(self, X):
        predictions = []
        self._init_test(X, y=[])
        self.model.eval()
        for i, x in enumerate(self.testloader):
            x = x.to(self.device)
            with torch.no_grad():
                y_hat = self.model(x)
                _, prediction = torch.max(y_hat, 1)
                predictions.append(prediction)
        return predictions

    def predict_proba(self, X):
        pred_proba = []
        self._init_test(X, y=[])
        self.model.eval()
        for i, x in enumerate(self.testloader):
            x = x.to(self.device)
            with torch.no_grad():
                y_hat = self.model(x)
                pred_proba.append(torch.exp(y_hat))
        return pred_proba

    def predict_log_proba(self, X):
        pred_log_proba = []
        self._init_test(X, y=[])
        self.model.eval()
        for i, x in enumerate(self.testloader):
            x = x.to(self.device)
            with torch.no_grad():
                y_hat = self.model(x)
                pred_log_proba.append(y_hat)
        return pred_log_proba

    def _stop_training(self, valid_acc, train_loss):
        if self.early_stopping:
            if valid_acc > self.best_valid_acc + self.tol:
                self._no_improvement_counter = 0
            else:
                self._no_improvement_counter += 1
            if self._no_improvement_counter > self.n_iter_no_change:
                print("[!] Validation score did not improve more than "
                      "tol=%f for %d consecutive epochs." % (
                      self.tol, self.n_iter_no_change))
                return True
            return False
        else:
            if train_loss > self.best_train_loss - self.tol:
                self._no_improvement_counter = 0
            else:
                self._no_improvement_counter += 1
            if self._no_improvement_counter > self.n_iter_no_change:
                print("[!] Training loss did not improve more than "
                      "tol=%f for %d consecutive epochs." % (
                          self.tol, self.n_iter_no_change))
                return True
            return False

    def _check_progress(self, epoch, valid_acc, train_loss):
        if self.early_stopping:
            is_best = valid_acc > self.best_valid_acc
            self.best_valid_acc = max(valid_acc, self.best_valid_acc)
        else:
            is_best = train_loss > self.best_train_loss
            self.best_valid_acc = max(valid_acc, self.best_valid_acc)
        state = {'epoch': epoch+1,
                 'model_state': self.model.state_dict(),
                 'optim_state': self.optimizer.state_dict(),
                 'best_valid_acc': self.best_valid_acc,
                 'best_train_loss': self.best_train_loss}
        if self.learning_rate == 'adaptive':
            state.update({'sched_state': self.scheduler.state_dict()})
        self._save_checkpoint(state, is_best)
        return

    def _update_loss_acc(self, train_loss, train_acc, valid_loss, valid_acc):
        self._train_losses.append(train_loss)
        self._train_accs.append(train_acc)
        self._valid_losses.append(valid_loss)
        self._valid_accs.append(valid_acc)

    def _save_checkpoint(self, state, is_best):
        filename = self.model_name + '_ckpt.pth'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        torch.save(state, ckpt_path)
        if is_best:
            filename = self.model_name + '_model_best.pth'
            shutil.copyfile(ckpt_path, os.path.join(self.ckpt_dir, filename))
        return

    def _load_checkpoint(self, best=False):
        if self.verbose:
            print("[*] Loading model from {}".format(self.ckpt_dir))
        filename = self.model_name + '_ckpt.pth'
        if best:
            filename = self.model_name + '_model_best.pth'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        ckpt = torch.load(ckpt_path)
        self._load_variables(filename, ckpt, best)
        return

    def _load_variables(self, filename, checkpoint, best):
        self.start_epoch = checkpoint['epoch']
        self.best_valid_acc = checkpoint['best_valid_acc']
        self.best_train_loss = checkpoint['best_train_loss']
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optim_state'])
        if self.learning_rate == 'adaptive':
            self.scheduler.load_state_dict(checkpoint['sched_state'])
        msg = "[*] Loaded {} checkpoint @ epoch {}".format(filename, self.start_epoch)
        if best:
            msg += " with best valid acc of {:.3f}".format(self.best_valid_acc)
        if self.verbose:
            print(msg)
        return

