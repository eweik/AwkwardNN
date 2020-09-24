import time


def print_time(start):
    time_elapsed = time.time() - start
    print('\nComplete in {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))
    return


def print_train_stat(epoch, batch_idx, trainsize, loss, acc):
    print('Train Epoch: {} [{}/{} ({:.0f}%)] \tLoss: {:.6f}\tAcc: {:.0f}%'.format(
        epoch,
        batch_idx,
        trainsize,
        100. * batch_idx / trainsize,
        loss,
        acc))
    return


def print_valid_stat(epoch, valid_loss, valid_acc, validsize, best_valid_acc):
    correct = int(valid_acc * validsize / 100.)
    print('\nValid set - epoch {}:\n    '.format(epoch), end="")
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

