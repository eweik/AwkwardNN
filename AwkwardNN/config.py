# config.py
#
#

import argparse


def str2bool(v):
    return v.lower() in ('true', '1')


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed


arg_lists = []
parser = argparse.ArgumentParser(description="Awkward NN")

# Awkward NN parameters
awkward_arg = add_argument_group('Awkward Network Parameters')
awkward_arg.add_argument('--hidden_size', type=int, default=32,
                         help='# of nodes in each hidden layer in Awkward NN')

# Data parameters
data_arg = add_argument_group('Data Parameters')
data_arg.add_argument('--train_size', type=int, default=10000,
                      help='# of data examples used for training')
data_arg.add_argument('--valid_size', type=int, default=1000,
                    help='# of data examples used for validation')
data_arg.add_argument('--test_size', type=int, default=1000,
                      help='# of data examples used for testing')
data_arg.add_argument('--batch_size', type=int, default=1,
                      help='# of data points in each batch of data')
data_arg.add_argument('--prob_nest', type=float, default=0.4,
                      help='Probability that element in awkward array is an awkward array')
data_arg.add_argument('--prob_signal', type=float, default=0.5,
                      help='Probability that element in awkward array is signal')
data_arg.add_argument('--prob_noise', type=float, default=0.1,
                      help='Probability that element in awkward array is noise')
data_arg.add_argument('--max_len', type=int, default=3,
                      help='Max length for each awkward array and its nested arrays')
data_arg.add_argument('--max_depth', type=int, default=3,
                      help='Max depth for each awkward array')

# Training parameters
train_arg = add_argument_group('Training Parameters')
train_arg.add_argument('--train', type=str2bool, default=True,
                       help='Whether to train or test the model')
train_arg.add_argument('--learning_rate', type=float, default=0.0003,
                       help='Learning rate value')
train_arg.add_argument('--epochs', type=int, default=200,
                       help='# of epochs to train for')
train_arg.add_argument('--momentum', type=float, default=0.9,
                       help='SGD momentum value')
train_arg.add_argument('--train_patience', type=int, default=10,
                       help='Number of epochs to wait before stopping training')
train_arg.add_argument('--lr_decay_step', type=int, default=50,
                       help='Number of steps before decreasing learning rate')
train_arg.add_argument('--lr_decay_factor', type=float, default=0.1,
                       help='Factor by which to decay learning rate')

# Miscellaneous parameters
misc_arg = add_argument_group('Misc. Params')
misc_arg.add_argument('--random_seed', type=int, default=2,
                      help='Seed to ensure reproducibility')
misc_arg.add_argument('--resume_training', type=str2bool, default=False,
                      help='Whether to resume training from checkpoint')
misc_arg.add_argument('--load_best', type=str2bool, default=True,
                      help='Load best model or most recent for training')
misc_arg.add_argument('--plot_dir', type=str, default='./plot',
                      help='Directory in which plots are stored')
misc_arg.add_argument('--ckpt_dir', type=str, default='./ckpt',
                      help='Directory in which to save model checkpoints')
misc_arg.add_argument('--print_freq', type=int, default=100,
                      help='How frequently to print statistics in epoch')
