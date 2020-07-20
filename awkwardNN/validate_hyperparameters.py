_STOCHASTIC_SOLVERS = ['sgd', 'adam']
_LEARNING_RATES = ["constant", "adaptive"]
ACTIVATIONS = ['tanh', 'relu']
MODES = ['gru', 'lstm', 'rnn', 'deepset', 'mixed']


def _layer_sizes_less_than_zero(layer_dim):
    for i in layer_dim:
        if i <= 0:
            return True
    return False


def _validate_hyperparameters(net):

    if net.hidden_size < 1:
        raise ValueError("hidden_size must be > 0, got %s." % net.hidden_size)
    if net.num_layers < 1:
        raise ValueError("num_layers must be > 0, got %s." % net.num_layers)
    if len(net.phi_sizes) < 2 or _layer_sizes_less_than_zero(net.phi_sizes):
        raise ValueError("phi_sizes must be a list with more than one element "
                         "and elements must be > 0, got {}".format(net.phi_sizes))
    if len(net.rho_sizes) < 2 or _layer_sizes_less_than_zero(net.rho_sizes):
        raise ValueError("rho_sizes must be a list with more than one element "
                         "and elements must be > 0, got {}".format(net.rho_sizes))
    if net.batch_size < 0:
        raise ValueError("batch_size must be > 0, got %s." % net.batch_size)
    if net.learning_rate in _LEARNING_RATES and net.learning_rate_init <= 0:
        raise ValueError("learning_rate_init must be > 0, got %s." %
                         net.learning_rate_init)
    if net.epochs <= 0:
        raise ValueError("max_iter must be > 0, got %s." % net.max_iter)
    if not isinstance(net.shuffle, bool):
        raise ValueError("shuffle must be either True or False %s." % net.shuffle)
    if net.tol < 0:
        raise ValueError("tol must be >= 0, got %s." % net.tol)
    if not isinstance(net.verbose, bool):
        raise ValueError("verbose must be either True or False, got %s." % net.verbose)
    if not isinstance(net.resume_training, bool):
        raise ValueError("resume_training must be either True or False, got %s." %
                         net.resume_training)
    if not isinstance(net.load_best, bool):
        raise ValueError("load_best must be either True or False %s." % net.load_best)
    if net.momentum > 1 or net.momentum < 0:
        raise ValueError("momentum must be >= 0 and <= 1, got %s." % net.momentum)
    if not isinstance(net.nesterovs_momentum, bool):
        raise ValueError("nesterovs_momentum must be either True or False,"
                         " got %s." % net.nesterovs_momentum)
    if not isinstance(net.early_stopping, bool):
        raise ValueError("early_stopping must be either True or False,"
                         " got %s." % net.early_stopping)
    if net.validation_fraction < 0 or net.validation_fraction >= 1:
        raise ValueError("validation_fraction must be >= 0 and < 1, "
                         "got %s" % net.validation_fraction)
    if net.beta_1 < 0 or net.beta_1 >= 1:
        raise ValueError("beta_1 must be >= 0 and < 1, got %s" % net.beta_1)
    if net.beta_2 < 0 or net.beta_2 >= 1:
        raise ValueError("beta_2 must be >= 0 and < 1, got %s" % net.beta_2)
    if net.epsilon <= 0.0:
        raise ValueError("epsilon must be > 0, got %s." % net.epsilon)
    if net.n_iter_no_change <= 0:
        raise ValueError("n_iter_no_change must be > 0, got %s." % net.n_iter_no_change)
    if net.lr_decay_step <= 0.0:
        raise ValueError("lr_decay_step must be > 0, got %s." % net.lr_decay_step)
    if net.lr_decay_factor <= 0.0:
        raise ValueError("lr_decay_factor must be > 0, got %s." % net.lr_decay_factor)
    if net.dropout < 0.0:
        raise ValueError("dropout must be >= 0, got %s." % net.dropout)

    # raise ValueError if not registered
    if net.mode not in MODES:
        raise ValueError("The mode '%s' is not supported. Supported modes"
                         " are %s." % (net.mode, list(sorted(MODES))))
    if net.activation not in ACTIVATIONS:
        raise ValueError("The activation '%s' is not supported. Supported "
                         "activations are %s."
                         % (net.activation, list(sorted(ACTIVATIONS))))
    if net.solver not in _STOCHASTIC_SOLVERS:
        raise ValueError("The solver %s is not supported. "
                         " Expected one of: %s" %
                         (net.solver, ", ".join(_STOCHASTIC_SOLVERS)))
    return
