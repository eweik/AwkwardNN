# createAwkwardData.py
# Create nested and variable length dataset for AwkwardNNpractice
# Binary and Data is nested data based on targets


import numpy as np
import awkward


def generate_data_target(num_events, prob_nest, prob_sig, prob_bkg, max_len=100, max_depth=10):
    check_probabilities_of_element_options(prob_nest, prob_sig, prob_bkg)
    assert num_events > 0
    targets = generate_binary_targets(num_events, prob_sig, prob_bkg)
    data = generate_data(targets, prob_nest, prob_sig + prob_bkg, max_depth,  max_len, max_depth)
    data = awkward.fromiter(data)
    return data, targets


def generate_data(targets, prob_nest, prob_data, noise_range, max_len=100, max_depth=10):
    data = []
    for i in targets:
        event = generate_event(i, prob_nest, prob_data, noise_range, max_len, max_depth)
        data.append(event)
    return data


def generate_event(target, prob_nest, prob_data, noise_range, max_len=100, max_depth=10):
    event = []
    event_length = np.random.randint(1, max_len + 1)
    for j in range(event_length):
        element = get_array_element(target, prob_nest, prob_data, noise_range, max_len, max_depth)
        event.append(element)
    return event


def get_array_element(target, prob_nest, prob_data, noise_range, max_len, max_depth):
    n = np.random.rand()
    if max_depth <= 1:
        n += prob_nest  # so it can't activate the first `if` statement
    if n < prob_nest:
        return generate_event(target, prob_nest, prob_data, noise_range, max_len, max_depth-1)
    return get_data_signal(target, max_depth)


def get_data_signal(target, max_depth):
    # data is Gaussian distributed with std deviation 1 and
    # mean depending on target and depth of signal
    # ie deeper in event, smaller signal magnitude.
    # less deep -> higher signal magnitude
    sign = +1 if target==1 else -1
    magnitude = max_depth
    #magnitude = 5
    return np.random.normal(sign*magnitude, 1)


def generate_binary_targets(size, prob_sig, prob_bkg):
    total_prob = prob_sig + prob_bkg
    new_p_sig = prob_sig / total_prob
    new_p_bkg = prob_bkg / total_prob
    return np.random.choice([0, 1], size=size, p=[new_p_sig, new_p_bkg])


def check_probabilities_of_element_options(p, q, r):
    assert 0 < p < 1 and 0 < q < 1 and 0 <= r < 1
    assert round(p + q + r) == 1.0


