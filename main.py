# main.py
# Awkwarnd NN Project
# Main program to train Awkward NN

import torch
import numpy as np
from awkwardNN.preprocessAwkwardData import get_dataloader
import time
from awkwardNN.trainAwkwardNet import AwkwardNNTrainer
from awkwardNN.config import get_config
from awkwardNN.utils import print_time
from awkwardNN.preprocessAwkwardData import *


def main(config):

    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.random_seed)

    if config.train:
        trainloader = get_dataloader(
            dataset_size=config.train_size, batch_size=config.batch_size,
            prob_nest=config.prob_nest, prob_signal=config.prob_signal,
            prob_noise=config.prob_noise, max_len=config.max_len,
            max_depth=config.max_depth
        )
        validloader = get_dataloader(
            dataset_size=config.valid_size, batch_size=config.batch_size,
            prob_nest=config.prob_nest, prob_signal=config.prob_signal,
            prob_noise=config.prob_noise, max_len=config.max_len,
            max_depth=config.max_depth
        )
        dataloader = (trainloader, validloader)
    else:
        dataloader = get_dataloader(
            dataset_size=config.test_size, batch_size=config.batch_size,
            prob_nest=config.prob_nest, prob_signal=config.prob_signal,
            prob_noise=config.prob_noise, max_len=config.max_len,
            max_depth=config.max_depth
        )

    trainer = AwkwardNNTrainer(config, config.max_depth, 1, 2, dataloader)
    if config.train:
        trainer.train()
    else:
        trainer.test()


if __name__ == "__main__":
    config, _ = get_config()
    start = time.time()
    main(config)
    print_time(start)

