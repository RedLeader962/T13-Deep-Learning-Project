import numpy as np
import math
import os
import pandas as pd


class EpochsLogger:
    """ Class logging useful information """

    def __init__(self, n_epoches, save_gap=1):
        self.n_epoches = n_epoches

        self.log_idx = 0
        self.current_epoch = 0

        self.save_gap = save_gap

        self.start_epoch = 0
