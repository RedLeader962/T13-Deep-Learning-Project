import numpy as np
import math
import os
import pandas as pd


class EpochsLogger:
    """ Class logging useful information """

    def __init__(self, n_epoches, period_avg=5, print=True, save_gap=1, trained_with_lstm=None):

        self.n_epoches = n_epoches
        self.period_avg = period_avg

        self.log_idx = 0
        self.current_epoch = 0

        self.n_log = save_gap

        # Log containers rewards
        self.e_avg_return = np.zeros(self.n_log, dtype=np.float)
        self.moving_avg = np.zeros(self.n_log, dtype=np.float)

        # Log containers trajectories
        self.e_traj_count = np.zeros(self.n_log, dtype=np.int)
        self.e_cumul_timestep = np.zeros(self.n_log, dtype=np.int)

        self.print = print
        self.save_gap = save_gap

        self.start_epoch = 0

        self.header = 'Epoch,E_trajectorie_count,E_cumulative_timestep,E_average_return'
        self.format = '%d,%d,%d,%f'

        self.trained_with_lstm = trained_with_lstm is not None

    def log_rewards(self, reward):
        self.e_avg_return[self.log_idx] += reward
        self.e_cumul_timestep[self.log_idx] += 1

    def log_traj(self):
        self.e_traj_count[self.log_idx] += 1

    def end_epoch(self, loss_v, loss_pi):

        self.e_avg_return[self.log_idx] /= self.e_traj_count[self.log_idx]
        #self.compute_avg_rewards()

        if self.print:
            self.print_epoch_info(loss_v, loss_pi)
        self.log_idx += 1
        self.current_epoch += 1

    def compute_avg_rewards(self):
        if self.log_idx < self.period_avg:
            self.moving_avg[self.log_idx] = np.sum(self.e_avg_return[0:self.log_idx + 1]) / (self.log_idx + 1)
        else:
            prev_log = self.log_idx - self.period_avg + 1
            self.moving_avg[self.log_idx] = np.sum(self.e_avg_return[prev_log:]) / self.period_avg

    def print_epoch_info(self, loss_v, loss_pi):

        traj_epoch = self.e_traj_count[self.log_idx]
        r_epoch = self.e_avg_return[self.log_idx]

        print(
            f"Epoch #{self.current_epoch}: e_avg_return: {r_epoch:.2f}, loss_pi = {loss_pi:.4f}, loss_v = "
            f"{loss_v:.2f}, n_traject : {traj_epoch}")

    def reset(self):
        self.log_idx = 0
        self.e_avg_return = np.zeros(self.n_log, dtype=np.float)
        self.e_traj_count = np.zeros(self.n_log, dtype=np.int)
        self.e_cumul_timestep = np.zeros(self.n_log, dtype=np.int)

    def save_data(self, dir_name, dim_NN, file_start_epoch, file_end_epoch):

        dim_in, dim_hid, dim_out = dim_NN
        file_name = "data_dimIn_" + str(dim_in) + "_dimHid_" + str(dim_hid) + "_dimOut_" + str(dim_out)

        if self.trained_with_lstm :
            file_name += '_with_rudder'

        file_name = os.path.join(dir_name, file_name)

        file_exists = True if os.path.exists(file_name) else False

        save_arr = slice(0, self.log_idx)
        epoch = np.arange(file_start_epoch + file_exists, file_end_epoch + 1)

        output = np.column_stack((epoch, self.e_traj_count[save_arr],
                                  self.e_cumul_timestep[save_arr], self.e_avg_return[save_arr]))

        with open(file_name, "ab") as file:
            if file_exists:
                np.savetxt(file, output, delimiter=",", fmt=self.format)
            else:
                np.savetxt(file, output, delimiter=",", header=self.header, fmt=self.format)

        self.reset()

    def load_data(self, dir_name, dim_NN):
        dim_in, dim_hid, dim_out = dim_NN
        file_name = "data_dimIn_" + str(dim_in) + "_dimHid_" + str(dim_hid) + "_dimOut_" + str(dim_out)

        if self.trained_with_lstm :
            file_name += '_with_rudder'

        file_name = os.path.join(dir_name, file_name)

        label = self.header.split(",")
        print(f'\n+Data stored in dictionary with following label : {label}')

        data = np.loadtxt(file_name, delimiter=",")

        result = {}
        for idx, l in enumerate(label):
            result[l] = data[:, idx]

        return result
