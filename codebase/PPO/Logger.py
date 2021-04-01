import numpy as np
import math
import os
import pandas as pd

""" Class logging useful information """
class Info_logger():
    def __init__(self, n_epoches, period_avg=5, print=True, save_gap=1):

        self.n_epoches    = n_epoches
        self.period_avg   = period_avg

        self.act_log = 0
        self.current_epoch   = 0

        self.n_log = save_gap

        # Log containers rewards
        self.rewards      = np.zeros(self.n_log, dtype=np.float)
        self.moving_avg   = np.zeros(self.n_log, dtype=np.float)

        # Log containers trajectories
        self.n_traject    = np.zeros(self.n_log, dtype=np.int)
        self.traj_length  = np.zeros(self.n_log, dtype=np.int)

        self.print = print
        self.save_gap = save_gap

        self.start_epoch = 0

        self.header = 'Epoch,N_trajectories,Trajectory_length,Rewards'
        self.format = '%d,%d,%d,%f'

    def log_rewards(self, reward):
        self.rewards[self.act_log] += reward
        self.traj_length[self.act_log] += 1

    def log_traj(self):
        self.n_traject[self.act_log] += 1

    def end_epoch(self, loss_v, loss_pi):

        self.rewards[self.act_log] /= self.n_traject[self.act_log]
        #self.compute_avg_rewards()

        if self.print: self.print_epoch_info(loss_v, loss_pi)
        self.act_log += 1
        self.current_epoch += 1

    def compute_avg_rewards(self):
        if self.act_log < self.period_avg:
            self.moving_avg[self.act_log] = np.sum(self.rewards[0:self.act_log+1])/(self.act_log+1)
        else:
            prev_log = self.act_log - self.period_avg + 1
            self.moving_avg[self.act_log] = np.sum(self.rewards[prev_log:]) / self.period_avg

    def print_epoch_info(self, loss_v, loss_pi):

        traj_epoch = self.n_traject[self.act_log]
        r_epoch    = self.rewards[self.act_log]

        print(f"For episode {self.current_epoch}, rewards: {r_epoch:.2f}, loss_pi = {loss_pi:.4f}, loss_v = {loss_v:.2f}, n_traject : {traj_epoch}")

    def reset(self):
        self.act_log = 0
        self.rewards      = np.zeros(self.n_log, dtype=np.float)
        self.n_traject    = np.zeros(self.n_log, dtype=np.int)
        self.traj_length  = np.zeros(self.n_log, dtype=np.int)

    def save_data(self, dir_name, dim_NN, file_start_epoch, file_end_epoch):

        dim_in, dim_hid, dim_out = dim_NN
        file_name = "data_dimIn_"+str(dim_in)+"_dimHid_"+str(dim_hid)+"_dimOut_"+str(dim_out)+"_.csv"
        file_name = os.path.join(dir_name, file_name)

        file_exists = True if os.path.exists(file_name) else False

        save_arr = slice(0, self.act_log)
        epoch = np.arange(file_start_epoch+file_exists, file_end_epoch + 1)

        output = np.column_stack((epoch, self.n_traject[save_arr], self.traj_length[save_arr], self.rewards[save_arr]))

        with open(file_name, "ab") as file:
            if file_exists:
                np.savetxt(file, output, delimiter=",", fmt=self.format)
            else:
                np.savetxt(file, output, delimiter=",", header=self.header, fmt=self.format)

        self.reset()

    def load_data(self, dir_name, dim_NN):
        dim_in, dim_hid, dim_out = dim_NN
        file_name = "data_dimIn_"+str(dim_in)+"_dimHid_"+str(dim_hid)+"_dimOut_"+str(dim_out)+"_.csv"
        file_name = os.path.join(dir_name, file_name)

        label = self.header.split(",")
        print(f'\n+Data stored in dictionary with following label : {label}')

        data = np.loadtxt(file_name, delimiter=",")

        result = {}
        for idx, l in enumerate(label):
            result[l] = data[:,idx]

        return result