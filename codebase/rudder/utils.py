import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import gym

def plot_lstm_reward(predicted, expected, epoch):
    """
    :param predicted: prediction from the LSTM on the whole sequence
    :param expected: expected rewards at the end of the sequence
    :param epoch: epoch #
    """
    predicted = np.round(predicted[:, -1, -1].detach().cpu().numpy(), decimals=1)
    expected = np.round(expected.detach().cpu().numpy(), decimals=1)

    plt.plot(predicted, label='Predicted', c='blue')
    plt.plot(expected, label='Expected', c='red')
    plt.title(f'Epoch {int(epoch)} : Predicted VS Expected Rewards')
    plt.xlabel('Trajectory')
    plt.ylabel('Rewards')
    plt.legend()
    plt.show()

def get_env_path(env : gym.Env):
    dir_name = env.unwrapped.spec.id
    root_path = os.path.relpath('../experiment/cherypicked')
    env_path  = os.path.join(root_path, dir_name)
    return env_path

def save_data(env : gym.Env, data_network, file_name : str):
    """
    :param env: Gym environnment
    :param data_network: Dataset of trajectories
    :param file_name: name of the file
    """
    env_path = get_env_path(env)
    file_path = os.path.join(env_path, file_name)
    torch.save(data_network, file_path)
    print(file_name, 'saved in', env_path)

def plot_reward_over_epoches():
    if spec.show_plot:
        plt.title(f"PPO - Number of epoches : {n_epoches} and steps by epoch : {steps_by_epoch}")
        plt.plot(epochs_data['E_average_return'], label='E_average_return')
        plt.legend()
        plt.xlabel("Epoches")
        plt.show()

