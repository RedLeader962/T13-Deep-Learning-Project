from .utils import plot_reward
import torch
from torch.utils.data import DataLoader
from codebase.rudder.environment import Environment
import numpy as np

def train_rudder(network_lstm, optimizer, n_epoches, env : Environment, show_gap=5, device='cpu', show_plot=True):
    """
    :param network_lstm: LSTM Class
    :param optimizer: optimizer ex. ADAM or SGD
    :param n_epoches: # of epoches to train
    :param env: Environnement Class
    :param show_gap: # of steps between each graph plot to keep track of training
    :param device: cpu or gpu
    :param show_plot: If the plot has to be show or not o keep track of training
    :return: training and test loss
    """
    # Load data
    data_train = DataLoader(env, batch_size=env.batch_size)
    data_test  = env.data_test

    # Loss tracker
    train_loss_tracker =  np.zeros(n_epoches)
    test_loss_tracker  = np.zeros(n_epoches)

    for epoch in range(n_epoches):
        train_loss = 0

        network_lstm.train()

        with torch.enable_grad():
            for data in data_train:

                # Hidden state
                hs = None

                # Get batchs of observations, actions, rewards and trajectory length
                observations, actions, rewards, length = data
                observations, actions, rewards, length = observations.to(device), actions.to(device), rewards.to(device), length.to(device)
                r_expected = rewards.sum(dim=1)

                # Reset gradients
                optimizer.zero_grad()

                # Get predicted reward form network
                r_predicted = network_lstm(observations, actions, length, hs)

                # Compute loss, backpropagate gradient and update
                loss = network_lstm.compute_loss(r_predicted[..., 0], r_expected)
                loss.backward()
                optimizer.step()

                # Track loss metric
                train_loss += loss/len(data_train)

        if epoch % show_gap == 0 and show_plot and epoch != 0:
            plot_reward(r_predicted, r_expected, epoch)

        # Validate on dataset
        test_loss = validate_model_loss(network_lstm, data_test, device)

        print(f"Epoch : {epoch}, loss_train: {train_loss:8.4f}, loss test: {test_loss:8.4f}")

        train_loss_tracker[epoch] = train_loss
        test_loss_tracker[epoch]  = test_loss

        epoch += 1

    return train_loss_tracker, test_loss_tracker

def validate_model_loss(network_lstm, data_test, device):
    """
    :param network_lstm: LSTM Class
    :param data_test: Data set of trajectories to test
    :param device: cpu or gpu
    :return: test loss
    """
    hs = None

    network_lstm.eval()

    with torch.no_grad():

        observations, actions, rewards, length = data_test

        # Send observations and label to device
        observations, actions, rewards, length = observations.to(device), actions.to(device), \
                                                     rewards.to(device), length.to(device)
        r_expected = rewards.sum(dim=1)

        # Get predicted reward form network
        r_predicted = network_lstm(observations, actions, length, hs)

        # Compute loss, backpropagate gradient and update
        loss = network_lstm.compute_loss(r_predicted[..., 0], r_expected)

    return loss