import torch
from torch.utils.data import Dataset
from codebase.rudder.utils import load_trajectories
import numpy as np
import gym

class Environment(Dataset):

    def __init__(self, env_name : str, batch_size : int, n_trajectories : int, perct_optimal : float):
        """
        :param env_name: Gym Environnement
        :param n_trajectories: Number of trajectories to include in the data
        :param perct_optimal: Percentage of optimal trajectory in the data
        """
        super(Environment, self).__init__()

        self.gym = gym.make(env_name)

        self.n_states = self.gym.observation_space.shape[0]
        self.n_actions = self.gym.action_space.n
        self.max_episode_length = self.gym.spec.max_episode_steps

        self.n_trajectories = n_trajectories
        self.batch_size = batch_size

        data = load_trajectories(self.gym, n_trajectories=n_trajectories, perct_optimal=perct_optimal)

        # Dictionnary keys in data : ['observation', 'action', 'reward', 'traj_len', 'delayed_reward']
        observations = data['observation']
        actions = data['action']
        rewards = data['delayed_reward'] / 10  # Apply correction if needed ex. log, division par 100 or other
        length_of_trajectory = data['traj_len']


        # Keep 15% for validation set
        n_data_for_test = -int(0.15 * n_trajectories)

        self.actions = actions[0:n_data_for_test]
        self.observations = observations[0:n_data_for_test]
        self.rewards = rewards[0:n_data_for_test]
        self.length = length_of_trajectory[0:n_data_for_test]

        self.data_test = actions[n_data_for_test:], observations[n_data_for_test:], \
                         rewards[n_data_for_test:], length_of_trajectory[n_data_for_test:]


    def __len__(self):
        return self.rewards.shape[0]

    def __getitem__(self, idx):
        data_train = self.observations[idx], self.actions[idx], self.rewards[idx], self.length[idx]
        return data_train
