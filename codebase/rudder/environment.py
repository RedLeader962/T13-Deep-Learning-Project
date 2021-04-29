import torch
from torch.utils.data import Dataset
from codebase.rudder.utils import get_env_path
from codebase.rudder.utils import save_data, generate_discete_env_single_episode
import numpy as np
import os
import gym

class Environment(Dataset):

    def __init__(self, env_name : str, n_trajectories, max_timestep: int, n_positions: int, rnd_gen: np.random.RandomState):
        super(Environment, self).__init__()

        self.gym = gym.make(env_name)
        # When we are ready to integrate, use these variables
        #n_states = self.gym.observation_space.shape[0]
        #n_actions = self.gym.action_space.n
        #max_episode_length = self.gym.spec.max_episode_steps

        n_actions = 2
        self.n_trajectories = n_trajectories
        self.batch_size = 8

        zero_position = int(np.ceil(n_positions / 2.))
        coin_position = zero_position + 2

        actions = np.asarray(rnd_gen.randint(low=0, high=2, size=(n_trajectories, max_timestep)), dtype=np.int)

        actions_onehot = np.identity(n_actions, dtype=np.float32)[actions]

        actions[:] = (actions * 2) - 1

        observations = np.full(fill_value=zero_position, shape=(n_trajectories, max_timestep), dtype=np.int)

        for t in range(max_timestep - 1):
            action = actions[:, t]

            # Essentiellement ce que ça fait c'est que ça additionne les actions au observations et ça clip
            # les observations entre 0 et la position n-1, c'est juste pour la simulation et pas pour notre problème.
            observations[:, t + 1] = np.clip(observations[:, t] + action, 0, n_positions - 1)

        observations_onehot = np.identity(n_positions, dtype=np.float32)[observations]


        rewards = np.zeros(shape=(n_trajectories, max_timestep), dtype=np.float32)
        rewards[:, -1] = observations_onehot[:, :, coin_position].sum(axis=1)

        length_of_trajectory = np.full(n_trajectories, max_timestep, dtype=np.int)

        # Keep 20% for validation set
        n_data_for_test = -int(0.2 * n_trajectories)

        self.actions = actions_onehot[0:n_data_for_test]
        self.observations = observations_onehot[0:n_data_for_test]
        self.rewards = rewards[0:n_data_for_test]
        self.length = length_of_trajectory[0:n_data_for_test]

        self.data_test = actions_onehot[n_data_for_test:], observations_onehot[n_data_for_test:], \
                         rewards[n_data_for_test:], length_of_trajectory[n_data_for_test:]

    def __len__(self):
        return self.rewards.shape[0]

    def __getitem__(self, idx):
        data_train = self.observations[idx], self.actions[idx], self.rewards[idx], self.length[idx]
        return data_train
