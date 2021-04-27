import torch
from torch.utils.data import Dataset
from codebase.rudder.utils import get_env_path
from codebase.rudder.utils import save_data_or_network, generate_discete_env_single_episode
import numpy as np
import os
import gym

class Environment(Dataset):

    def __init__(self, env_name : str, batch_size, max_timestep: int, n_positions: int, rnd_gen: np.random.RandomState):
        """Our simple 1D environment as PyTorch Dataset"""
        super(Environment, self).__init__()

        self.gym = gym.make(env_name)

        n_actions = 2
        self.batch_size = batch_size

        # C'est juste les règles du jeu et rien d'autre. Zéro position c'est la position de départ du joueur
        zero_position = int(np.ceil(n_positions / 2.))
        coin_position = zero_position + 2

        # Generate random action sequences. Génère une séquence d'action aléatoire de soit 0 ou 1.
        # Une ligne représente une séquence complète d'actions
        actions = np.asarray(rnd_gen.randint(low=0, high=2, size=(batch_size, max_timestep)), dtype=np.int)

        # Si l'agent a joué l'action 0 alors on prend l'identité, soit [1,0]
        # Si l'agent a joué l'action 1 alors on prend l'identité, soit [0,1]
        actions_onehot = np.identity(n_actions, dtype=np.float32)[actions]

        # Generate observations from action sequences
        # Ça permet de convertir une action 0 en -1 et l'action 1 reste là même. On va avoir une matrice de -1 et 1.
        # Ça permet de faire déplacer l'agent.
        actions[:] = (actions * 2) - 1

        # On génère 1000 samples dont la trajectoire est de 50 pas. Donc la simulation est de 50 steps au maximum
        observations = np.full(fill_value=zero_position, shape=(batch_size, max_timestep), dtype=np.int)

        #print(observations.shape)

        # On prend toutes les actions au temps 1
        for t in range(max_timestep - 1):
            action = actions[:, t]

            # Essentiellement ce que ça fait c'est que ça additionne les actions au observations et ça clip
            # les observations entre 0 et la position n-1, c'est juste pour la simulation et pas pour notre problème.
            observations[:, t + 1] = np.clip(observations[:, t] + action, 0, n_positions - 1)

        observations_onehot = np.identity(n_positions, dtype=np.float32)[observations]

        # On veut donc avoir ici la position du joueur
        #print(np.identity(n_positions, dtype=np.float32))
        #print(observations_onehot.shape)
        #print(observations.shape)

        # Calculate rewards (sum over coin position for all timesteps)
        rewards = np.zeros(shape=(batch_size, max_timestep), dtype=np.float32)
        rewards[:, -1] = observations_onehot[:, :, coin_position].sum(axis=1)

        add_reward = np.zeros((1, max_timestep), dtype=np.float32)
        add_reward[-1] = 10
        rewards = np.concatenate((rewards, add_reward), axis=0)

        add_observation = np.full(fill_value=zero_position, shape=(1, max_timestep, n_positions), dtype=np.float32)
        observations_onehot = np.concatenate((observations_onehot, add_observation), axis=0)

        print(observations_onehot.shape, add_observation.shape)

        length_of_trajectory = np.full(batch_size, max_timestep, dtype=np.int)
        add_length = np.full(fill_value=32, shape=1, dtype=np.int)
        length_of_trajectory = np.concatenate((length_of_trajectory, add_length), axis=0)
        print(length_of_trajectory.shape)

        add_action = np.full(fill_value=zero_position, shape=(1, max_timestep, 2), dtype=np.float32)
        add_action[-1,:] = np.array([1,0],dtype=np.float32)
        actions_onehot = np.concatenate((actions_onehot, add_action), axis=0)

        print(actions_onehot.shape)

        self.actions = actions_onehot
        self.observations = observations_onehot
        self.rewards = rewards
        self.length = length_of_trajectory

    def __len__(self):
        return self.rewards.shape[0]

    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx], self.rewards[idx], self.length[idx]

    def generate_trajectories(self, env: gym.Env, n_trajectory_per_policy: int, agent):
        """
        :param env: Gym environnment
        :param n_trajectory_per_policy: number of tajectories to be generated for each policy
        :param agent: PPO Neural network
        """
        data = self.__generate_trajectories(env, n_trajectory_per_policy, agent, optimal_policy=True)
        save_data_or_network(env, data, 'trajectories_optimal.csv')

        data = self.__generate_trajectories(env, n_trajectory_per_policy, agent, optimal_policy=False)
        save_data_or_network(env, data, 'trajectories_suboptimal.csv')

    def __get_policies(self, env, optimal_policy: bool):
        """
        :param env: Gym environnment
        :param optimal_policy: boolean indicating which directory to return
        :return: list of policies to create trajectories from and the path of the environnement (ex. CartPole)
        """
        env_path = get_env_path(env)

        if optimal_policy:
            dir = os.path.join(env_path, 'Optimal_policy')
        else:
            dir = os.path.join(env_path, 'SubOptimal_policy')

        policy_file = [os.path.join(dir, i) for i in os.listdir(dir)]

        return policy_file, env_path

    def __generate_trajectories(self, env: gym.Env, n_trajectory_per_policy: int, agent, optimal_policy: bool):
        """
        :param env: Gym environnment
        :param n_trajectory_per_policy: number of tajectories to be generated for each policy
        :param agent: PPO Neural network
        :param optimal_policy: if true generate optimal trajectories otherwise generate suboptimal policies
        :return a dictionnary of observations, actions, rewards, trajectory_length, delayed_rewards
        """
        max_episode_length = env.spec.max_episode_steps

        policies_names, env_path = self.__get_policies(env, optimal_policy)
        n_policies = len(policies_names)

        # Track observations, actions, rewards, trajectory length for each policy
        observations = torch.zeros((n_policies, n_trajectory_per_policy, max_episode_length, agent.state_dim))
        actions = torch.zeros((n_policies, n_trajectory_per_policy, max_episode_length, agent.action_dim))
        rewards = torch.zeros((n_policies, n_trajectory_per_policy, max_episode_length))
        delayed_rewards = torch.zeros((n_policies, n_trajectory_per_policy, max_episode_length))
        trajectory_length = torch.zeros((n_policies, n_trajectory_per_policy))

        for i, policy in enumerate(policies_names):

            # Load policy
            agent.load_state_dict(torch.load(policy))

            # Generate trajectories
            for T in range(n_trajectory_per_policy):

                obs, act, r, delayed_r, t_step = generate_discete_env_single_episode(env, agent, max_episode_length)

                observations[i, T] = obs
                actions[i, T] = act
                rewards[i, T] = r
                trajectory_length[i, T] = t_step

                # Correction of timestep if episode ends
                if t_step == max_episode_length:
                    t_step -= 1
                delayed_rewards[i, T, t_step] = delayed_r

            # Save the trajectory should go here
            print(f'    Policy {i + 1}/{n_policies} trajectory generated 100%')

        return {"observation": observations, "action": actions, "reward": rewards, 'traj_len': trajectory_length,
                'delayed_reward': delayed_rewards}

