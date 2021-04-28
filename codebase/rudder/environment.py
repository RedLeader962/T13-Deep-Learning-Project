import torch
from torch.utils.data import Dataset
from codebase.rudder.utils import get_env_path
from codebase.rudder.utils import save_data_or_network
import numpy as np
import os
import gym

class Environment(Dataset):

    def __init__(self, env_name : str, batch_size, max_timestep: int, n_positions: int, rnd_gen: np.random.RandomState):
        super(Environment, self).__init__()

        self.gym = gym.make(env_name)
        self.n_states = self.gym.observation_space.shape[0]
        self.n_actions = self.gym.action_space.n
        self.max_episode_length = self.gym.spec.max_episode_steps

        n_actions = 2
        self.batch_size = batch_size

        zero_position = int(np.ceil(n_positions / 2.))
        coin_position = zero_position + 2

        actions = np.asarray(rnd_gen.randint(low=0, high=2, size=(batch_size, max_timestep)), dtype=np.int)
        actions_onehot = np.identity(n_actions, dtype=np.float32)[actions]
        actions[:] = (actions * 2) - 1

        observations = np.full(fill_value=zero_position, shape=(batch_size, max_timestep), dtype=np.int)

        for t in range(max_timestep - 1):
            action = actions[:, t]

            observations[:, t + 1] = np.clip(observations[:, t] + action, 0, n_positions - 1)

        observations_onehot = np.identity(n_positions, dtype=np.float32)[observations]

        rewards = np.zeros(shape=(batch_size, max_timestep), dtype=np.float32)
        rewards[:, -1] = observations_onehot[:, :, coin_position].sum(axis=1)

        self.actions = actions_onehot
        self.observations = observations_onehot
        self.rewards = rewards

    def __len__(self):
        return self.rewards.shape[0]

    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx], self.rewards[idx]

    def generate_trajectories(self, n_trajectory_per_policy: int, agent):
        """
        :param env: Gym environnment
        :param n_trajectory_per_policy: number of tajectories to be generated for each policy
        :param agent: PPO Neural network
        """
        data = self.__generate_trajectories(n_trajectory_per_policy, agent, optimal_policy=True)
        save_data_or_network(self.gym, data, 'trajectories_optimal.csv')

        data = self.__generate_trajectories(n_trajectory_per_policy, agent, optimal_policy=False)
        save_data_or_network(self.gym, data, 'trajectories_suboptimal.csv')

    def load_trajectories(self, n_trajectories, perct_optimal: float = 0.5):
        """
        :param env: Gym environnment
        :param n_trajectories: number of trajectories to return
        :param perct_optimal : Percentage of optimal trajectories to return
        :return: dict of observations, actions, rewards, trajectory_length, delayed_rewards
        """
        env_path = get_env_path(self.gym)

        optimal_data = torch.load(os.path.join(env_path, 'trajectories_optimal.pt'))
        suboptimal_data = torch.load(os.path.join(env_path, 'trajectories_suboptimal.pt'))

        total_idx = len(optimal_data['observation'])

        n_optimal = int(n_trajectories * perct_optimal)
        n_suboptimal = int(n_trajectories * (1 - perct_optimal))

        assert total_idx >= n_suboptimal, f'Pas assez de données sous-optimales. Réduisez n_trajectoires ou modifier le pourcentage de données optimales.'
        assert total_idx >= n_optimal, f'Pas assez de données optimales. Réduisez n_trajectoires ou modifier le pourcentage de données optimales.'

        optimal_idx, suboptimal_idx = self.__random_idx_sample(n_optimal, n_suboptimal, total_idx)

        data = {}
        for key in optimal_data.keys():
            optim = optimal_data[key]
            suboptim = suboptimal_data[key]
            data[key] = torch.cat((optim[optimal_idx], suboptim[suboptimal_idx]))

        print(
            f'Optimal data loaded : {round(n_optimal / (n_trajectories) * 100, 2)}% or {n_optimal} trajectories out of {n_trajectories} trajectories')

        # {"observation", "action", "reward", 'traj_len', 'delayed_reward'}
        return data

    def __get_policies(self, optimal_policy: bool):
        """
        :param env: Gym environnment
        :param optimal_policy: boolean indicating which directory to return
        :return: list of policies to create trajectories from and the path of the environnement (ex. CartPole)
        """
        env_path = get_env_path(self.gym)

        if optimal_policy:
            dir = os.path.join(env_path, 'Optimal_policy')
        else:
            dir = os.path.join(env_path, 'SubOptimal_policy')

        policy_file = [os.path.join(dir, i) for i in os.listdir(dir)]

        return policy_file, env_path

    def __generate_trajectories(self, n_trajectory_per_policy: int, agent, optimal_policy: bool):
        """
        :param n_trajectory_per_policy: number of tajectories to be generated for each policy
        :param agent: PPO Neural network
        :param optimal_policy: if true generate optimal trajectories otherwise generate suboptimal policies
        :return a dictionnary of observations, actions, rewards, trajectory_length, delayed_rewards
        """
        max_episode_length = self.max_episode_length
        n_states = self.n_states
        n_actions = self.n_actions

        policies_names, env_path = self.__get_policies(optimal_policy)
        n_policies = len(policies_names)

        # Track observations, actions, rewards, trajectory length for each policy
        observations = torch.zeros((n_policies, n_trajectory_per_policy, max_episode_length, n_states))
        actions = torch.zeros((n_policies, n_trajectory_per_policy, max_episode_length, n_actions))
        rewards = torch.zeros((n_policies, n_trajectory_per_policy, max_episode_length))
        delayed_rewards = torch.zeros((n_policies, n_trajectory_per_policy, max_episode_length))
        trajectory_length = torch.zeros((n_policies, n_trajectory_per_policy))

        for i, policy in enumerate(policies_names):

            # Load policy
            agent.load_state_dict(torch.load(policy))

            # Generate trajectories
            for T in range(n_trajectory_per_policy):

                obs, act, r, delayed_r, t_step = self.__generate_single_episode(agent)

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

    def __generate_single_episode(self, agent):
        """
        :param agent: policy network
        :return: observations, actions, rewards, cumulative rewards and total number of timesteps
        """
        max_episode_length = self.max_episode_length
        n_states = self.n_states
        n_actions = self.n_actions

        observation = torch.zeros((max_episode_length, n_states), device=agent.device)
        action = torch.zeros((max_episode_length, n_actions), device=agent.device)
        reward = torch.zeros(max_episode_length, device=agent.device)

        t_step = 0
        reward_count = 0
        done = False
        s = torch.as_tensor(self.gym.reset(), dtype=torch.float32)

        observation[t_step] = s

        while not done:
            a, _, _ = agent.step(s)
            next_s, r, done, _ = self.gym.step(a)

            # Log state, actions and reward
            observation[t_step] = s
            action[t_step, a] = 1

            # Correction to avoid rewards to big
            # r = np.tanh(r)

            reward[t_step] = r
            reward_count += r

            s = torch.as_tensor(next_s, dtype=torch.float32)

            # Next time step
            t_step += 1

        return observation, action, reward, reward_count, t_step

    def __random_idx_sample(self, n_idx_optimal: int, n_idx_suboptimal: int, total_idx: int):
        """
        :param n_idx_optimal: number of optimal index to select
        :param n_idx_suboptimal: number of suboptimal index to select
        :param total_idx: Total possible index to select from
        :return: Index of optimal and suboptimal policies
        """
        t_optimal = torch.tensor(range(total_idx), dtype=torch.float)
        idx_optimal = torch.multinomial(t_optimal, n_idx_optimal)

        t_suboptimal = torch.tensor(range(total_idx), dtype=torch.float)
        idx_suboptimal = torch.multinomial(t_suboptimal, n_idx_suboptimal)

        return idx_optimal, idx_suboptimal
