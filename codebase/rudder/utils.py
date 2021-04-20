import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import gym

def plot_reward(predicted, expected, epoch):
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

def get_policies(env, optimal_policy : bool):
    """
    :param env: Gym environnment
    :param optimal_policy: boolean indicating which directory to return
    :return: list of policies to create trajectories from and the path of the environnement (ex. CartPole)
    """
    env_path  = get_env_path(env)

    if optimal_policy:
        dir = os.path.join(env_path, 'Optimal_policy')
    else:
        dir = os.path.join(env_path, 'SubOptimal_policy')

    policy_file = [os.path.join(dir, i) for i in os.listdir(dir)]

    return policy_file, env_path

def generate_trajectories(env : gym.Env, n_trajectory_per_policy : int, agent):
    """
    :param env: Gym environnment
    :param n_trajectory_per_policy: number of tajectories to be generated for each policy
    :param agent: PPO Neural network
    """
    print('Generate optimal trajectories')
    data = __generate_trajectories(env, n_trajectory_per_policy, agent, optimal_policy=True)
    save_trajectories(env, data, optimal_policy=True)

    print('Generate suboptimal trajectories')
    data = __generate_trajectories(env, n_trajectory_per_policy, agent, optimal_policy=False)
    save_trajectories(env, data, optimal_policy=False)


def __generate_trajectories(env : gym.Env, n_trajectory_per_policy : int, agent, optimal_policy : bool):
    """
    :param env: Gym environnment
    :param n_trajectory_per_policy: number of tajectories to be generated for each policy
    :param agent: PPO Neural network
    :param optimal_policy: if true generate optimal trajectories otherwise generate suboptimal policies
    :return a dictionnary of observations, actions, rewards, trajectory_length, delayed_rewards
    """
    max_episode_length = env.spec.max_episode_steps

    policies_names, env_path = get_policies(env, optimal_policy)
    n_policies = len(policies_names)

    # Track observations, actions, rewards, trajectory length for each policy
    observation = torch.zeros((n_policies, n_trajectory_per_policy, max_episode_length, agent.state_dim))
    action = torch.zeros((n_policies, n_trajectory_per_policy, max_episode_length, agent.action_dim))
    reward = torch.zeros((n_policies, n_trajectory_per_policy, max_episode_length))
    delayed_reward = torch.zeros((n_policies, n_trajectory_per_policy, max_episode_length))
    traject_len = torch.zeros((n_policies, n_trajectory_per_policy))

    for i, policy in enumerate(policies_names):

        # Load policy
        agent.load_state_dict(torch.load(policy))

        # Generate trajectories
        for T in range(n_trajectory_per_policy):

            t_step = 0
            reward_count = 0
            done = False
            s = torch.as_tensor(env.reset(), dtype=torch.float32)

            observation[i, T, 0] = s

            while not done:
                a, _, _ = agent.step(s)
                next_s, r, done, _ = env.step(a)

                # Log state, actions and reward
                observation[i, T, t_step] = s
                action[i, T, t_step, a] = 1

                # Correction to avoid rewards to big
                r = np.tanh(r)

                reward[i, T, t_step] = r
                reward_count += r

                s = torch.as_tensor(next_s, dtype=torch.float32)

                # Next time step
                t_step += 1

            # Track trajectory length
            traject_len[i,T] = t_step

            # Track delayed reward
            if t_step == max_episode_length :
                t_step -= 1
            delayed_reward[i, T, t_step] = reward_count

        # Save the trajectory should go here
        print(f'    Policy {i+1}/{n_policies} trajectory generated 100%')

    return {"observation": observation, "action": action, "reward": reward, 'traj_len': traject_len, 'delayed_reward': delayed_reward}

def save_trajectories(env, trajectories, optimal_policy):
    """
    :param env: Gym environnment
    :param trajectories: Dataset of trajectories
    :param optimal_policy: Save the optimal or suboptimal policy
    """
    if optimal_policy:
        name = 'trajectories_optimal.csv'
    else:
        name = 'trajectories_suboptimal.csv'

    env_path = get_env_path(env)
    file_path = os.path.join(env_path, name)
    torch.save(trajectories, file_path)

def random_idx_sample(n_idx_optimal : int, n_idx_suboptimal : int, total_idx : int):
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

def load_trajectories(env : gym.Env, n_trajectories, perct_optimal : float = 0.5):
    """
    :param env: Gym environnment
    :param n_trajectories: number of trajectories to return
    :param perct_optimal : Percentage of optimal trajectories to return
    :return: dict of observations, actions, rewards, trajectory_length, delayed_rewards
    """
    env_path = get_env_path(env)

    optimal_data = torch.load(os.path.join(env_path, 'trajectories_optimal.csv'))
    suboptimal_data = torch.load(os.path.join(env_path, 'trajectories_suboptimal.csv'))

    total_idx = len(optimal_data['observation'])

    n_optimal = int(n_trajectories * perct_optimal)
    n_suboptimal = int(n_trajectories * (1 - perct_optimal))

    assert total_idx >= n_suboptimal, f'Pas assez de données sous-optimales. Réduisez n_trajectoires ou modifier le pourcentage de données optimales.'
    assert total_idx >= n_optimal, f'Pas assez de données optimales. Réduisez n_trajectoires ou modifier le pourcentage de données optimales.'

    optimal_idx, suboptimal_idx = random_idx_sample(n_optimal, n_suboptimal, total_idx)

    data = {}
    for key in optimal_data.keys():
        optim = optimal_data[key]
        suboptim = suboptimal_data[key]
        data[key] = torch.cat((optim[optimal_idx], suboptim[suboptimal_idx]))

    print(f'Optimal data loaded : {round(n_optimal/(n_trajectories)*100,2)}% or {n_optimal} trajectories out of {n_trajectories} trajectories')

    # {"observation", "action", "reward", 'traj_len', 'delayed_reward'}
    return data

def assign_LSTM_param_to_LSTMCell(lstm, lstmcell):
    """
    Take the weights of the trained LSTM and assign them to the LSTMCell. LSTMCell is used on PPO at each timesteps.
    :param lstm: LSTMRudder class
    :param lstmcell: LSTMCellRudder class
    """
    param_lstm = lstm.named_parameters()
    param_lstmcell = lstmcell.named_parameters()

    for (name1, weight1), (name2,weight2) in zip(param_lstm, param_lstmcell):
        assert weight1.shape == weight2.shape, f'Lstm a une dimension de {weight1.shape} alors que LSTMCell a une dimension de {weight2.shape}'
        weight1 = weight2