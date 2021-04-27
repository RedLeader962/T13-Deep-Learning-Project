import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import gym

TRAJECTORIES_OPTIMAL = 'trajectories_optimal'
TRAJECTORIES_SUBOPTIMAL = 'trajectories_suboptimal'


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
    root_path = os.path.relpath('experiment/cherypicked')
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
        myCoolDir = os.path.join(env_path, 'Optimal_policy')
    else:
        myCoolDir = os.path.join(env_path, 'SubOptimal_policy')

    print(os.listdir(myCoolDir))
    policy_file = [os.path.join(myCoolDir, i) for i in os.listdir(myCoolDir)]

    return policy_file, env_path

def generate_trajectories(env : gym.Env, n_trajectory_per_policy : int, agent):
    """
    :param env: Gym environnment
    :param n_trajectory_per_policy: number of tajectories to be generated for each policy
    :param agent: PPO Neural network
    """
    data = _generate_trajectories(env, n_trajectory_per_policy, agent, optimal_policy=True)
    save_data(env, data, TRAJECTORIES_OPTIMAL)

    data = _generate_trajectories(env, n_trajectory_per_policy, agent, optimal_policy=False)
    save_data(env, data, TRAJECTORIES_SUBOPTIMAL)

def generate_discete_env_single_episode(env, agent, max_episode_length):
    observation = torch.zeros((max_episode_length, agent.state_dim), device=agent.device)
    action = torch.zeros((max_episode_length, agent.action_dim), device=agent.device)
    reward = torch.zeros(max_episode_length, device=agent.device)

    t_step = 0
    reward_count = 0
    done = False
    s = torch.as_tensor(env.reset(), dtype=torch.float32)

    observation[t_step] = s

    while not done:
        a, _, _ = agent.step(s)
        next_s, r, done, _ = env.step(a)

        # Log state, actions and reward
        observation[t_step] = s
        action[t_step, a] = 1

        # Correction to avoid rewards to big
        #r = np.tanh(r)

        reward[t_step] = r
        reward_count += r

        s = torch.as_tensor(next_s, dtype=torch.float32)

        # Next time step
        t_step += 1

    return observation, action, reward, reward_count, t_step


def _generate_trajectories(env : gym.Env, n_trajectory_per_policy : int, agent, optimal_policy : bool):
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
            if t_step == max_episode_length :
                t_step -= 1
            delayed_rewards[i, T, t_step] = delayed_r

        # Save the trajectory should go here
        print(f'    Policy {i+1}/{n_policies} trajectory generated 100%')

    return {"observation": observations, "action": actions, "reward": rewards, 'traj_len': trajectory_length, 'delayed_reward': delayed_rewards}

def save_data(env, data, file_name):
    """
    :param env: Gym environnment
    :param data: Dataset of trajectories
    :param file_name: name of the file
    """
    env_path = get_env_path(env)
    file_path = os.path.join(env_path, f'{file_name}.pt')
    torch.save(data, file_path)
    print(file_name, 'saved in', env_path)

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

    optimal_data = torch.load(os.path.join(env_path, f'{TRAJECTORIES_OPTIMAL}.pt'))
    suboptimal_data = torch.load(os.path.join(env_path, f'{TRAJECTORIES_SUBOPTIMAL}.pt'))

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

def plot_lstm_loss(loss_train, loss_test):
    plt.rcParams.update({"font.size": 18, "font.family": "sans-serif", "figure.figsize": (8, 6)})

    plt.title(f"LSTM Loss")
    plt.plot(loss_train, label='Train loss', linewidth=2.5)
    plt.plot(loss_test, label='Test loss', linewidth=2.5)
    plt.legend()
    plt.xlabel("Epoches")
    plt.ylabel('Loss')
    plt.show()