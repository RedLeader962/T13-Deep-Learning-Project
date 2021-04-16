import numpy as np
import torch
import random
import gym
import os
import re
import math


def set_random_seed(environment, seed):
    environment.seed(seed)
    environment.action_space.seed(seed)
    environment.observation_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def cumul_discounted_rewards(rewards, gamma, device):
    """
    Takes as input a list of rewards and a gamma discount factor and returns the list of cumulated discounted rewards.
    The first item in the returned list is G_0 = r_0 + gamma * r_1 + gamma^2 * r_2 + ... + gamma^T * r_T.
    """
    G = 0
    cumul_rewards = torch.zeros_like(rewards, dtype=torch.float32, device=device)

    for i, r in enumerate(reversed(rewards)):
        G = gamma * G + r
        cumul_rewards[len(rewards) - 1 - i] = G

    return cumul_rewards


def find_most_recent_matching_network(dir, dim_in, dim_hid, dim_out):
    n_epoch = 0

    for file in os.listdir(dir):
        if file[-4:] != ".pth":
            continue
        pattern_epoch = int(re.search("epochRun_(.*?)_", file).group(1))
        pattern_dim_in = int(re.search("dimIn_(.*?)_", file).group(1)) != dim_in
        pattern_dim_out = int(re.search("dimHid_(.*?)_", file).group(1)) != dim_hid
        pattern_dim_hid = int(re.search("dimOut_(.*?)_", file).group(1)) != dim_out

        if pattern_dim_in or pattern_dim_hid or pattern_dim_out or file[0:2] != "pi":
            continue

        epoch_num = pattern_epoch
        n_epoch = max(n_epoch, epoch_num)

    return n_epoch


def file_name(dir_name, dim, network_file_name=None):
    dim_in, dim_hid, dim_out = dim

    # Create dir if not exists
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)

    epoch = find_most_recent_matching_network(dir_name, dim_in, dim_hid, dim_out)

    file_name = "dimIn_" + str(dim_in) + "_dimHid_" + str(dim_hid) + "_dimOut_" + str(dim_out) + "_epochRun_" + str(
        epoch) + "_.pth"

    network_file_name = "pi_" + file_name if network_file_name is None else network_file_name
    network_file_name = os.path.join(dir_name, network_file_name)

    return network_file_name, epoch


def file_end_epoch(info_logger, actor_critic):
    current_epoch = info_logger.current_epoch - 1
    n_epoches = info_logger.n_epoches - 1
    save_gap = info_logger.save_gap

    last_epoch = current_epoch == n_epoches
    save_time = current_epoch % info_logger.save_gap == 0

    if current_epoch == 0:
        save_gap = 0
    if last_epoch and not save_time:
        save_gap = n_epoches % save_gap

    if actor_critic.load == True:
        actor_critic.load = False
        load = 1
    else:
        load = 0

    return actor_critic.start_epoch + save_gap + load


def run_NN(environment, agent, device):
    set_random_seed(environment, seed=42)

    # On ajoute un wrapper Monitor et on écrit dans un folder demos les données et la vidéo
    env = gym.wrappers.Monitor(environment, 'demos', force=True)

    done = False

    s = torch.as_tensor(environment.reset(), dtype=torch.float32, device=device)
    rewards = 0

    while not done:
        # On rajoute un appel à render pour faire afficher les pas dans l'environnement
        env.render()
        a, _, _ = agent.step(s)
        next_s, r, done, _ = environment.step(a)

        s = torch.as_tensor(next_s, dtype=torch.float32, device=device)
        rewards += r

    print(f"Rewards for test : {rewards}")
    env.close()

def get_env_path(env : gym.Env):
    dir_name = env.unwrapped.spec.id
    root_path = os.path.relpath('../experiment/cherypicked')
    env_path  = os.path.join(root_path, dir_name)
    return env_path

def get_policies(env, optimal_policy):
    env_path  = get_env_path(env)

    if optimal_policy:
        dir = os.path.join(env_path, 'Optimal_policy')
    else:
        dir = os.path.join(env_path, 'SubOptimal_policy')

    policy_file = [os.path.join(dir, i) for i in os.listdir(dir)]

    return policy_file, env_path

def generate_trajectories(env : gym.Env, n_trajectory_per_policy : int, agent):
    """
    :param env: Gym environnme
    :param n_trajectory_per_policy: number of tajectories to be generated for each policy
    :param agent: PPO Neural network
    """
    print('Generate optimal trajectories')
    data = __generate_trajectories(env, n_trajectory_per_policy, agent, optimal_policy=True)
    save_trajectories(env, data, optimal_policy=True)

    print('Generate suboptimal trajectories')
    data = __generate_trajectories(env, n_trajectory_per_policy, agent, optimal_policy=False)
    save_trajectories(env, data, optimal_policy=False)


def __generate_trajectories(env : gym.Env, n_trajectory_per_policy : int, agent, optimal_policy : bool = True):
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

    idx_optimal = int(n_trajectories * perct_optimal)
    idx_suboptimal = int(n_trajectories * (1 - perct_optimal))

    assert total_idx >= idx_suboptimal, f'Pas assez de données sous-optimales. Réduisez n_trajectoires ou modifier le pourcentage de données optimales.'
    assert total_idx >= idx_optimal, f'Pas assez de données optimales. Réduisez n_trajectoires ou modifier le pourcentage de données optimales.'

    data = {}
    for key in optimal_data.keys():
        optim = optimal_data[key]
        suboptim = suboptimal_data[key]
        data[key] = torch.cat((optim[0:idx_optimal], suboptim[0:idx_suboptimal]))

    print(f'Optimal data loaded : {round(idx_optimal/(n_trajectories)*100,2)}% or {idx_optimal} trajectories out of {n_trajectories} trajectories')

    # {"observation", "action", "reward", 'traj_len', 'delayed_reward'}
    return data
