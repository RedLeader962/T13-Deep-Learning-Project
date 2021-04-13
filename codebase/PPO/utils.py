import numpy as np
import torch
import random
import gym
import os
import re

def set_random_seed(environment, seed):
    environment.seed(seed)
    environment.action_space.seed(seed)
    environment.observation_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

def cumul_discounted_rewards(rewards, gamma, device):
    '''
    Takes as input a list of rewards and a gamma discount factor and returns the list of cumulated discounted rewards.
    The first item in the returned list is G_0 = r_0 + gamma * r_1 + gamma^2 * r_2 + ... + gamma^T * r_T.
    '''
    G = 0
    cumul_rewards = torch.zeros_like(rewards, dtype=torch.float32, device=device)

    for i, r in enumerate(reversed(rewards)):
        G = gamma * G + r
        cumul_rewards[len(rewards) - 1 - i] = G

    return cumul_rewards

def find_most_recent_matching_network(dir, dim_in, dim_hid, dim_out):
    n_epoch = 0

    for file in os.listdir(dir):
        if file[-4:] != ".pth" : continue
        pattern_epoch  = int(re.search("epochRun_(.*?)_", file).group(1))
        pattern_dim_in = int(re.search("dimIn_(.*?)_", file).group(1)) != dim_in
        pattern_dim_out = int(re.search("dimHid_(.*?)_", file).group(1)) != dim_hid
        pattern_dim_hid = int(re.search("dimOut_(.*?)_", file).group(1)) != dim_out

        if pattern_dim_in or pattern_dim_hid or pattern_dim_out or file[0:2] != "pi":
            continue

        epoch_num = pattern_epoch
        n_epoch   = max(n_epoch, epoch_num)

    return n_epoch

def file_name(dir_name, dim, network_file_name=None):
    dim_in, dim_hid, dim_out = dim

    # Create dir if not exists
    if not os.path.isdir(dir_name): os.mkdir(dir_name)

    epoch = find_most_recent_matching_network(dir_name, dim_in, dim_hid, dim_out)

    file_name = "dimIn_"+str(dim_in)+"_dimHid_"+str(dim_hid)+"_dimOut_"+str(dim_out)+"_epochRun_"+str(epoch)+"_.pth"

    network_file_name = "pi_" + file_name if network_file_name is None else network_file_name
    network_file_name = os.path.join(dir_name, network_file_name)

    return network_file_name, epoch

def file_end_epoch(info_logger, actor_critic):

    current_epoch = info_logger.current_epoch - 1
    n_epoches = info_logger.n_epoches - 1
    save_gap = info_logger.save_gap

    last_epoch = current_epoch == n_epoches
    save_time = current_epoch % info_logger.save_gap == 0

    if current_epoch == 0: save_gap = 0
    if last_epoch and not save_time: save_gap = n_epoches % save_gap

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

def get_policies(env):
    dir_name = env.unwrapped.spec.id
    print(os.getcwd())
    root_path = os.path.relpath('../experiment/cherypicked')

    optimal_dir = os.path.join(root_path, os.path.join(dir_name, 'Optimal_policy'))
    suboptimal_dir = os.path.join(root_path, os.path.join(dir_name, 'SubOptimal_policy'))

    optimal_policy_file = [os.path.join(optimal_dir, i) for i in os.listdir(optimal_dir)]
    suboptimal_policy_file = [os.path.join(optimal_dir, i) for i in os.listdir(suboptimal_dir)]

    return optimal_policy_file, suboptimal_policy_file

def generate_trajectories(env, n_trajectory_per_policy, agent, optimal_policy=True, device='cpu'):

    max_episode_length = env.spec.max_episode_steps

    optimal_policy_name_list, suboptimal_policy_name_list = get_policies(env)
    policies = optimal_policy_name_list if optimal_policy == True else suboptimal_policy_name_list

    for i, policy in enumerate(policies):

        # Load policy
        agent.load_state_dict(torch.load(policy))

        # Track observations, actions, rewards for each policy
        observations = torch.zeros((n_trajectory_per_policy, max_episode_length, agent.state_dim))
        actions = torch.zeros((n_trajectory_per_policy, max_episode_length, agent.action_dim))
        rewards = torch.zeros((n_trajectory_per_policy, max_episode_length))

        # Generate trajectories
        for T in range(n_trajectory_per_policy):

            t_step = 0
            done = False
            s = torch.as_tensor(env.reset(), dtype=torch.float32, device=device)

            observations[T,0] = s

            while not done:
                a, _, _ = agent.step(s)
                next_s, r, done, _ = env.step(a)

                # Log state, actions and reward
                observations[T, t_step] = s
                actions[T, t_step, a] = 1
                rewards[T, t_step] = r

                s = torch.as_tensor(next_s, dtype=torch.float32, device=device)

                t_step += 1

        # Save the trajectory should go here
        print(f'Policy {i+1}/{len(policies)} trajectory generated 100%')



