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

def file_name(dir_name, dim, pi_file_name=None, v_file_name=None):
    dim_in, dim_hid, dim_out = dim

    # Create dir if not exists
    if not os.path.isdir(dir_name): os.mkdir(dir_name)

    epoch = find_most_recent_matching_network(dir_name, dim_in, dim_hid, dim_out)

    file_name = "dimIn_"+str(dim_in)+"_dimHid_"+str(dim_hid)+"_dimOut_"+str(dim_out)+"_epochRun_"+str(epoch)+"_.pth"

    pi_file_name = "pi_" + file_name if pi_file_name is None else pi_file_name
    v_file_name = "v_" + file_name if v_file_name is None else v_file_name

    return pi_file_name, v_file_name, epoch

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

