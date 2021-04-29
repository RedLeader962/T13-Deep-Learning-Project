from .utils import set_random_seed, plot_angle_reward_ppo, plot_loss_PPO
from .ppo_nn import NnActorCritic
from .ppo_experience_buffer import PpoBuffer
from codebase.logger.log_epoch import EpochsLogger
from codebase.rudder.lstmcell_RUDDER import LstmCellRudder

import torch
import numpy as np


def run_ppo(env,
            lstmcell_rudder: LstmCellRudder = None,
            gamma=0.99,
            lr=1e-3,
            seed=42,
            n_epoches=1000,
            steps_by_epoch=4000,
            lam=0.97,
            target_kl=0.015,
            max_train_iters=80,
            n_hidden_layers=1,
            hidden_dim=16,
            save_gap=5,
            device='cpu'):
    set_random_seed(env, seed)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    show_plot = 10
    # Initialize agent network
    agent = NnActorCritic(state_size,
                          action_size,
                          lr=lr,
                          target_kl=target_kl,
                          max_train_pi_iters=max_train_iters,
                          n_hidden_layers=n_hidden_layers,
                          hidden_dim=hidden_dim,
                          device=device)

    # Load model
    #agent.load_model(env)

    # Load LSTMCell model from LSTM (Rudder)
    if lstmcell_rudder is not None:
        lstmcell_rudder.load_model(env)

    # Initialize experience replay
    replay_buffer = PpoBuffer(steps_by_epoch, state_size, device=device, lstmcell_rudder=lstmcell_rudder)

    # Track trajectories info
    info_logger = EpochsLogger(n_epoches, save_gap=save_gap)

    # Track reward
    rewards_epoches_logger = []
    angle_list = []
    reward_list = []
    loss_v_list = []
    loss_pi_list = []
    for epoch in range(n_epoches):

        s = torch.tensor(env.reset(), dtype=torch.float32, device=device)
        reward_logger = 0
        reward_tracker = []
        episode_tracker = 0

        # Reset hidden
        if lstmcell_rudder is not None:
            lstmcell_rudder.reset_cell_hidden_state()

        for t in range(steps_by_epoch):

            # Select action according to policy pi(a|s)
            a, v, log_prob_a = agent.step(s)

            # Obtain rewards r and observe next state s
            s_next, r, trajectory_done, _ = env.step(a)
            #r = np.tanh(r)
            angle_list.append(s_next[2])
            reward_list.append(r)
            s_next = torch.tensor(s_next, dtype=torch.float32, device=device)

            # Store information in buffer
            replay_buffer.store(s, a, r, s_next, trajectory_done, v, log_prob_a)

            s = s_next

            # log rewards
            reward_logger += r

            if trajectory_done or t == steps_by_epoch - 1:

                # If trajectory not done, bootstrap
                if not trajectory_done:
                    _, last_v, _ = agent.step(s)
                else:
                    reward_tracker.append(reward_logger)
                    last_v = torch.tensor([0], dtype=torch.float32, device=device)
                    episode_tracker += 1
                    reward_logger = 0

                replay_buffer.epoch_ended(last_v, gamma, lam)
                s = torch.tensor(env.reset(), dtype=torch.float32, device=device)

        reward_mean = sum(reward_tracker)/ episode_tracker

        trajectories_data = replay_buffer.get_trajectories()

        # Update critic v and actor policy
        loss_v = agent.update_critic(trajectories_data)
        loss_pi, approx_KL = agent.update_actor(trajectories_data, target_kl)
        loss_v_list.append(loss_v.item())
        loss_pi_list.append(loss_pi.item())
        # Save models
        #agent.save_model_data(info_logger)
        #When CartPole env is use plot the reward by angle every time the condition is respected
        if env.env.spec.id == 'CartPole-v1' and epoch % show_plot == 0  and epoch != 0 :
         plot_angle_reward_ppo(angle_list,reward_list,epoch)

        print(f'Epoch {epoch} :  e_avg_return: {reward_mean:.2f}, loss_pi = {loss_pi:.4f}, loss_v = {loss_v:.2f}, '
              f'n_traject : {episode_tracker}')
        plot_loss_PPO(loss_v_list,loss_pi_list)
        rewards_epoches_logger.append(reward_mean)

    return agent, rewards_epoches_logger
