from .utils import set_random_seed
from .ppo_nn import NnActorCritic
from .ppo_experience_buffer import PpoBuffer
from codebase.logger.log_epoch import EpochsLogger

import torch
import numpy as np


def run_ppo(env,
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
    agent.load_model(env)

    # Initialize experience replay
    replay_buffer = PpoBuffer(steps_by_epoch, state_size, device)

    # Track trajectories info
    info_logger = EpochsLogger(n_epoches, save_gap=save_gap)

    # Track reward
    rewards_epoches_logger = []

    for epoch in range(n_epoches):

        s = torch.tensor(env.reset(), dtype=torch.float32, device=device)
        reward_logger = 0
        reward_tracker = []
        episode_tracker = 0

        for t in range(steps_by_epoch):

            # Select action according to policy pi(a|s)
            a, v, log_prob_a = agent.step(s)

            # Obtain rewards r and observe next state s
            s_next, r, trajectory_done, _ = env.step(a)
            #r = np.tanh(r)

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

        # Save models
        agent.save_model_data(info_logger)

        print(f'Epoch {epoch} :  e_avg_return: {reward_mean:.2f}, loss_pi = {loss_pi:.4f}, loss_v = {loss_v:.2f}, '
              f'n_traject : {episode_tracker}')

        rewards_epoches_logger.append(reward_mean)

    return agent, rewards_epoches_logger
