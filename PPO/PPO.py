from .utils import *
from .PPO_NN import NN_ActorCritic
from .PPO_Experience_Buffer import PPO_Buffer
from .Logger import Info_logger

import torch

def PPO(env,
        gamma = 0.99,
        lr=1e-3,
        seed=42,
        n_epoches=1000,
        steps_by_epoch=4000,
        lam=0.97,
        target_kl = 0.015,
        max_train_iters=80,
        n_hidden_layers=1,
        hidden_dim=16,
        device='cpu'):

    set_random_seed(env, seed)

    state_size  = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Initialize agent network
    agent  = NN_ActorCritic(state_size,
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
    replay_buffer = PPO_Buffer(steps_by_epoch, state_size, device)

    # Track trajectories info
    info_logger = Info_logger(n_epoches, print=True)

    episode_tracker = 0
    delay = 1

    for epoch in range(n_epoches):

        s = torch.tensor(env.reset(), dtype=torch.float32, device=device)

        for t in range(steps_by_epoch):

            # Select action according to policy pi(a|s)
            a, v, log_prob_a = agent.step(s)

            # Obtain rewards r and observe next state s
            s_next, r, trajectory_done, _ = env.step(a)
            #r = np.tanh(r* np.pi/100)

            s_next = torch.tensor(s_next, dtype=torch.float32, device=device)

            # Store information in buffer
            replay_buffer.store(s, a, r, s_next, trajectory_done, v, log_prob_a)

            s = s_next

            # log rewards
            info_logger.log_rewards(r)

            episode_tracker += 1

            if trajectory_done or t == steps_by_epoch - 1:
                episode_tracker = 0

                # If trajectory not done, bootstrap
                if not trajectory_done:
                    _, last_v, _ = agent.step(s)
                else:
                    last_v = torch.tensor([0], dtype=torch.float32, device=device)
                    info_logger.log_traj()

                replay_buffer.epoch_ended(last_v, gamma, lam)
                s = torch.tensor(env.reset(), dtype=torch.float32, device=device)

        data = replay_buffer.get_trajectories()

        # Update critic v and actor policy
        loss_v = agent.update_critic(data)
        loss_pi, approx_KL = agent.update_actor(data, target_kl)

        # Compute and store information
        info_logger.end_epoch(loss_v, loss_pi)

        # Save models
        agent.save_model_data(info_logger)

    return agent, info_logger


