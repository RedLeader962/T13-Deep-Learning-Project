# coding=utf-8


def test_run_ppo_and_load_data_PASS():
    import gym
    from codebase import ppo

    # Environment : CartPole-v1, MountainCar-v0, LunarLander-v2
    environment = gym.make("CartPole-v1")

    agent, info_logger = ppo.run_ppo(environment, hidden_dim=6, n_hidden_layers=2, lr=0.01, n_epoches=2,
                                     steps_by_epoch=1000, save_gap=1, seed=42, device="cpu")

    dir_name = environment.unwrapped.spec.id
    dim_NN = environment.observation_space.shape[0], 6, environment.action_space.n
    # data = info_logger.load_data(dir_name, dim_NN)
