# coding=utf-8

import pytest

from script.general_utils import show_plot_if_not_a_CI_server_run


def test_run_ppo_and_load_data_PASS():
    import gym
    from codebase import ppo

    # Environment : CartPole-v1, MountainCar-v0, LunarLander-v2
    environment = gym.make("CartPole-v1")

    agent, info_logger = ppo.run_ppo(environment,
                                     seed=42,
                                     steps_by_epoch=1000,
                                     n_epoches=2,
                                     n_hidden_layers=2,
                                     hidden_dim=6,
                                     lr=0.01,
                                     save_gap=1,
                                     device="cpu")

    dir_name = environment.unwrapped.spec.id
    dim_NN = environment.observation_space.shape[0], 6, environment.action_space.n
    data = info_logger.load_data(dir_name, dim_NN)


def test_Script_run_ppo_main_PASS():
    from script.Script_run_ppo import main, PpoExperimentSpec

    test_spec = PpoExperimentSpec(
        steps_by_epoch=150,
        n_epoches=2,
        hidden_dim=6,
        n_hidden_layers=2,
        device="cpu",
        show_plot=show_plot_if_not_a_CI_server_run(False),
        n_trajectory_per_policy=6)

    main(test_spec)
