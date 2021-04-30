# coding=utf-8
import dataclasses

import gym
import matplotlib.pyplot as plt
import torch

from codebase import ppo

from experiment_runner.test_related_utils import check_testspec_flag_and_setup_spec
from experiment_runner.experiment_spec import PpoExperimentSpec


def main(spec: PpoExperimentSpec) -> None:
    device = "cpu"

    # Environment : CartPole-v1, MountainCar-v0, LunarLander-v2
    env = gym.make("CartPole-v1")

    steps_by_epoch = spec.steps_by_epoch
    n_epoches = spec.n_epoches
    hidden_dim = spec.hidden_dim
    n_hidden_layers = spec.n_hidden_layers

    agent, reward_logger = ppo.run_ppo(env,
                                     steps_by_epoch=steps_by_epoch,
                                     n_epoches=n_epoches,
                                     n_hidden_layers=n_hidden_layers,
                                     hidden_dim=hidden_dim,
                                     lr=0.001,
                                     save_gap=1,
                                     reward_delayed=True,
                                     device=device)

    if spec.show_plot:
        ppo.plot_agent_rewards(env_name='CartPole', reward_logger=reward_logger,
                               n_epoches=n_epoches, label='PPO baseline')
        plt.show()

    return None


if __name__ == '__main__':

    user_spec = PpoExperimentSpec(
        # steps_by_epoch=1000,
        steps_by_epoch=1000,
        # n_epoches=400,
        n_epoches=225,
        # hidden_dim=18,
        hidden_dim=18,
        # n_hidden_layers=1,
        n_hidden_layers=1,
        show_plot=True,
        n_trajectory_per_policy=1)

    test_spec = dataclasses.replace(user_spec,
                                    steps_by_epoch=1000,
                                    n_epoches=2,
                                    show_plot=False, n_trajectory_per_policy=2)

    theSpec, _ = check_testspec_flag_and_setup_spec(user_spec, test_spec)
    main(theSpec)
