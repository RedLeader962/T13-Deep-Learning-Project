# coding=utf-8
import dataclasses

import gym
import matplotlib.pyplot as plt
import torch

from codebase import ppo

from script.general_utils import check_testspec_flag_and_setup_spec
from script.experiment_spec import PpoExperimentSpec


def main(spec: PpoExperimentSpec) -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Environment : CartPole-v1, MountainCar-v0, LunarLander-v2
    environment = gym.make("CartPole-v1")

    steps_by_epoch = spec.steps_by_epoch
    n_epoches = spec.n_epoches
    hidden_dim = spec.hidden_dim
    n_hidden_layers = spec.n_hidden_layers

    agent, reward_logger = ppo.run_ppo(environment,
                                     steps_by_epoch=steps_by_epoch,
                                     n_epoches=n_epoches,
                                     n_hidden_layers=n_hidden_layers,
                                     hidden_dim=hidden_dim,
                                     lr=0.001,
                                     save_gap=1,
                                     device=device)

    if spec.show_plot:
        plt.title(f"PPO - Number of epoches : {n_epoches} and steps by epoch : {steps_by_epoch}")
        plt.plot(reward_logger, label='E_average_return')
        plt.legend()
        plt.xlabel("Epoches")
        plt.show()

    return None


if __name__ == '__main__':

    user_spec = PpoExperimentSpec(
        # steps_by_epoch=1000,
        steps_by_epoch=1000,
        # n_epoches=400,
        n_epoches=25,
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
