# coding=utf-8
import dataclasses

import gym
import matplotlib.pyplot as plt
import torch

from codebase import ppo

from experiment_runner.test_related_utils import check_testspec_flag_and_setup_spec
from experiment_runner.experiment_spec import PpoExperimentSpec


def main(spec: PpoExperimentSpec) -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    env = gym.make(spec.env_name)

    agent, reward_logger = ppo.run_ppo(env,
                                       hidden_dim=spec.hidden_dim,
                                       n_hidden_layers=spec.n_hidden_layers,
                                       lr=spec.optimizer_lr,
                                       weight_decay=spec.optimizer_weight_decay,
                                       n_epoches=spec.n_epoches,
                                       steps_by_epoch=spec.steps_by_epoch,
                                       reward_delayed=spec.reward_delayed,
                                       rew_factor=spec.rew_factor,
                                       save_gap=1,
                                       device=device,
                                       print_to_consol=spec.print_to_consol,
                                       )

    if spec.show_plot:
        ppo.plot_agent_rewards(env_name=spec.env_name,
                               reward_logger=reward_logger,
                               n_epoches=spec.n_epoches,
                               label='PPO baseline')
        plt.show()

    return None


if __name__ == '__main__':

    user_spec = PpoExperimentSpec(
        env_name='CartPole-v1',  # Environment : CartPole-v1, MountainCar-v0, LunarLander-v2
        steps_by_epoch=1000,
        n_epoches=225,
        hidden_dim=18,
        n_hidden_layers=1,
        n_trajectory_per_policy=1,
        reward_delayed=True,
        rew_factor=1.0,
        optimizer_weight_decay=0.0,
        optimizer_lr=1e-3,
        seed=42,
        show_plot=True,
        print_to_consol=True,
        )

    test_spec = dataclasses.replace(user_spec,
                                    steps_by_epoch=500,
                                    n_epoches=2,
                                    show_plot=False,
                                    n_trajectory_per_policy=2,
                                    print_to_consol=False,
                                    )

    theSpec, _ = check_testspec_flag_and_setup_spec(user_spec, test_spec)
    main(theSpec)
