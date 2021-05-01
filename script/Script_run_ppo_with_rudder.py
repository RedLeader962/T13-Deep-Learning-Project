# coding=utf-8
import dataclasses

import torch

from codebase import ppo
from codebase import rudder as rd
import matplotlib.pyplot as plt

from experiment_runner.test_related_utils import check_testspec_flag_and_setup_spec
from experiment_runner.experiment_spec import PpoRudderExperimentSpec


def main(spec: PpoRudderExperimentSpec) -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Create environment
    env_name = "CartPole-v1"
    env = rd.Environment(env_name,
                         batch_size=spec.env_batch_size,
                         n_trajectories=spec.env_n_trajectories,
                         perct_optimal=spec.env_perct_optimal,
                         )

    # Initialize LSTMCell network

    lstmcell_rudder = rd.LstmCellRudder_with_PPO(n_states=env.n_states,
                                                 n_actions=env.n_actions,
                                                 hidden_size=(spec.rudder_hidden_size),
                                                 device=device,
                                                 init_weights=True).to(device)

    # Run rudder
    agent_w_rudder, reward_logger_w_rudder = ppo.run_ppo(env.gym, lstmcell_rudder=lstmcell_rudder,
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

    ppo.plot_agent_rewards(env_name=env_name,
                           reward_logger=reward_logger_w_rudder,
                           n_epoches=spec.n_epoches,
                           label='RUDDER')

    # Run PPO
    agent_no_rudder, reward_logger_no_rudder = ppo.run_ppo(env.gym,
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
        ppo.plot_agent_rewards(env_name=env_name, reward_logger=reward_logger_no_rudder,
                               n_epoches=spec.n_epoches, label='PPO - Delayed Rewards', alpha=1)

        plt.savefig(f'{env_name}_PPO_RUDDER.jpg')
        plt.show()
    return None


if __name__ == '__main__':

    user_spec = PpoRudderExperimentSpec(
        env_name='CartPole-v1',  # Environment : CartPole-v1, MountainCar-v0, LunarLander-v2
        n_epoches=75,
        steps_by_epoch=1000,
        hidden_dim=18,
        n_hidden_layers=1,
        n_trajectory_per_policy=1,
        reward_delayed=True,
        rew_factor=1.0,
        optimizer_weight_decay=0.0,
        optimizer_lr=1e-3,
        rudder_hidden_size=35,
        env_batch_size=8,
        env_n_trajectories=3200,
        env_perct_optimal=0.7,
        seed=42,
        show_plot=True,
        print_to_consol=True,
        )

    test_spec = dataclasses.replace(user_spec,
                                    steps_by_epoch=500,
                                    n_epoches=4,
                                    env_batch_size=8,
                                    env_n_trajectories=300,
                                    env_perct_optimal=0.5,
                                    show_plot=False,
                                    n_trajectory_per_policy=1,
                                    print_to_consol=False,
                                    )

    theSpec, _ = check_testspec_flag_and_setup_spec(user_spec, test_spec)
    main(theSpec)
