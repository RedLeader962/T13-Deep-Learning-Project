# coding=utf-8
import dataclasses
import os

import torch

from codebase import ppo
from codebase import rudder as rd
import matplotlib.pyplot as plt

from experiment_runner.constant import TEST_EXPERIMENT_RUN_DIR
from experiment_runner.experiment_runner_utils import ExperimentResultsPpoRudder
from experiment_runner.test_related_utils import check_testspec_flag_and_setup_spec
from experiment_runner.experiment_spec import PpoRudderExperimentSpec


def main(spec: PpoRudderExperimentSpec) -> ExperimentResultsPpoRudder:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # device = "cpu"

    spec.setup_run_dir()

    # Create environment
    env_name = "CartPole-v1"
    env = rd.Environment(env_name,
                         batch_size=spec.env_batch_size,
                         n_trajectories=spec.env_n_trajectories,
                         perct_optimal=spec.env_perct_optimal,
                         )

    # Initialize LSTMCell network
    n_lstm_layers = 1  # Note: Hardcoded because our lstmCell implementation doesn't use 2 layers
    network = rd.LstmRudder(n_states=env.n_states,
                            n_actions=env.n_actions,
                            hidden_size=spec.hidden_dim,
                            n_lstm_layers=n_lstm_layers,
                            device=device, ).to(device)

    optimizer = torch.optim.Adam(network.parameters(), lr=spec.optimizer_lr, weight_decay=spec.optimizer_weight_decay)

    # Train LSTM
    loss_train, loss_test = rd.train_rudder(network, optimizer,
                                            n_epoches=spec.n_epoches,
                                            env=env,
                                            show_gap=1000,
                                            device=device,
                                            show_plot=spec.show_plot,
                                            print_to_consol=spec.print_to_consol,
                                            )
    lstmcell_name = f'{spec.hidden_dim}_{spec.optimizer_lr}_{spec.env_n_trajectories}_{spec.env_perct_optimal}'
    network.save_model(spec.experiment_path, lstmcell_name)
    spec.selected_lstm_model_path = os.path.join(spec.experiment_path, f"{network.file_name}_{lstmcell_name}.pt")

    lstmcell_rudder = rd.LstmCellRudder_with_PPO(n_states=env.n_states,
                                                 n_actions=env.n_actions,
                                                 hidden_size=spec.hidden_dim,
                                                 device=device,
                                                 init_weights=True).to(device)

    # Run rudder
    agent_w_rudder, reward_logger_w_rudder = ppo.run_ppo(env.gym, spec, lstmcell_rudder=lstmcell_rudder,
                                                         # hidden_dim=spec.hidden_dim,                # <-- Largeur de la couche caché
                                                         hidden_dim=18,                # <-- Largeur de la couche caché
                                                         n_hidden_layers=spec.n_hidden_layers,
                                                         lr=spec.optimizer_lr,
                                                         weight_decay=spec.optimizer_weight_decay,
                                                         n_epoches=spec.n_epoches,
                                                         steps_by_epoch=spec.steps_by_epoch,
                                                         reward_delayed=spec.reward_delayed,
                                                         rew_factor=spec.rew_factor,
                                                         save_gap=1, print_to_consol=spec.print_to_consol,
                                                         device=device)

    # Run PPO
    agent_no_rudder, reward_logger_no_rudder = ppo.run_ppo(env.gym, spec,
                                                           hidden_dim=spec.hidden_dim,
                                                           n_hidden_layers=spec.n_hidden_layers,
                                                           lr=spec.optimizer_lr,
                                                           weight_decay=spec.optimizer_weight_decay,
                                                           n_epoches=spec.n_epoches,
                                                           steps_by_epoch=spec.steps_by_epoch,
                                                           reward_delayed=spec.reward_delayed,
                                                           rew_factor=spec.rew_factor,
                                                           save_gap=1,
                                                           print_to_consol=spec.print_to_consol, device=device)

    ppo.plot_agent_rewards(env_name=env_name,
                           reward_logger=reward_logger_w_rudder,
                           n_epoches=spec.n_epoches,
                           label='RUDDER')

    ppo.plot_agent_rewards(env_name=env_name, reward_logger=reward_logger_no_rudder,
                           n_epoches=spec.n_epoches, label='PPO - Delayed Rewards', alpha=1)
    plt.savefig(os.path.join(spec.experiment_path, f'{env_name}_PPO_RUDDER.jpg'))

    if spec.show_plot:
        plt.show()

    return ExperimentResultsPpoRudder(reward_logger_w_rudder=reward_logger_w_rudder,
                                      reward_logger_no_rudder=reward_logger_no_rudder)


if __name__ == '__main__':

    user_spec = PpoRudderExperimentSpec(
        env_name='CartPole-v1',  # Environment : CartPole-v1, MountainCar-v0, LunarLander-v2
        n_epoches=5,
        steps_by_epoch=1000,
        hidden_dim=15,
        n_hidden_layers=1,
        n_trajectory_per_policy=1,
        reward_delayed=True,
        rew_factor=1.0,
        optimizer_weight_decay=0.0,
        optimizer_lr=1e-3,
        env_batch_size=8,
        env_n_trajectories=3200,
        env_perct_optimal=0.5,
        seed=42,
        show_plot=True,
        print_to_consol=True,
        experiment_tag='Manual Run',
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
                                    root_experiment_dir=TEST_EXPERIMENT_RUN_DIR,
                                    experiment_tag='Test Run',
                                    )

    theSpec, _ = check_testspec_flag_and_setup_spec(user_spec, test_spec)
    main(theSpec)
