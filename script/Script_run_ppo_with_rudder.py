# coding=utf-8
import dataclasses

import torch

from codebase import ppo
from codebase import rudder as rd
import matplotlib.pyplot as plt

from script.general_utils import check_testspec_flag_and_setup_spec
from script.experiment_spec import PpoExperimentSpec


def main(spec: PpoExperimentSpec) -> None:
    # device = "cpu"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Create environment
    env_name = "CartPole-v1"
    env = rd.Environment(env_name, batch_size=8, n_trajectories=3200,
                         # perct_optimal=0.7,
                         ## (Priority) todo:quick fix temporaire ‹‹‹ C'est ce qui cause l'assertion error dans les test.
                         ##                      Ça devient un field de ExperimentSpec dans mon prochain pull request
                         perct_optimal=0.5,
                         )

    steps_by_epoch = spec.steps_by_epoch
    n_epoches = spec.n_epoches
    hidden_dim = spec.hidden_dim
    n_hidden_layers = spec.n_hidden_layers


    # Initialize LSTMCell network

    hidden_size = 35
    lstmcell_rudder = rd.LstmCellRudder_with_PPO(n_states=env.n_states, n_actions=env.n_actions,
                                hidden_size=hidden_size, device=device, init_weights=True).to(device)

    # Run rudder
    agent, reward_logger = ppo.run_ppo(env.gym,
                                     lstmcell_rudder=lstmcell_rudder,
                                     steps_by_epoch=steps_by_epoch,
                                     n_epoches=n_epoches,
                                     n_hidden_layers=n_hidden_layers,
                                     hidden_dim=hidden_dim,
                                     lr=0.001,
                                     save_gap=1,
                                     device=device)

    ppo.plot_agent_rewards(env_name=env_name, reward_logger=reward_logger,
                               n_epoches=n_epoches, label='RUDDER')

    # Run PPO
    agent, reward_logger = ppo.run_ppo(env.gym,
                                     steps_by_epoch=steps_by_epoch,
                                     n_epoches=n_epoches,
                                     n_hidden_layers=n_hidden_layers,
                                     hidden_dim=hidden_dim,
                                     lr=0.001,
                                     save_gap=1,
                                     reward_delayed=True,
                                     device=device)

    if spec.show_plot:
        ppo.plot_agent_rewards(env_name=env_name, reward_logger=reward_logger,
                                   n_epoches=n_epoches, label='PPO - Delayed Rewards', alpha=1)

        plt.savefig(f'{env_name}_PPO_RUDDER.jpg')
        plt.show()
    return None


if __name__ == '__main__':

    user_spec = PpoExperimentSpec(
        steps_by_epoch=1000,
        n_epoches=75,
        hidden_dim=18,
        n_hidden_layers=1,
        show_plot=True,
        n_trajectory_per_policy=1)

    test_spec = dataclasses.replace(user_spec,
                                    steps_by_epoch=500,
                                    n_epoches=4,
                                    show_plot=False, n_trajectory_per_policy=1)

    theSpec, _ = check_testspec_flag_and_setup_spec(user_spec, test_spec)
    main(theSpec)
