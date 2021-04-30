# coding=utf-8
import dataclasses

import torch

from codebase import ppo
from codebase import rudder as rd
import numpy as np


from script.general_utils import check_testspec_flag_and_setup_spec
from script.experiment_spec import PpoExperimentSpec


def main(spec: PpoExperimentSpec) -> None:
    device = "cpu"

    # Create environment
    env = rd.Environment("CartPole-v1", batch_size=8, n_trajectories=3200, perct_optimal=0.7)

    steps_by_epoch = spec.steps_by_epoch
    n_epoches = spec.n_epoches
    hidden_dim = spec.hidden_dim
    n_hidden_layers = spec.n_hidden_layers


    # Initialize LSTMCell network

    hidden_size = 40
    lstmcell_rudder = rd.LstmCellRudder_with_PPO(n_states=env.n_states, n_actions=env.n_actions,
                                hidden_size=hidden_size, device=device, init_weights=True).to(device)

    # Initialize agent
    agent, info_logger = ppo.run_ppo(env.gym,
                                     lstmcell_rudder=lstmcell_rudder,
                                     steps_by_epoch=steps_by_epoch,
                                     n_epoches=n_epoches,
                                     n_hidden_layers=n_hidden_layers,
                                     hidden_dim=hidden_dim,
                                     lr=0.001,
                                     save_gap=1,
                                     device=device)


    return None


if __name__ == '__main__':

    user_spec = PpoExperimentSpec(
        steps_by_epoch=1000,
        n_epoches=200,
        hidden_dim=18,
        n_hidden_layers=1,
        show_plot=True,
        n_trajectory_per_policy=1)

    test_spec = dataclasses.replace(user_spec,
                                    steps_by_epoch=100,
                                    n_epoches=4,
                                    show_plot=False, n_trajectory_per_policy=1)

    theSpec, _ = check_testspec_flag_and_setup_spec(user_spec, test_spec)
    main(theSpec)
