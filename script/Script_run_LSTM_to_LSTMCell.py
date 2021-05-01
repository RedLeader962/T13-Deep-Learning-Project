import dataclasses

import torch
from torch.utils.data import DataLoader
import numpy as np
import gym

from codebase import rudder as rd
from script.general_utils import check_testspec_flag_and_setup_spec
from script.experiment_spec import RudderExperimentSpec


def main(spec: RudderExperimentSpec) -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    lr = 0.01
    n_trajectories = 200
    percet_optimal = 0.9

    # Create environment
    env = rd.Environment("CartPole-v1", batch_size=8, n_trajectories=n_trajectories, perct_optimal=percet_optimal)

    # Create LSTM Network
    n_lstm_layers = 1
    hidden_size = 10
    lstm = rd.LstmRudder(n_states=env.n_states, n_actions=env.n_actions,
                            hidden_size=hidden_size, n_lstm_layers=n_lstm_layers, device=device).to(device)

    optimizer = torch.optim.Adam(lstm.parameters(), lr=lr, weight_decay=1e-4)

    # Train and save LSTM in the gym environnement
    rd.train_rudder(lstm, optimizer, n_epoches=spec.n_epoches, env=env, show_gap=100, device=device,
                    show_plot=spec.show_plot)

    lstm.save_model(env.gym, f'{hidden_size}_{lr}_{n_trajectories}_{percet_optimal}')

    # Create LSTMCell Network
    lstmcell = rd.LstmCellRudder(n_states=env.n_states, n_actions=env.n_actions, hidden_size=hidden_size,
                                 device=device, init_weights=False).to(device)

    # Load LSTMCell
    lstmcell.load_lstm_model(env.gym, f'{hidden_size}_{lr}_{n_trajectories}_{percet_optimal}')

    # Train LSTMCell
    optimizer = torch.optim.Adam(lstmcell.parameters(), lr=1e-3, weight_decay=1e-4)
    rd.train_rudder(lstmcell, optimizer, n_epoches=spec.n_epoches, env=env, show_gap=100, device=device,
                    show_plot=spec.show_plot)


if __name__ == '__main__':

    user_spec = RudderExperimentSpec(
        n_epoches=1,
        env_batch_size=1000,
        loader_batch_size=4,
        show_plot=True,
        )

    test_spec = dataclasses.replace(user_spec,
                                    n_epoches=2,
                                    env_batch_size=20,
                                    show_plot=False,
                                    )

    theSpec, _ = check_testspec_flag_and_setup_spec(user_spec, test_spec)
    main(theSpec)
