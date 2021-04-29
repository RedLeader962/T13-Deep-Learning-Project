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

    # Prepare some random generators for later
    rnd_gen = np.random.RandomState(seed=123)
    _ = torch.manual_seed(123)

    # Create environment
    n_positions = 13
    env = rd.Environment("CartPole-v1", n_trajectories=spec.env_batch_size, max_timestep=50, n_positions=13, rnd_gen=rnd_gen)

    # Load data
    env_loader = DataLoader(env, batch_size=spec.loader_batch_size)

    # Create LSTM Network
    n_lstm_layers = 1
    hidden_size = 40
    lstm = rd.LstmRudder(n_positions=n_positions, n_actions=2,
                         hidden_size=hidden_size, n_lstm_layers=n_lstm_layers, device=device).to(device)

    optimizer = torch.optim.Adam(lstm.parameters(), lr=1e-3, weight_decay=1e-4)

    # Train and save LSTM in the gym environnement
    rd.train_rudder(lstm, optimizer, n_epoches=spec.n_epoches, env=env, show_gap=100, device=device,
                    show_plot=spec.show_plot)
    lstm.save_model(env.gym)

    # Create LSTMCell Network
    lstmcell = rd.LstmCellRudder(n_positions=n_positions, n_actions=2, hidden_size=hidden_size,
                                 device=device, init_weights=False).to(device)

    # Load LSTMCell
    lstmcell.load_lstm_model(env.gym)

    # Train LSTMCell
    optimizer = torch.optim.Adam(lstmcell.parameters(), lr=1e-3, weight_decay=1e-4)
    rd.train_rudder(lstmcell, optimizer, n_epoches=spec.n_epoches, env=env, show_gap=100, device=device,
                    show_plot=spec.show_plot)


if __name__ == '__main__':

    user_spec = RudderExperimentSpec(
        n_epoches=5,
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
