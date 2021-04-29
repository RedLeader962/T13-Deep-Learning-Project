import dataclasses

import torch
from torch.utils.data import DataLoader
import numpy as np

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

    # Create Network
    n_lstm_layers = 1
    hidden_size = 40
    network = rd.LstmRudder(n_positions=n_positions, n_actions=2,
                            hidden_size=hidden_size, n_lstm_layers=n_lstm_layers, device=device).to(device)

    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3, weight_decay=1e-2)

    # Train LSTM
    loss_train, loss_test = rd.train_rudder(network, optimizer, n_epoches=spec.n_epoches, env=env, show_gap=100, device=device,
                    show_plot=spec.show_plot)

    not_show = False
    if not_show:
        rd.plot_lstm_loss(loss_train=loss_train, loss_test=loss_test)

if __name__ == '__main__':

    user_spec = RudderExperimentSpec(
        n_epoches=40,
        env_batch_size=1500,
        loader_batch_size=8,
        show_plot=True,
        )

    test_spec = dataclasses.replace(user_spec,
                                    n_epoches=2,
                                    env_batch_size=8,
                                    show_plot=False,
                                    )

    theSpec, _ = check_testspec_flag_and_setup_spec(user_spec, test_spec)
    main(theSpec)
