import dataclasses
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
import numpy as np

from codebase import rudder as rd
from script.general_utils import ExperimentSpec, check_testspec_flag_and_setup_spec


@dataclass(frozen=True)
class RudderExperimentSpec(ExperimentSpec):
    n_epoches: int

def main(spec: RudderExperimentSpec) -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Prepare some random generators for later
    rnd_gen = np.random.RandomState(seed=123)
    _ = torch.manual_seed(123)

    # Create environment
    n_positions = 13
    env = rd.Environment(batch_size=1000, max_timestep=50, n_positions=13, rnd_gen=rnd_gen)

    # Load data
    batch_size = 8
    env_loader = DataLoader(env, batch_size=batch_size)

    # Create LSTM Network
    n_lstm_layers = 1
    hidden_size = 15
    lstm = rd.LstmRudder(n_positions=n_positions, n_actions=2,
                            hidden_size=hidden_size, n_lstm_layers=n_lstm_layers, device=device).to(device)

    optimizer = torch.optim.Adam(lstm.parameters(), lr=1e-3, weight_decay=1e-4)

    # Train LSTM
    rd.train_rudder(lstm, optimizer, epoches=10, data_loader=env_loader, show_gap=100, device=device,
                    show_plot=spec.show_plot)

    # Create LSTMCell Network
    lstmcell = rd.LstmCellRudder(n_positions=n_positions, n_actions=2,
                            hidden_size=hidden_size, n_lstm_layers=n_lstm_layers, device=device, train=False).to(device)

    # Assign weights
    rd.assign_LSTM_param_to_LSTMCell(lstm=lstm, lstmcell=lstmcell)
    optimizer = torch.optim.Adam(lstmcell.parameters(), lr=1e-3, weight_decay=1e-4)

    # Train LSTMCell
    rd.train_rudder(lstmcell, optimizer, epoches=30, data_loader=env_loader, show_gap=10, device=device,
                    show_plot=spec.show_plot)


if __name__ == '__main__':

    user_spec = RudderExperimentSpec(
        n_epoches=2000,
        show_plot=True,
        )

    test_spec = dataclasses.replace(user_spec, show_plot=False, n_epoches=2)

    theSpec, _ = check_testspec_flag_and_setup_spec(user_spec, test_spec)
    main(theSpec)
