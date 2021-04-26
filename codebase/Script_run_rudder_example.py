import dataclasses
from dataclasses import dataclass

import torch
import numpy as np

from codebase import rudder as rd
from codebase.general_utils import ExperimentSpec, check_testspec_flag_and_setup_spec


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
    env = rd.Environment(batch_size=2000, max_timestep=50, n_positions=13, rnd_gen=rnd_gen)

    # Load data
    batch_size = 10
    env_loader = torch.utils.data.DataLoader(env, batch_size=batch_size)

    # Create Network
    n_lstm_layers = 1
    hidden_size = 40
    network = rd.LstmRudder(n_positions=n_positions, n_actions=2,
                            hidden_size=hidden_size, n_lstm_layers=n_lstm_layers
                            ).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3, weight_decay=1e-4)

    # Train LSTM
    rd.train_rudder(network, optimizer, epoches=spec.n_epoches, data_loader=env_loader, show_gap=3, device=device,
                    show_plot=spec.show_plot)


if __name__ == '__main__':

    user_spec = RudderExperimentSpec(
        n_epoches=6,
        show_plot=True,
        )

    test_spec = dataclasses.replace(user_spec, show_plot=False, n_epoches=2)

    theSpec, _ = check_testspec_flag_and_setup_spec(user_spec, test_spec)
    main(theSpec)
