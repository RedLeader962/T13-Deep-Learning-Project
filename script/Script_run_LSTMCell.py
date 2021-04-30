import dataclasses

import torch

from codebase import rudder as rd
from experiment_runner.test_related_utils import check_testspec_flag_and_setup_spec
from experiment_runner.experiment_spec import RudderExperimentSpec


def main(spec: RudderExperimentSpec) -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Create environment
    env = rd.Environment("CartPole-v1", batch_size=8, n_trajectories=500, perct_optimal=0.7)

    # Create Network
    n_lstm_layers = 1
    hidden_size = 40
    network = rd.LstmCellRudder(n_states=env.n_states, n_actions=env.n_actions,
                                hidden_size=hidden_size, device=device, init_weights=True).to(device)

    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3, weight_decay=1e-2)

    # Train LSTM
    rd.train_rudder(network, optimizer, n_epoches=spec.n_epoches, env=env, show_gap=100, device=device,
                    show_plot=spec.show_plot)


if __name__ == '__main__':

    user_spec = RudderExperimentSpec(
        n_epoches=2,
        env_batch_size=1000,
        loader_batch_size=8,
        show_plot=True,
        )

    test_spec = dataclasses.replace(user_spec,
                                    n_epoches=2,
                                    env_batch_size=20,
                                    show_plot=False,
                                    )

    theSpec, _ = check_testspec_flag_and_setup_spec(user_spec, test_spec)
    main(theSpec)
