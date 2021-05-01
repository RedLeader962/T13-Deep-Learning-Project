import dataclasses

import torch
import numpy as np

from codebase import rudder as rd
from script.general_utils import check_testspec_flag_and_setup_spec
from script.experiment_spec import RudderExperimentSpec


def main(spec: RudderExperimentSpec) -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    lr = 0.01
    n_trajectories = 300
    percet_optimal = 0.9

    # Create environment
    env = rd.Environment("CartPole-v1", batch_size=8, n_trajectories=n_trajectories, perct_optimal=percet_optimal)

    hidden_size = 15

    network = rd.LstmRudder(n_states=env.n_states, n_actions=env.n_actions,
                            hidden_size=hidden_size, n_lstm_layers=1, device=device).to(device)
    # Save LSTM
    network.save_model(env.gym, f'{hidden_size}_{lr}_{n_trajectories}_{percet_optimal}')

    # Load LSTM
    network.load_model(env.gym, f'{hidden_size}_{lr}_{n_trajectories}_{percet_optimal}')

    # Create Network
    network = rd.LstmCellRudder(n_states=env.n_states, n_actions=env.n_actions,
                                hidden_size=hidden_size, device=device, init_weights=True).to(device)

    # Save LSTM
    network.save_model(env.gym, f'{hidden_size}_{lr}_{n_trajectories}_{percet_optimal}')

    # Load LSTM
    network.load_lstm_model(env.gym, f'{hidden_size}_{lr}_{n_trajectories}_{percet_optimal}')



if __name__ == '__main__':

    user_spec = RudderExperimentSpec(
        n_epoches=None,
        env_batch_size=None,
        loader_batch_size=None,
        show_plot=True,
        )

    test_spec = dataclasses.replace(user_spec,
                                    n_epoches=None,
                                    show_plot=False,
                                    )

    theSpec, _ = check_testspec_flag_and_setup_spec(user_spec, test_spec)
    main(theSpec)
