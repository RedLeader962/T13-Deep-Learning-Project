import dataclasses

import torch

from codebase import rudder as rd
from experiment_runner.test_related_utils import check_testspec_flag_and_setup_spec
from experiment_runner.experiment_spec import RudderExperimentSpec


def main(spec: RudderExperimentSpec) -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Create environment
    env = rd.Environment("CartPole-v1", batch_size=8, n_trajectories=500, perct_optimal=0.7)

    hidden_size = 15

    network = rd.LstmRudder(n_states=env.n_states, n_actions=env.n_actions,
                            hidden_size=hidden_size, n_lstm_layers=1, device=device).to(device)
    # Save LSTM
    network.save_model(env.gym)

    # Load LSTM
    network.load_model(env.gym)

    # Create Network
    network = rd.LstmCellRudder(n_states=env.n_states, n_actions=env.n_actions,
                                hidden_size=hidden_size, device=device, init_weights=True).to(device)

    # Save LSTM
    network.save_model(env.gym)

    # Load LSTM
    network.load_lstm_model(env.gym)



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
