import dataclasses

import torch
import numpy as np

from codebase import rudder as rd
from script.general_utils import check_testspec_flag_and_setup_spec
from script.experiment_spec import RudderExperimentSpec


def main(spec: RudderExperimentSpec) -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    rnd_gen = np.random.RandomState(seed=123)

    # Environment : CartPole-v1, MountainCar-v0, LunarLander-v2
    env = rd.Environment("CartPole-v1", batch_size=1000, max_timestep=50, n_positions=13, rnd_gen=rnd_gen)

    hidden_size = 15

    network = rd.LstmRudder(n_positions=2, n_actions=2,
                            hidden_size=hidden_size, n_lstm_layers=1, device=device).to(device)
    # Save LSTM
    network.save_model(env.gym)

    # Load LSTM
    network.load_model(env.gym)

    # Create Network
    network = rd.LstmCellRudder(n_positions=2, n_actions=2,
                                hidden_size=hidden_size, n_lstm_layers=1, device=device, init_weights=True).to(device)

    # Save LSTM
    network.save_model(env.gym)

    # Load LSTM
    network.load_model(env.gym)



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
