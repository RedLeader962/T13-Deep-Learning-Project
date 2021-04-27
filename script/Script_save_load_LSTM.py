import dataclasses

import torch
import gym

from codebase import rudder as rd
from script.general_utils import check_testspec_flag_and_setup_spec
from script.experiment_spec import RudderExperimentSpec


def main(spec: RudderExperimentSpec) -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Environment : CartPole-v1, MountainCar-v0, LunarLander-v2
    env = gym.make("CartPole-v1")

    network = rd.LstmRudder(n_positions=2, n_actions=2,
                            hidden_size=2, n_lstm_layers=1, device=device).to(device)

    # Save LSTM
    rd.save_data_or_network(env, network.state_dict(), 'lstm')

    # Load LSTM
    rd.load_network(env, network, 'lstm')


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
