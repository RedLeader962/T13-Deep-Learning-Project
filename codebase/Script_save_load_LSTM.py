import dataclasses
from dataclasses import dataclass

import torch
import gym

from codebase import rudder as rd
from general_utils import ExperimentSpec, check_testspec_flag_and_setup_spec

@dataclass(frozen=True)
class RudderExperimentSpec(ExperimentSpec):
    n_epoches: int

def main(spec: RudderExperimentSpec) -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Environment : CartPole-v1, MountainCar-v0, LunarLander-v2
    env = gym.make("CartPole-v1")

    network = rd.LstmRudder(n_positions=2, n_actions=2,
                            hidden_size=2, n_lstm_layers=1, device=device).to(device)

    # Save LSTM
    rd.save_lstm_or_lstmcell_in_env(env, network, lstmcell=False)

    # Load LSTM
    rd.load_lstm_or_lstmcell_from_env(env, lstmcell=None)


if __name__ == '__main__':

    user_spec = RudderExperimentSpec(
        n_epoches=5,
        show_plot=True,
        )

    test_spec = dataclasses.replace(user_spec, show_plot=False, n_epoches=2)

    theSpec, _ = check_testspec_flag_and_setup_spec(user_spec, test_spec)
    main(theSpec)
