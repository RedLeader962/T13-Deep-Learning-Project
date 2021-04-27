# coding=utf-8
import dataclasses
from dataclasses import dataclass

import ppo
import rudder as rd
import numpy as np


from general_utils import check_testspec_flag_and_setup_spec, ExperimentSpec


@dataclass(frozen=True)
class PpoExperimentSpec(ExperimentSpec):
    steps_by_epoch: int
    n_epoches: int
    hidden_dim: int
    n_hidden_layers: int
    device: str
    n_trajectory_per_policy: int


def main(spec: PpoExperimentSpec) -> None:
    # keep gpu ! Quicker for PPO !
    device = spec.device

    # Prepare some random generators for later
    rnd_gen = np.random.RandomState(seed=123)

    # Create environment
    n_positions = 13
    env = rd.Environment("CartPole-v1", batch_size=1000, max_timestep=50, n_positions=13, rnd_gen=rnd_gen)

    steps_by_epoch = spec.steps_by_epoch
    n_epoches = spec.n_epoches
    hidden_dim = spec.hidden_dim
    n_hidden_layers = spec.n_hidden_layers


    # Initialize LSTMCell network
    n_lstm_layers = 1
    hidden_size = 15
    lstmcell_rudder = None #rd.LstmCellRudder(n_positions=2, n_actions=2,
                      #          hidden_size=hidden_size, n_lstm_layers=n_lstm_layers, device=device, init_weights=True).to(device)

    # Initialize agent
    agent, info_logger = ppo.run_ppo(env.gym,
                                     lstmcell_rudder=lstmcell_rudder,
                                     steps_by_epoch=steps_by_epoch,
                                     n_epoches=n_epoches,
                                     n_hidden_layers=n_hidden_layers,
                                     hidden_dim=hidden_dim,
                                     lr=0.001,
                                     save_gap=1,
                                     device=device)


    return None


if __name__ == '__main__':

    user_spec = PpoExperimentSpec(
        steps_by_epoch=1000,
        n_epoches=400,
        hidden_dim=18,
        n_hidden_layers=1,
        device="cpu",
        show_plot=True,
        n_trajectory_per_policy=1)

    test_spec = dataclasses.replace(user_spec, show_plot=False, n_trajectory_per_policy=2)

    theSpec, _ = check_testspec_flag_and_setup_spec(user_spec, test_spec)
    main(theSpec)
