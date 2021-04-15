# coding=utf-8
import dataclasses
from dataclasses import dataclass

import gym
import matplotlib.pyplot as plt
import ppo

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

    # Environment : CartPole-v1, MountainCar-v0, LunarLander-v2
    environment = gym.make("CartPole-v1")

    steps_by_epoch = spec.steps_by_epoch
    n_epoches = spec.n_epoches
    hidden_dim = spec.hidden_dim
    n_hidden_layers = spec.n_hidden_layers

    agent, info_logger = ppo.run_ppo(environment,
                                     steps_by_epoch=steps_by_epoch,
                                     n_epoches=n_epoches,
                                     n_hidden_layers=n_hidden_layers,
                                     hidden_dim=hidden_dim,
                                     lr=0.01,
                                     save_gap=1,
                                     device=device)

    dir_name = environment.unwrapped.spec.id
    dim_NN = environment.observation_space.shape[0], hidden_dim, environment.action_space.n

    epochs_data = info_logger.load_data(dir_name, dim_NN)

    if spec.show_plot:
        plt.title(f"PPO - Number of epoches : {n_epoches} and steps by epoch : {steps_by_epoch}")
        plt.plot(epochs_data['E_average_return'], label='E_average_return')
        plt.legend()
        plt.xlabel("Epoches")
        plt.show()

    ppo.generate_trajectories(environment, spec.n_trajectory_per_policy, agent, optimal_policy=True)

    return None


if __name__ == '__main__':

    user_spec = PpoExperimentSpec(
        steps_by_epoch=1000,
        n_epoches=2,
        hidden_dim=6,
        n_hidden_layers=2,
        device="cpu",
        show_plot=True,
        n_trajectory_per_policy=6)

    test_spec = dataclasses.replace(user_spec, steps_by_epoch=500, show_plot=False, n_trajectory_per_policy=2)

    theSpec, _ = check_testspec_flag_and_setup_spec(user_spec, test_spec)
    main(theSpec)
