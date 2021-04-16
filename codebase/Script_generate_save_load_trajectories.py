
import dataclasses
from dataclasses import dataclass

import gym
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

    # Environment : CartPole-v1, MountainCar-v0, LunarLander-v2
    environment = gym.make("CartPole-v1")

    hidden_dim = spec.hidden_dim
    n_hidden_layers = spec.n_hidden_layers

    state_size = environment.observation_space.shape[0]
    action_size = environment.action_space.n

    # Initialize agent network
    agent = ppo.NnActorCritic(state_size,
                          action_size,
                          n_hidden_layers=n_hidden_layers,
                          hidden_dim=hidden_dim)

    # Generate and save trajectories in experiment
    ppo.generate_trajectories(environment, spec.n_trajectory_per_policy, agent)

    data = ppo.load_trajectories(environment, n_trajectories=40, perct_optimal=0.5)
    print('keys of data :', data.keys())

    return None

if __name__ == '__main__':

    user_spec = PpoExperimentSpec(
        steps_by_epoch=1000,
        n_epoches=40,
        hidden_dim=5,
        n_hidden_layers=2,
        device="cpu",
        show_plot=True,
        n_trajectory_per_policy=1)

    test_spec = dataclasses.replace(user_spec, show_plot=False, n_trajectory_per_policy=2)

    theSpec, _ = check_testspec_flag_and_setup_spec(user_spec, test_spec)
    main(theSpec)
