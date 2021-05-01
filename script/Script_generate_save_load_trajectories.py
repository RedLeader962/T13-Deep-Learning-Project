import dataclasses

import gym
from codebase import ppo
from codebase import rudder

from script.general_utils import check_testspec_flag_and_setup_spec
from script.experiment_spec import PpoExperimentSpec


def main(spec: PpoExperimentSpec) -> None:
    # Environment : CartPole-v1, MountainCar-v0, LunarLander-v2
    # environment = gym.make("MountainCar-v0") # <-- (Priority) todo:fixme!! (ref task T13PRO-140)
    environment = gym.make("CartPole-v1")

    hidden_dim = spec.hidden_dim
    n_hidden_layers = spec.n_hidden_layers

    state_size = environment.observation_space.shape[0]
    action_size = environment.action_space.n

    # Initialize agent network
    agent = ppo.NnActorCritic(state_size,
                              action_size,
                              n_hidden_layers=n_hidden_layers,
                              hidden_dim=hidden_dim,
                              device='cpu')

    # Generate and save trajectories in experiment
    rudder.generate_trajectories(environment, spec.n_trajectory_per_policy, agent)

    data = rudder.load_trajectories(environment, n_trajectories=5, perct_optimal=0.5)
    print('keys of data :', data.keys())
    print(data['reward'].shape)
    #print(data['delayed_reward'][-1])

    return None


if __name__ == '__main__':

    user_spec = PpoExperimentSpec(
        steps_by_epoch=1500,
        n_epoches=50,
        hidden_dim=18,
        n_hidden_layers=1,
        show_plot=True,
        n_trajectory_per_policy=20)

    test_spec = dataclasses.replace(user_spec,
                                    steps_by_epoch=10,
                                    n_epoches=2,
                                    n_trajectory_per_policy=2,
                                    show_plot=False,
                                    )

    theSpec, _ = check_testspec_flag_and_setup_spec(user_spec, test_spec)
    main(theSpec)
