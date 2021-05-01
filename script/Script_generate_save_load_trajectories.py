import dataclasses

import gym
from codebase import ppo
from codebase import rudder

from experiment_runner.test_related_utils import check_testspec_flag_and_setup_spec
from experiment_runner.experiment_spec import GenerateSaveLoadTrjExperimentSpec


def main(spec: GenerateSaveLoadTrjExperimentSpec) -> None:

    environment = gym.make(spec.env_name)

    state_size = environment.observation_space.shape[0]
    action_size = environment.action_space.n

    # Initialize agent network
    agent = ppo.NnActorCritic(state_size, action_size,
                              n_hidden_layers=spec.n_hidden_layers,
                              hidden_dim=spec.hidden_dim,
                              lr=spec.optimizer_lr,
                              weight_decay=spec.optimizer_weight_decay,
                              device='cpu', )

    # Generate and save trajectories in experiment
    rudder.generate_trajectories(environment, spec.n_trajectory_per_policy, agent)

    data = rudder.load_trajectories(environment,
                                    n_trajectories=spec.env_n_trajectories,
                                    perct_optimal=spec.env_perct_optimal)
    print('keys of data :', data.keys())
    print(data['reward'].shape)
    #print(data['delayed_reward'][-1])

    return None


if __name__ == '__main__':

    user_spec = GenerateSaveLoadTrjExperimentSpec(
        # Environment : CartPole-v1, LunarLander-v2
        #               "MountainCar-v0" # <-- (Priority) todo:fixme!! (ref task T13PRO-140)
        env_name='CartPole-v1',
        steps_by_epoch=1500,
        n_epoches=50,
        hidden_dim=18,
        n_hidden_layers=1,
        optimizer_weight_decay=0.0,
        optimizer_lr=1e-3,
        show_plot=True,
        n_trajectory_per_policy=20,
        env_n_trajectories=5,
        env_perct_optimal=0.5,
        )

    test_spec = dataclasses.replace(user_spec,
                                    steps_by_epoch=10,
                                    n_epoches=2,
                                    n_trajectory_per_policy=2,
                                    show_plot=False,
                                    )

    theSpec, _ = check_testspec_flag_and_setup_spec(user_spec, test_spec)
    main(theSpec)
