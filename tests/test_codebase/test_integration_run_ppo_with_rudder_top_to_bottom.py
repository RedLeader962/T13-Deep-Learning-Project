# coding=utf-8
import pytest

from experiment_runner.constant import TEST_EXPERIMENT_RUN_DIR
from experiment_runner.test_related_utils import show_plot_unless_CI_server_runned


def test_Script_run_ppo_with_rudder_top_to_bottom_main_PASS():
    from experiment_runner.experiment_spec import PpoRudderExperimentSpec
    from script.Script_run_ppo_with_rudder_top_to_bottom import main as ppo_with_rudder_top_to_bottom_main

    test_spec = PpoRudderExperimentSpec(
        env_name='CartPole-v1',  # Environment : CartPole-v1, MountainCar-v0, LunarLander-v2
        n_epoches=2,
        steps_by_epoch=500,
        hidden_dim=6,
        n_hidden_layers=2,
        n_trajectory_per_policy=1,
        reward_delayed=True,
        rew_factor=1.0,
        optimizer_weight_decay=0.0,
        optimizer_lr=1e-3,
        env_batch_size=8,
        env_n_trajectories=3200,
        env_perct_optimal=0.5,
        seed=42,
        show_plot=show_plot_unless_CI_server_runned(False),
        print_to_consol=True,
        root_experiment_dir=TEST_EXPERIMENT_RUN_DIR,
        experiment_tag='Test Run',
        )

    ppo_with_rudder_top_to_bottom_main(test_spec)
