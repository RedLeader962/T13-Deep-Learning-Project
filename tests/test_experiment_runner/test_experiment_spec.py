# coding=utf-8
import random

import pytest

pytestmark = pytest.mark.automated_test


def test_ExperimentSpec_comment_PASS():
    from experiment_runner.experiment_spec import ExperimentSpec

    test_spec = ExperimentSpec(
        show_plot=False,
        )

    test_spec.spec_name = "The experiement"
    test_spec.comment = "That was a smooth test run!"

    print(f"\n{test_spec}\n")


def test_RudderLstmExperimentSpec_PASS():
    from experiment_runner.experiment_spec import RudderLstmExperimentSpec

    test_spec = RudderLstmExperimentSpec(
        env_name="CartPole-v1",
        env_batch_size=8,
        model_hidden_size=15,
        env_n_trajectories=10,
        env_perct_optimal=0.5,
        n_epoches=20,
        optimizer_weight_decay=1e-2,
        optimizer_lr=1e-3,
        show_plot=False,
        seed=None,
        # seed=42,
        )

    print(f"\n{test_spec}\n")


def test_RudderLstmParameterSearchMap_PASS():
    from experiment_runner.parameter_search_map import RudderLstmParameterSearchMap

    test_spec = RudderLstmParameterSearchMap(
        env_name="CartPole-v1",
        env_batch_size=8,
        model_hidden_size=lambda: random.choice([1, 3, 10]),
        env_n_trajectories=10,
        env_perct_optimal=0.5,
        n_epoches=2,
        optimizer_weight_decay=1e-2,
        optimizer_lr=1e-3,
        show_plot=False,
        seed=None,
        # seed=42,
        )

    assert test_spec.model_hidden_size in [1, 3, 10]

    print(f"\n{test_spec}\n")


def test_PpoRudderExperimentSpec_PASS():
    from experiment_runner.experiment_spec import PpoRudderExperimentSpec

    test_spec = PpoRudderExperimentSpec(
        env_name='CartPole-v1',  # Environment : CartPole-v1, MountainCar-v0, LunarLander-v2
        n_epoches=75,
        steps_by_epoch=1000,
        hidden_dim=18,
        n_hidden_layers=1,
        n_trajectory_per_policy=1,
        reward_delayed=True,
        rew_factor=1.0,
        optimizer_weight_decay=0.0,
        optimizer_lr=1e-3,
        rudder_hidden_size=35,
        env_batch_size=8,
        env_n_trajectories=3200,
        env_perct_optimal=0.7,
        seed=42,
        show_plot=True,
        print_to_consol=True,
        )

    print(f"\n{test_spec}\n")


def test_PpoRudderParameterSearchMap_PASS():
    from experiment_runner.parameter_search_map import PpoRudderParameterSearchMap

    test_spec = PpoRudderParameterSearchMap(
        env_name='CartPole-v1',  # Environment : CartPole-v1, MountainCar-v0, LunarLander-v2
        n_epoches=75,
        steps_by_epoch=1000,
        hidden_dim=lambda: random.choice([1, 3, 10]),
        n_hidden_layers=1,
        n_trajectory_per_policy=1,
        reward_delayed=True,
        rew_factor=1.0,
        optimizer_weight_decay=0.0,
        optimizer_lr=1e-3,
        rudder_hidden_size=lambda: random.choice([1, 3, 10]),
        env_batch_size=8,
        env_n_trajectories=3200,
        env_perct_optimal=0.7,
        seed=42,
        show_plot=True,
        print_to_consol=True,
        )

    assert test_spec.hidden_dim in [1, 3, 10]
    assert test_spec.rudder_hidden_size in [1, 3, 10]

    print(f"\n{test_spec}\n")
