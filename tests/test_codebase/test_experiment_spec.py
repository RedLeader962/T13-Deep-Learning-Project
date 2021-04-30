# coding=utf-8
import random

import pytest

pytestmark = pytest.mark.automated_test


def test_ExperimentSpec_comment_PASS():
    from script.experiment_spec import ExperimentSpec

    test_spec = ExperimentSpec(
        show_plot=False,
        )

    test_spec.name = "The experiement"
    test_spec.comment = "That was a smooth test run!"

    print(f"\n{test_spec}\n")


def test_RudderLstmExperimentSpec_PASS():
    from script.experiment_spec import RudderLstmExperimentSpec

    test_spec = RudderLstmExperimentSpec(
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
    from script.experiment_spec import RudderLstmParameterSearchMap

    test_spec = RudderLstmParameterSearchMap(
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
