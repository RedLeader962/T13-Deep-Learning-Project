# coding=utf-8
import dataclasses
import statistics
from copy import copy
import random
from typing import Dict

import pytest

from experiment_runner.constant import TEST_EXPERIMENT_RUN_DIR
from experiment_runner.experiment_spec import ExperimentSpec

pytestmark = pytest.mark.automated_test


def test_execute_experiment_plan_PASS():
    from experiment_runner.experiment_runner_utils import execute_experiment_plan
    from experiment_runner.experiment_spec import RudderLstmExperimentSpec
    from script.Script_run_LSTM import main as lstm_main

    test_spec = RudderLstmExperimentSpec(
        env_name="CartPole-v1",
        env_batch_size=8,
        model_hidden_size=15,
        env_n_trajectories=10,
        env_perct_optimal=0.5,
        n_epoches=2,
        optimizer_weight_decay=1e-2,
        optimizer_lr=1e-3,
        show_plot=False,
        seed=42,
        root_experiment_dir=TEST_EXPERIMENT_RUN_DIR,
        batch_tag='Test Plan Batch',
        spec_name='UONE'
        )

    test_spec.spec_name = "test spec 1"
    test_spec2 = dataclasses.replace(test_spec, spec_name='DEUX')
    test_spec2.spec_name = "test spec 2"

    specs: Dict[str, ExperimentSpec] = execute_experiment_plan(exp_specs=[test_spec, test_spec2], script_fct=lstm_main)

    assert len(specs) == 2
    assert specs['1'].spec_idx == 1
    assert specs['2'].spec_idx == 2


def test_execute_parameter_search_pre_condition_PASS():
    from experiment_runner.parameter_search_map import RudderLstmParameterSearchMap
    from experiment_runner.experiment_runner_utils import execute_parameter_search
    from script.Script_run_LSTM import main as lstm_main

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
        seed=42,
        root_experiment_dir=TEST_EXPERIMENT_RUN_DIR,
        batch_tag='Test Param Seach Batch'
        )

    with pytest.raises(AssertionError):
        execute_parameter_search(exp_spec=test_spec,
                                 script_fct=lstm_main(test_spec),
                                 exp_size=3,
                                 start_count_at=1,
                                 consol_print=False,
                                 )


def test_execute_parameter_search_on_rudder_PASS():
    from experiment_runner.parameter_search_map import RudderLstmParameterSearchMap
    from experiment_runner.experiment_runner_utils import execute_parameter_search
    from script.Script_run_LSTM import main as lstm_main

    test_spec = RudderLstmParameterSearchMap(
        env_name="CartPole-v1",
        env_batch_size=4,
        model_hidden_size=lambda: random.choice([11, 22]),
        env_n_trajectories=2,
        env_perct_optimal=0.5,
        n_epoches=2,
        optimizer_weight_decay=lambda: random.choice([1e-1, 1e-3]),
        optimizer_lr=1e-3,
        show_plot=False,
        seed=42,
        root_experiment_dir=TEST_EXPERIMENT_RUN_DIR,
        batch_tag='Test Param Seach Batch'
        )

    results = execute_parameter_search(exp_spec=test_spec, script_fct=lstm_main, exp_size=30, consol_print=False)

    values_hsize = []
    values_wd = []
    for _, each_spec in results.items():
        values_hsize.append(each_spec.model_hidden_size)
        values_wd.append(each_spec.optimizer_weight_decay)

    statistics_mean = statistics.mean(values_hsize)
    assert statistics_mean not in [11, 22]

    print(f"\n\n››› model_hidden_size list: {values_hsize}"
          f"\n››› statistics_mean: {statistics_mean}"
          f"\n\n››› optimizer_weight_decay list: {values_wd}\n\n")


def test_execute_parameter_search_on_ppoRudder_ppo_top_to_bottom_PASS():
    from experiment_runner.parameter_search_map import PpoRudderParameterSearchMap
    from experiment_runner.experiment_runner_utils import execute_parameter_search
    from script.Script_run_ppo_with_rudder_top_to_bottom import main as ppo_with_rudder_top_to_bottom_main

    test_spec = PpoRudderParameterSearchMap(
        env_name="CartPole-v1",
        env_batch_size=2,
        rudder_hidden_size=lambda: random.choice([11, 22]),
        hidden_dim=18,
        env_n_trajectories=2,
        env_perct_optimal=0.5,
        n_epoches=2,
        optimizer_weight_decay=lambda: random.choice([1e-1, 1e-3]),
        optimizer_lr=1e-3,
        seed=42,
        steps_by_epoch=500,
        n_hidden_layers=2,
        n_trajectory_per_policy=1,
        reward_delayed=True,
        rew_factor=1.0,
        print_to_consol=False,
        show_plot=False,
        root_experiment_dir=TEST_EXPERIMENT_RUN_DIR,
        batch_tag='Test Param Seach Batch ETE'
        )

    results = execute_parameter_search(exp_spec=test_spec, script_fct=ppo_with_rudder_top_to_bottom_main, exp_size=10,
                                       consol_print=False)

    values_hsize = []
    values_wd = []
    for _, each_spec in results.items():
        values_hsize.append(each_spec.rudder_hidden_size)
        values_wd.append(each_spec.optimizer_weight_decay)

    statistics_mean = statistics.mean(values_hsize)
    assert statistics_mean not in [11, 22]

    print(f"\n\n››› model_hidden_size list: {values_hsize}"
          f"\n››› statistics_mean: {statistics_mean}"
          f"\n\n››› optimizer_weight_decay list: {values_wd}\n\n")
