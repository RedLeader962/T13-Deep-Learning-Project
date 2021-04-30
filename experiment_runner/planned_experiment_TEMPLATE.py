# coding=utf-8
from typing import Any

from experiment_runner.experiment_runner_utils import execute_experiment_plan
from experiment_runner.experiment_spec import RudderLstmExperimentSpec
from script.Script_run_LSTM import main as script_rudder_lstm_main

spec_1 = RudderLstmExperimentSpec(
    name="Hypotheses #1",
    env_name="CartPole-v1",
    env_batch_size=8,
    model_hidden_size=15,
    env_n_trajectories=10,
    env_perct_optimal=0.5,
    env_rew_factor=0.1,
    n_epoches=2,
    optimizer_weight_decay=1e-2,
    optimizer_lr=1e-3,
    show_plot=False,
    seed=42,
    )

spec_2 = RudderLstmExperimentSpec(
    name="Hypotheses #2",
    env_name="CartPole-v1",
    env_batch_size=8,
    model_hidden_size=8,
    env_n_trajectories=7,
    env_perct_optimal=0.5,
    n_epoches=2,
    show_plot=False,
    seed=42,
    )

spec_3 = RudderLstmExperimentSpec(
    name="Hypotheses #3",
    env_name="CartPole-v1",
    env_batch_size=8,
    model_hidden_size=10,
    env_n_trajectories=7,
    env_perct_optimal=0.5,
    n_epoches=2,
    optimizer_weight_decay=1e-6,
    show_plot=False,
    seed=42,
    )


def main() -> Any:
    specs_dict_w_resuts = execute_experiment_plan(exp_specs=[spec_1,
                                                            spec_2,
                                                            spec_3, ],
                                                 script_fct=script_rudder_lstm_main,
                                                 )

    exp_configuration_id_no2 = specs_dict_w_resuts['2']

    exp_configuration_id_no2.comment = "You can add comment to the experiment run"

    return specs_dict_w_resuts


if __name__ == '__main__':
    main()
