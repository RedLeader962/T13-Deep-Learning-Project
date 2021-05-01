# coding=utf-8
import random
from typing import Any

from experiment_runner.experiment_runner_utils import execute_parameter_search
from experiment_runner.parameter_search_map import RudderLstmParameterSearchMap
from script.Script_run_LSTM import main as script_rudder_lstm_main

parameter_search_spec_example = RudderLstmParameterSearchMap(
    env_name="CartPole-v1",
    spec_name="An example of spec configuration space",
    env_batch_size=8,
    model_hidden_size=lambda: random.choice([10, 15, 20, 25, 30, 40, 60]),
    env_n_trajectories=lambda: random.choice([3000, 4000, 5000, ]),
    env_perct_optimal=lambda: random.choice([0.3, 0.4, 0.5, 0.5, 0.5, 0.6, 0.7, ]),
    rew_factor=1.0,
    n_epoches=lambda: random.choice([20, 40, 60, 80, 100, 140, 200, 300, 400, 600]),
    optimizer_weight_decay=1e-5,
    optimizer_lr=lambda: random.choice([1e-1, 1e-2, 1e-3]),
    show_plot=True,
    print_to_consol=False,
    seed=42,
    )


def main() -> Any:
    specs_dict_w_resuts = execute_parameter_search(exp_spec=parameter_search_spec_example,
                                                   script_fct=script_rudder_lstm_main,
                                                   exp_size=10,
                                                   start_count_at=1,
                                                   )

    exp_configuration_id_no3 = specs_dict_w_resuts['3']

    exp_configuration_id_no3.comment = "You can add comment to the experiment run"

    return specs_dict_w_resuts


if __name__ == '__main__':
    main()
