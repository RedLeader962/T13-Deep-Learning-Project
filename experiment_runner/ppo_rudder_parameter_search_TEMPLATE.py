# coding=utf-8
import random
from typing import Any

from experiment_runner.experiment_runner_utils import execute_parameter_search
from experiment_runner.parameter_search_map import PpoRudderParameterSearchMap
from script.Script_run_ppo_with_rudder import main as script_run_ppo_with_rudder_main

parameter_search_spec_example = PpoRudderParameterSearchMap(
    env_name='CartPole-v1',  # Environment : CartPole-v1, MountainCar-v0, LunarLander-v2
    spec_name="An example of spec configuration space",
    env_batch_size=8,
    hidden_dim=lambda: random.choice([10, 15, 20, 25, 30, 40, 60]),
    env_n_trajectories=lambda: random.choice([3000, 4000, 5000, ]),
    env_perct_optimal=lambda: random.choice([0.3, 0.4, 0.5, 0.5, 0.5, 0.6, 0.7, ]),
    rew_factor=1.0,
    n_epoches=lambda: random.choice([20, 40, 60, 80, 100, 140, 200, 300, 400, 600]),
    optimizer_weight_decay=1e-9,
    optimizer_lr=lambda: random.choice([1e-1, 1e-2, 1e-3]),
    steps_by_epoch=1000,
    n_hidden_layers=1,
    n_trajectory_per_policy=1,
    reward_delayed=True,
    rudder_hidden_size=35,
    show_plot=True,
    print_to_consol=False,
    seed=42,
    )


def main() -> Any:
    specs_dict_w_resuts = execute_parameter_search(exp_spec=parameter_search_spec_example,
                                                   script_fct=script_run_ppo_with_rudder_main,
                                                   exp_size=10,
                                                   start_count_at=1,
                                                   )

    exp_configuration_id_no3 = specs_dict_w_resuts['3']

    exp_configuration_id_no3.comment = "You can add comment to the experiment run"

    return specs_dict_w_resuts


if __name__ == '__main__':
    main()
