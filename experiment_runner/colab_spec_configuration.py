# coding=utf-8
import random

from experiment_runner.parameter_search_map import PpoRudderParameterSearchMap, RudderLstmParameterSearchMap

colab_rudder_lstm_specs = RudderLstmParameterSearchMap(
    env_name="CartPole-v1",
    env_batch_size=lambda: random.choice([4, 6, 8, 10]),
    model_hidden_size=lambda: random.choice([10, 15, 20, 25, 30, 40, 60]),
    env_n_trajectories=lambda: random.choice([3000, 4000, 5000, ]),
    env_perct_optimal=lambda: random.choice([0.3, 0.4, 0.5, 0.5, 0.5, 0.6, 0.7, ]),
    n_epoches=lambda: random.choice([20, 40, 60, 80, 100, 140, 200, 300, 400, 600]),
    optimizer_weight_decay=lambda: random.choice([1e-1, 1e-2, 1e-3]),
    optimizer_lr=lambda: random.choice([1e-1, 1e-2, 1e-3]),
    show_plot=True,
    print_to_consol=False,
    seed=42,
    )

colab_ppo_rudder_top_to_bottom_specs = PpoRudderParameterSearchMap(
    env_name="CartPole-v1",  # MountainCar-v0, LunarLander-v2
    env_batch_size=lambda: random.choice([4, 6, 8, 10]),
    hidden_dim=lambda: random.choice([10, 15, 20, 25, 30, 40, 60]),
    env_n_trajectories=lambda: random.choice([3000, 4000, 5000, ]),
    env_perct_optimal=lambda: random.choice([0.3, 0.4, 0.5, 0.5, 0.5, 0.6, 0.7, ]),
    n_epoches=lambda: random.choice([20, 40, 60, 80, 100, 140, 200, 300, 400, 600]),
    optimizer_weight_decay=lambda: random.choice([1e-1, 1e-2, 1e-3]),
    optimizer_lr=lambda: random.choice([1e-1, 1e-2, 1e-3]),
    show_plot=True,
    print_to_consol=False,
    seed=42,
    steps_by_epoch=1000,
    n_hidden_layers=1,
    n_trajectory_per_policy=1,
    reward_delayed=True,
    rew_factor=1.0,
    # spec_name=,
    # batch_tag=,
    )

T1CP_colab_ppo_rudder_top_to_bottom_specs = PpoRudderParameterSearchMap(
    env_name="CartPole-v1",  # MountainCar-v0, LunarLander-v2
    env_batch_size=lambda: random.choice([4, 8, 16]),
    hidden_dim=lambda: random.choice([30, 35, 40, 60]),
    env_n_trajectories=lambda: random.choice([3000, 4000, 5000, ]),
    env_perct_optimal=lambda: random.choice([0.7, 0.6, 0.5, 0.4, 0.3, ]),
    n_epoches=500,
    optimizer_weight_decay=lambda: random.choice([0.01, 0.0001, 0.000001]),
    optimizer_lr=0.001,
    show_plot=True,
    print_to_consol=False,
    seed=42,
    steps_by_epoch=1000,
    n_hidden_layers=1,
    n_trajectory_per_policy=1,
    reward_delayed=True,
    rew_factor=1.0,
    spec_name='Configuration excel 1',
    batch_tag='T1CP',
    )

T1MC_colab_ppo_rudder_top_to_bottom_specs = PpoRudderParameterSearchMap(
    env_name="MountainCar-v0",  # MountainCar-v0, LunarLander-v2
    env_batch_size=lambda: random.choice([4, 8, 16]),
    hidden_dim=lambda: random.choice([20, 25, 30, 60]),
    env_n_trajectories=lambda: random.choice([3000, 4000, 5000, ]),
    env_perct_optimal=lambda: random.choice([0.7, 0.6, 0.5, 0.4, 0.3, ]),
    n_epoches=400,
    optimizer_weight_decay=lambda: random.choice([0.01, 0.0001, 0.000001]),
    optimizer_lr=0.001,
    show_plot=True,
    print_to_consol=False,
    seed=42,
    steps_by_epoch=1000,
    n_hidden_layers=1,
    n_trajectory_per_policy=1,
    reward_delayed=True,
    rew_factor=1.0,
    spec_name='Configuration excel 1',
    batch_tag='T1MC',
    )
