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
    rudder_hidden_size=lambda: random.choice([10, 15, 20, 25, 30, 40, 60]),
    hidden_dim=18,
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

T1Bravo_CP_colab_ppo_rudder_top_to_bottom_specs = PpoRudderParameterSearchMap(
    env_name="CartPole-v1",  # MountainCar-v0, LunarLander-v2
    env_batch_size=lambda: random.choice([4, 8]),
    rudder_hidden_size=lambda: random.choice([25, 30, 35, 40]),
    hidden_dim=18,
    env_n_trajectories=2800,
    env_perct_optimal=lambda: random.choice([0.9, 0.7, 0.6, 0.5, 0.4, 0.3, 0.1]),
    n_epoches=500,
    # n_epoches=3,            # <-- temp test dev
    optimizer_weight_decay=lambda: random.choice([0.01, 0.001]),
    optimizer_lr=0.001,
    show_plot=True,
    print_to_consol=False,
    seed=42,
    steps_by_epoch=1000,
    n_hidden_layers=1,
    n_trajectory_per_policy=1,
    reward_delayed=True,
    rew_factor=1.0,
    spec_name='Configuration excel 1.c - revision F-A',
    batch_tag='T1CP',
    )

T1Bravo_MC_colab_ppo_rudder_top_to_bottom_specs = PpoRudderParameterSearchMap(
    env_name="MountainCar-v0",  # MountainCar-v0, LunarLander-v2
    env_batch_size=lambda: random.choice([4, 8]),
    rudder_hidden_size=lambda: random.choice([20, 25, 30, 35, 40]),
    hidden_dim=18,
    env_n_trajectories=4000,
    env_perct_optimal=lambda: random.choice([0.9, 0.7, 0.6, 0.5, 0.4, 0.3, 0.1]),
    n_epoches=450,
    # n_epoches=3,            # <-- temp test dev
    optimizer_weight_decay=lambda: random.choice([0.01, 0.001]),
    optimizer_lr=0.001,
    show_plot=True,
    print_to_consol=False,
    seed=42,
    steps_by_epoch=600,
    n_hidden_layers=1,
    n_trajectory_per_policy=1,
    reward_delayed=True,
    rew_factor=1.0,
    spec_name='Configuration excel 1.c - revision F-A',
    batch_tag='T1MC',
    )
