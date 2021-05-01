# coding=utf-8
import random

from experiment_runner.parameter_search_map import RudderLstmParameterSearchMap

colab_specs = RudderLstmParameterSearchMap(
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
