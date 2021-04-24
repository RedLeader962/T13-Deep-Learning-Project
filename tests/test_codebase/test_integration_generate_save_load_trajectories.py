# coding=utf-8

import pytest
import os

def test_rudder_generate_trajectories_PASS():
    import gym
    import ppo
    import rudder
    from Script_generate_save_load_trajectories import PpoExperimentSpec
    from Script_generate_save_load_trajectories import main as rudder_main

    print(f"\n››› CWD: {os.getcwd()}")

    test_spec = PpoExperimentSpec(
        steps_by_epoch=100,
        n_epoches=5,
        hidden_dim=15,
        n_hidden_layers=1,
        device="cpu",
        show_plot=False,
        n_trajectory_per_policy=1)

    rudder_main(test_spec)

    # assert data['E_average_return'][-1] == 500.0
