# coding=utf-8

import pytest
import os

from script.general_utils import show_plot_while_not_on_CI_server


def test_rudder_generate_trajectories_PASS():
    from script.Script_generate_save_load_trajectories import PpoExperimentSpec
    from script.Script_generate_save_load_trajectories import main as rudder_main

    print(f"\n››› CWD: {os.getcwd()}")

    test_spec = PpoExperimentSpec(
        steps_by_epoch=10,
        n_epoches=2,
        hidden_dim=16,  # ‹‹‹ (CRITICAL) todo:fixme!! (ref task T13PRO-121 )
        n_hidden_layers=1,
        device="cpu",
        n_trajectory_per_policy=1,
        show_plot=show_plot_while_not_on_CI_server(False),
        )

    rudder_main(test_spec)

