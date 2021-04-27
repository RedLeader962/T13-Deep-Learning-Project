# coding=utf-8

import pytest

from script.general_utils import show_plot_unless_CI_server_runned


def test_rudder_generate_load_trajectories_PASS():
    from script.experiment_spec import PpoExperimentSpec
    from script.Script_generate_save_load_trajectories import main as rudder_main

    test_spec = PpoExperimentSpec(
        steps_by_epoch=10,
        n_epoches=2,
        hidden_dim=16,  # ‹‹‹ (CRITICAL) todo:fixme!! (ref task T13PRO-121 )
        n_hidden_layers=1,
        n_trajectory_per_policy=1,
        show_plot=show_plot_unless_CI_server_runned(False),
        )

    rudder_main(test_spec)
