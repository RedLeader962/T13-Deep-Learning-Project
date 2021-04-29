# coding=utf-8

import pytest

from script.general_utils import show_plot_unless_CI_server_runned
from tests.test_codebase.test_script_run_from_command_line import is_run_on_TeamCity_CI_server


# @pytest.mark.skipif(not is_run_on_TeamCity_CI_server, reason="Pour ne pas écraser les données dans le experiment/ dir")
@pytest.mark.skip(reason="Just mute")
def test_rudder_generate_load_trajectories_PASS():
    from script.experiment_spec import PpoExperimentSpec
    from script.Script_generate_save_load_trajectories import main as rudder_main

    test_spec = PpoExperimentSpec(
        steps_by_epoch=10,
        n_epoches=2,
        hidden_dim=18,  # ‹‹‹ (CRITICAL) todo:fixme!! (ref task T13PRO-121 )
        n_hidden_layers=1,
        n_trajectory_per_policy=2,
        show_plot=show_plot_unless_CI_server_runned(False),
        )

    rudder_main(test_spec)
