# coding=utf-8
from experiment_runner.test_related_utils import show_plot_unless_CI_server_runned


def test_Script_run_ppo_with_rudder_main_PASS():
    from experiment_runner.experiment_spec import PpoExperimentSpec
    from script.Script_run_ppo_with_rudder import main as ppo_with_rudder_main

    test_spec = PpoExperimentSpec(
        steps_by_epoch=500,
        n_epoches=2,
        hidden_dim=6,
        n_hidden_layers=2,
        show_plot=show_plot_unless_CI_server_runned(False),
        n_trajectory_per_policy=1)

    ppo_with_rudder_main(test_spec)
