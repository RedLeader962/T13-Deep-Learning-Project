# coding=utf-8
from script.general_utils import show_plot_unless_CI_server_runned


def test_Script_run_ppo_main_PASS():
    from script.experiment_spec import PpoExperimentSpec
    from script.Script_run_ppo import main as ppo_main

    test_spec = PpoExperimentSpec(
        steps_by_epoch=150,
        n_epoches=2,
        hidden_dim=6,
        n_hidden_layers=2,
        show_plot=show_plot_unless_CI_server_runned(False),
        n_trajectory_per_policy=1)

    ppo_main(test_spec)
