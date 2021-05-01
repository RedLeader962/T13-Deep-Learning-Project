# coding=utf-8
from experiment_runner.test_related_utils import show_plot_unless_CI_server_runned


def test_Script_run_ppo_main_PASS():
    from experiment_runner.experiment_spec import PpoExperimentSpec
    from script.Script_run_ppo import main as ppo_main

    test_spec = PpoExperimentSpec(
        env_name='CartPole-v1',
        steps_by_epoch=1000,
        n_epoches=2,
        hidden_dim=18,
        n_hidden_layers=1,
        show_plot=show_plot_unless_CI_server_runned(False),
        n_trajectory_per_policy=1)

    ppo_main(test_spec)
