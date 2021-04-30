# coding=utf-8

from experiment_runner.test_related_utils import show_plot_unless_CI_server_runned


def test_Script_run_LSTMCell_PASS():
    from experiment_runner.experiment_spec import RudderExperimentSpec
    from script.Script_run_LSTMCell import main as LSTMCell_main

    test_spec = RudderExperimentSpec(
        n_epoches=2,
        env_batch_size=10,
        loader_batch_size=8,
        show_plot=show_plot_unless_CI_server_runned(False),
        )

    LSTMCell_main(test_spec)
