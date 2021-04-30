# coding=utf-8

from experiment_runner.test_related_utils import show_plot_unless_CI_server_runned


def test_Script_run_LSTM_PASS():
    from experiment_runner.experiment_spec import RudderExperimentSpec
    from script.Script_save_load_LSTM import main as save_load_LSTM_main

    test_spec = RudderExperimentSpec(
        n_epoches=None,
        env_batch_size=None,
        loader_batch_size=None,
        show_plot=show_plot_unless_CI_server_runned(False),
        )

    save_load_LSTM_main(test_spec)
