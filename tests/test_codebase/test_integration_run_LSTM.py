# coding=utf-8

import pytest

from script.general_utils import show_plot_unless_CI_server_runned


def test_Script_run_LSTM_PASS():
    from script.experiment_spec import RudderExperimentSpec
    from script.Script_run_LSTM import main as LSTM_main

    test_spec = RudderExperimentSpec(
        n_epoches=2,
        env_batch_size=10,
        loader_batch_size=4,
        show_plot=show_plot_unless_CI_server_runned(False),
        )

    LSTM_main(test_spec)
