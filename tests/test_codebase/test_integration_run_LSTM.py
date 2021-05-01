# coding=utf-8

from experiment_runner.test_related_utils import show_plot_unless_CI_server_runned


def test_Script_run_LSTM_PASS():
    from experiment_runner.experiment_spec import RudderLstmExperimentSpec
    from script.Script_run_LSTM import main as LSTM_main

    test_spec = RudderLstmExperimentSpec(
        env_name="CartPole-v1",
        env_batch_size=8,
        model_hidden_size=15,
        env_n_trajectories=10,
        env_perct_optimal=0.5,
        rew_factor=0.1,
        n_epoches=20,
        optimizer_weight_decay=1e-2,
        optimizer_lr=1e-3,
        show_plot=show_plot_unless_CI_server_runned(False),
        seed=42,
        )

    print(f"\n{test_spec}\n")

    LSTM_main(test_spec)
