# coding=utf-8
from experiment_runner.constant import TEST_EXPERIMENT_RUN_DIR
from experiment_runner.test_related_utils import show_plot_unless_CI_server_runned


def test_Script_run_LSTMCell_PASS():
    from experiment_runner.experiment_spec import RudderLstmExperimentSpec
    from script.Script_run_LSTMCell import main as LSTMCell_main

    test_spec = RudderLstmExperimentSpec(
        env_name="CartPole-v1",
        env_batch_size=10,
        model_hidden_size=35,
        env_n_trajectories=500,
        env_perct_optimal=0.7,
        n_epoches=2,
        optimizer_weight_decay=1e-2,
        optimizer_lr=1e-3,
        show_plot=show_plot_unless_CI_server_runned(False),
        root_experiment_dir=TEST_EXPERIMENT_RUN_DIR,
        experiment_tag='Test Run',
        )

    LSTMCell_main(test_spec)
