# coding=utf-8
import os
from typing import Tuple

import pytest
import torch
from matplotlib import pyplot as plt

# from experiment_runner.experiment_runner_organizer import (
#     # get_batch_run_dir,
#     # get_spec_run_dir,
#     # get_spec_run_path,
#     # setup_run_dir,
#     ExperimentOrganizer
#     )
from experiment_runner.constant import TEST_EXPERIMENT_RUN_DIR
from experiment_runner.test_related_utils import show_plot_unless_CI_server_runned
from experiment_runner.experiment_spec import ExperimentSpec, RudderLstmExperimentSpec, generate_batch_run_dir_name
from codebase import rudder as rd

EXPERIMENT_RUNNER_ORGANIZER_DEV_DONE = True


@pytest.fixture(scope="function")
def setup_test_spec() -> Tuple[ExperimentSpec, ExperimentSpec]:
    basic_exp_spec = ExperimentSpec(
        spec_name='My cool the test',
        experiment_tag='theTest',
        spec_idx=1,
        is_batch_spec=False,
        root_experiment_dir=TEST_EXPERIMENT_RUN_DIR,
        )

    basic_exp_spec2 = ExperimentSpec(
        spec_name='My cool the test 2',
        experiment_tag='the great Test',
        spec_idx=2,
        is_batch_spec=False,
        root_experiment_dir=TEST_EXPERIMENT_RUN_DIR,
        )

    return basic_exp_spec, basic_exp_spec2


# @pytest.fixture(scope="function")
# def setup_test_batch_spec():
#     basic_exp_spec = ExperimentSpec(
#         spec_name='My cool the test',
#         experiment_tag='theTest',
#         batch_tag='the batch',
#         spec_idx=1,
#         is_batch_spec=True
#         )
#
#     basic_exp_spec2 = ExperimentSpec(
#         spec_name='My cool the test 2',
#         experiment_tag='theTest',
#         is_batch_spec=True
#         )
#
#     return basic_exp_spec, basic_exp_spec2


def test_get_experiment_run_are_unique_path(setup_test_spec):
    basic_exp_spec, basic_exp_spec2 = setup_test_spec

    the_path1 = basic_exp_spec.get_spec_run_dir()
    the_path2 = basic_exp_spec2.get_spec_run_dir()

    assert the_path1 != the_path2


def test_get_experiment_run_path(setup_test_spec):
    basic_exp_spec, basic_exp_spec2 = setup_test_spec

    the_path = basic_exp_spec.get_spec_run_dir()

    print(f"\n\n{the_path}\n\n")


def test_get_experiment_BATCH_path(setup_test_spec):
    basic_exp_spec, basic_exp_spec2 = setup_test_spec

    basic_exp_spec.experiment_tag = 'theTest'
    basic_exp_spec.batch_tag = 'the batch'

    the_path = basic_exp_spec.get_spec_run_path()

    print(f"\n\n{the_path}\n\n")


@pytest.mark.skipif(condition=EXPERIMENT_RUNNER_ORGANIZER_DEV_DONE, reason="Development test")
def test_setup_run_dir_SINGLE_RUN_PASS(setup_test_spec):
    basic_exp_spec, basic_exp_spec2 = setup_test_spec

    # basic_exp_spec.experiment_path = get_spec_run_dir(basic_exp_spec)

    basic_exp_spec.setup_run_dir()                                                              # <-- add this


@pytest.mark.skipif(condition=EXPERIMENT_RUNNER_ORGANIZER_DEV_DONE, reason="Development test")
def test_setup_run_dir_BATCH_PASS(setup_test_spec):
    basic_exp_spec, basic_exp_spec2 = setup_test_spec

    batch_tag = 'the batch'
    batch_run_dir = generate_batch_run_dir_name(batch_tag)                                      # <-- add this

    basic_exp_spec.configure_batch_spec(batch_tag=batch_tag, batch_dir=batch_run_dir)           # <-- add this
    basic_exp_spec2.configure_batch_spec(batch_tag=batch_tag, batch_dir=batch_run_dir)          # <-- add this

    basic_exp_spec.setup_run_dir()                                                              # <-- add this
    basic_exp_spec2.setup_run_dir()                                                             # <-- add this


@pytest.mark.skipif(condition=EXPERIMENT_RUNNER_ORGANIZER_DEV_DONE, reason="Development test")
def test_INTEGRATION_setup_run_dir_PASS():
    test_spec = RudderLstmExperimentSpec(
        experiment_tag='lstm manual test run',
        env_name="CartPole-v1",
        env_batch_size=8,
        model_hidden_size=15,
        env_n_trajectories=10,
        env_perct_optimal=0.5,
        n_epoches=20,
        optimizer_weight_decay=1e-2,
        optimizer_lr=1e-3,
        show_plot=show_plot_unless_CI_server_runned(False),
        seed=42,
        root_experiment_dir=TEST_EXPERIMENT_RUN_DIR,
        )

    print(f"\n{test_spec}\n")

    test_spec.setup_run_dir()                                                                   # <-- add this

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    env_n_trajectories = test_spec.env_n_trajectories
    env_perct_optimal = test_spec.env_perct_optimal
    model_hidden_size = test_spec.model_hidden_size
    optimizer_lr = test_spec.optimizer_lr

    env = rd.Environment(env_name=test_spec.env_name,
                         batch_size=test_spec.env_batch_size,
                         n_trajectories=env_n_trajectories,
                         perct_optimal=env_perct_optimal,
                         )

    # Create Network
    n_lstm_layers = 1  # Note: Hardcoded because our lstmCell implementation doesn't use 2 layers
    network = rd.LstmRudder(n_states=env.n_states,
                            n_actions=env.n_actions,
                            hidden_size=model_hidden_size,
                            n_lstm_layers=n_lstm_layers,
                            device=device, ).to(device)

    # print(network)

    optimizer = torch.optim.Adam(network.parameters(),
                                 lr=optimizer_lr,
                                 weight_decay=test_spec.optimizer_weight_decay)

    # Train LSTM
    loss_train, loss_test = rd.train_rudder(network, optimizer,
                                            n_epoches=test_spec.n_epoches,
                                            env=env,
                                            show_gap=25,
                                            device=device,
                                            show_plot=test_spec.show_plot,
                                            print_to_consol=test_spec.print_to_consol,
                                            )

    network.save_model(test_spec.experiment_path,                                                   # <-- add this
                       f'{model_hidden_size}_{optimizer_lr}_{env_n_trajectories}_{env_perct_optimal}')

    rd.plot_lstm_loss(loss_train=loss_train, loss_test=loss_test)
    plt.savefig(os.path.join(test_spec.experiment_path,                                             # <-- add this
                             f'lstm_fig_loss_{model_hidden_size}_{optimizer_lr}_{env_n_trajectories}_'
                             f'{env_perct_optimal}.jpg')
                )
    if test_spec.show_plot:
        plt.show()

    assert os.path.exists(os.path.relpath(test_spec.experiment_path))
