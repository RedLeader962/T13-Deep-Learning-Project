import dataclasses

import torch

from codebase import rudder as rd
from experiment_runner.constant import TEST_EXPERIMENT_RUN_DIR
from experiment_runner.test_related_utils import check_testspec_flag_and_setup_spec
from experiment_runner.experiment_spec import RudderLstmExperimentSpec


def main(spec: RudderLstmExperimentSpec) -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    spec.setup_run_dir()

    lr = spec.optimizer_lr
    n_trajectories = spec.env_n_trajectories
    percet_optimal = spec.env_perct_optimal

    # Create environment
    env = rd.Environment(env_name=spec.env_name,
                         batch_size=spec.env_batch_size,
                         n_trajectories=n_trajectories,
                         perct_optimal=percet_optimal,
                         )

    hidden_size = spec.model_hidden_size

    network = rd.LstmRudder(n_states=env.n_states,
                            n_actions=env.n_actions,
                            hidden_size=hidden_size,
                            n_lstm_layers=1,
                            device=device).to(device)
    # Save LSTM
    network.save_model(spec.experiment_path,
                       f'{hidden_size}_{lr}_{n_trajectories}_{percet_optimal}')

    # Load LSTM
    network.load_model(spec.experiment_path,
                       f'{hidden_size}_{lr}_{n_trajectories}_{percet_optimal}')

    # Create Network
    network = rd.LstmCellRudder(n_states=env.n_states,
                                n_actions=env.n_actions,
                                hidden_size=hidden_size,
                                device=device,
                                init_weights=True).to(device)

    # Save LSTM
    network.save_model(spec.experiment_path, f'{hidden_size}_{lr}_{n_trajectories}_{percet_optimal}')

    # Load LSTM
    network.load_lstm_model(spec.experiment_path, f'{hidden_size}_{lr}_{n_trajectories}_{percet_optimal}')


if __name__ == '__main__':

    user_spec = RudderLstmExperimentSpec(
        env_name="CartPole-v1",
        env_batch_size=8,
        model_hidden_size=15,
        env_n_trajectories=300,
        env_perct_optimal=0.9,
        n_epoches=250,
        optimizer_weight_decay=1e-2,
        optimizer_lr=0.02,
        show_plot=True,
        # seed=42,
        seed=None,
        experiment_tag='Manual Run',
        )

    test_spec = dataclasses.replace(user_spec,
                                    env_batch_size=8,
                                    model_hidden_size=15,
                                    env_n_trajectories=10,
                                    env_perct_optimal=0.5,
                                    n_epoches=20,
                                    show_plot=False,
                                    root_experiment_dir=TEST_EXPERIMENT_RUN_DIR,
                                    experiment_tag='Test Run',
                                    )

    theSpec, _ = check_testspec_flag_and_setup_spec(user_spec, test_spec)
    main(theSpec)
