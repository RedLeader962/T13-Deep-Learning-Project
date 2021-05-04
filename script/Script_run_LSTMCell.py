import dataclasses

import torch

from codebase import rudder as rd
from experiment_runner.constant import TEST_EXPERIMENT_RUN_DIR
from experiment_runner.test_related_utils import check_testspec_flag_and_setup_spec
from experiment_runner.experiment_spec import RudderLstmExperimentSpec


def main(spec: RudderLstmExperimentSpec) -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Create environment
    env = rd.Environment(env_name=spec.env_name,
                         batch_size=spec.env_batch_size,
                         n_trajectories=spec.env_n_trajectories,
                         perct_optimal=spec.env_perct_optimal,
                         )

    # Create Network
    n_lstm_layers = 1
    network = rd.LstmCellRudder(n_states=env.n_states,
                                n_actions=env.n_actions,
                                hidden_size=spec.model_hidden_size,
                                device=device,
                                init_weights=True).to(device)

    optimizer = torch.optim.Adam(network.parameters(),
                                 lr=spec.optimizer_lr,
                                 weight_decay=spec.optimizer_weight_decay,
                                 )

    # Train LSTM
    rd.train_rudder(network, optimizer,
                    n_epoches=spec.n_epoches,
                    env=env,
                    show_gap=100,
                    device=device,
                    show_plot=spec.show_plot)


if __name__ == '__main__':

    user_spec = RudderLstmExperimentSpec(
        env_name="CartPole-v1",
        env_batch_size=1000,
        model_hidden_size=35,
        env_n_trajectories=500,
        env_perct_optimal=0.7,
        n_epoches=2,
        optimizer_weight_decay=1e-2,
        optimizer_lr=1e-3,
        show_plot=True,
        seed=None,
        experiment_tag='Manual Run',
        )

    test_spec = dataclasses.replace(user_spec,
                                    n_epoches=2,
                                    env_batch_size=20,
                                    show_plot=False,
                                    root_experiment_dir=TEST_EXPERIMENT_RUN_DIR,
                                    experiment_tag='Test Run',
                                    )

    theSpec, _ = check_testspec_flag_and_setup_spec(user_spec, test_spec)
    main(theSpec)
