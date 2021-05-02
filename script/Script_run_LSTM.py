import dataclasses

import torch

from codebase import rudder as rd
from experiment_runner.test_related_utils import (
    check_testspec_flag_and_setup_spec,
    )
from experiment_runner.experiment_runner_utils import ExperimentResults
from experiment_runner.experiment_spec import RudderLstmExperimentSpec
import matplotlib.pyplot as plt

def main(spec: RudderLstmExperimentSpec) -> ExperimentResults:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if spec.seed:
        torch.manual_seed(spec.seed)

    # Create environment
    env_n_trajectories = spec.env_n_trajectories
    env_perct_optimal = spec.env_perct_optimal
    model_hidden_size = spec.model_hidden_size
    optimizer_lr = spec.optimizer_lr

    env = rd.Environment(env_name=spec.env_name,
                         batch_size=spec.env_batch_size,
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

    optimizer = torch.optim.Adam(network.parameters(), lr=optimizer_lr, weight_decay=spec.optimizer_weight_decay)

    # Train LSTM
    loss_train, loss_test = rd.train_rudder(network, optimizer,
                                            n_epoches=spec.n_epoches,
                                            env=env,
                                            show_gap=25,
                                            device=device,
                                            show_plot=spec.show_plot,
                                            print_to_consol=spec.print_to_consol,
                                            )

    if spec.show_plot:
        rd.plot_lstm_loss(loss_train=loss_train, loss_test=loss_test)
        plt.savefig(f'lstm_fig_loss_{model_hidden_size}_{optimizer_lr}_{env_n_trajectories}_{env_perct_optimal}.jpg')
        plt.show()

    network.save_model(env.gym, f'{model_hidden_size}_{optimizer_lr}_{env_n_trajectories}_{env_perct_optimal}')

    return ExperimentResults(loss_train=loss_train, loss_test=loss_test)


if __name__ == '__main__':

    user_spec = RudderLstmExperimentSpec(
        env_name="CartPole-v1",
        env_batch_size=8,
        model_hidden_size=35,
        env_n_trajectories=2500,
        env_perct_optimal=0.2,
        n_epoches=250,
        optimizer_weight_decay=1e-2,
        optimizer_lr=0.02,
        show_plot=True,
        # seed=42,
        seed=None,
        )

    test_spec = dataclasses.replace(user_spec,
                                    env_batch_size=8,
                                    model_hidden_size=15,
                                    env_n_trajectories=10,
                                    env_perct_optimal=0.5,
                                    n_epoches=20,
                                    show_plot=False,
                                    )

    theSpec, _ = check_testspec_flag_and_setup_spec(user_spec, test_spec)
    main(theSpec)
