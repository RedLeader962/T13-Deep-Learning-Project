import dataclasses

import torch
from torch.utils.data import DataLoader
import numpy as np

from codebase import rudder as rd
from script.general_utils import check_testspec_flag_and_setup_spec
from script.experiment_spec import RudderExperimentSpec
import matplotlib.pyplot as plt

def main(spec: RudderExperimentSpec) -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    lr = 0.01 # 0.02
    n_trajectories = 2500 # 3200
    percet_optimal = 0.8 # 0.4

    # Create environment
    # Environment : CartPole-v1, MountainCar-v0, LunarLander-v2
    env = rd.Environment("CartPole-v1", batch_size=8, n_trajectories=n_trajectories,
                         perct_optimal=percet_optimal, reward_adjust=1)

    # Create Network
    n_lstm_layers = 1
    hidden_size = 35    #35
    network = rd.LstmRudder(n_states=env.n_states, n_actions=env.n_actions,
                            hidden_size=hidden_size, n_lstm_layers=n_lstm_layers, device=device).to(device)

    #network.load_model(env.gym, f'{hidden_size}_{lr}_{n_trajectories}_{percet_optimal}')

    optimizer = torch.optim.Adam(network.parameters(), lr=lr, weight_decay=1e-2)

    # Train LSTM
    loss_train, loss_test = rd.train_rudder(network, optimizer, n_epoches=spec.n_epoches, env=env, show_gap=50, device=device,
                    show_plot=spec.show_plot)

    network.save_model(env.gym, f'{hidden_size}_{lr}_{n_trajectories}_{percet_optimal}')

    not_show = True
    if not_show:
        rd.plot_lstm_loss(loss_train=loss_train, loss_test=loss_test)
        plt.savefig(f'lstm_fig_loss_{hidden_size}_{lr}_{n_trajectories}_{percet_optimal}.jpg')
        plt.show()


if __name__ == '__main__':

    user_spec = RudderExperimentSpec(
        n_epoches=350, # un peu plus que 100 époques
        env_batch_size=100,
        loader_batch_size=10,
        show_plot=True,
        )

    test_spec = dataclasses.replace(user_spec,
                                    n_epoches=2,
                                    env_batch_size=8,
                                    show_plot=False,
                                    )

    theSpec, _ = check_testspec_flag_and_setup_spec(user_spec, test_spec)
    main(theSpec)
