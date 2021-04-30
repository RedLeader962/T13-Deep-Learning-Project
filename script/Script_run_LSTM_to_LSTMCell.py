import dataclasses

import torch

from codebase import rudder as rd
from experiment_runner.test_related_utils import check_testspec_flag_and_setup_spec
from experiment_runner.experiment_spec import RudderLstmExperimentSpec


def main(spec: RudderLstmExperimentSpec) -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Create environment
    env = rd.Environment(env_name=spec.env_name, batch_size=spec.env_batch_size, n_trajectories=spec.env_n_trajectories,
                         perct_optimal=spec.env_perct_optimal)


    # Create LSTM Network
    n_lstm_layers = 1  # Note: Hardcoded because our lstmCell implementation doesn't use 2 layers
    lstm = rd.LstmRudder(n_states=env.n_states, n_actions=env.n_actions,
                         hidden_size=spec.model_hidden_size, n_lstm_layers=n_lstm_layers,
                         device=device).to(device)

    # print(lstm)

    optimizer = torch.optim.Adam(lstm.parameters(), lr=spec.optimizer_lr, weight_decay=spec.optimizer_weight_decay)

    # Train and save LSTM in the gym environnement
    rd.train_rudder(lstm, optimizer, n_epoches=spec.n_epoches, env=env, show_gap=100, device=device,
                    show_plot=spec.show_plot)

    lstm.save_model(env.gym, f'{spec.model_hidden_size}_{spec.optimizer_lr}_{spec.env_n_trajectories}_{spec.env_perct_optimal}')

    # Create LSTMCell Network
    lstmcell = rd.LstmCellRudder(n_states=env.n_states, n_actions=env.n_actions, hidden_size=spec.model_hidden_size,
                                 device=device, init_weights=False).to(device)

    # Load LSTMCell
    lstmcell.load_lstm_model(env.gym, f'{spec.model_hidden_size}_{spec.optimizer_lr}_{spec.env_n_trajectories}_{spec.env_perct_optimal}')

    # Train LSTMCell
    optimizer = torch.optim.Adam(lstmcell.parameters(), lr=spec.optimizer_lr, weight_decay=spec.optimizer_weight_decay)
    rd.train_rudder(lstmcell, optimizer, n_epoches=spec.n_epoches, env=env, show_gap=100, device=device,
                    show_plot=spec.show_plot)


if __name__ == '__main__':

    user_spec = RudderLstmExperimentSpec(
        env_name="CartPole-v1",
        env_batch_size=8,
        model_hidden_size=10,
        env_n_trajectories=200,
        env_perct_optimal=0.9,
        env_rew_factor=0.1,
        n_epoches=1,
        optimizer_weight_decay=1e-4,
        optimizer_lr=0.01,
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
