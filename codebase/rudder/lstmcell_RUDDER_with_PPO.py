from typing import Optional

from .utils import get_cherypicked_env_path
import os

import torch

class LstmCellRudder_with_PPO(torch.nn.Module):
    """
    Class to run within PPO. The difference with LSTMCell is that this version can be played with PPO
    """

    def __init__(self, n_states, n_actions, hidden_size, device='cpu', init_weights=False):
        super(LstmCellRudder_with_PPO, self).__init__()

        self.hidden_size = hidden_size
        self.input_dim = n_states + n_actions
        self.device = device
        self.file_name = 'lstmcell'
        self.n_actions = n_actions

        self.lstm = torch.nn.LSTMCell(input_size=self.input_dim, hidden_size=self.hidden_size).to(device)
        self.fc_out = torch.nn.Linear(self.hidden_size, 1)

        if init_weights:
            self.init_weights()

        self.one_hot_action = torch.zeros(n_actions, dtype=torch.float32)

        self.lstm.eval()

    def forward(self, observation, action):
        self.one_hot_action[action] = 1.0

        x_t = torch.cat([observation, self.one_hot_action]).unsqueeze(0)
        self.hidden_state, self.cell_state = self.lstm(x_t.squeeze(1), (self.hidden_state, self.cell_state))

        output = self.fc_out(self.hidden_state)

        self.one_hot_action = torch.zeros(self.n_actions, dtype=torch.float32)

        return output

    def compute_loss(self, r_predicted, r_expected):

        main_loss = torch.mean(r_predicted[:, -1] - r_expected) ** 2
        aux_loss = torch.mean(r_predicted[:, :] - r_expected[..., None]) ** 2

        # Combine losses
        loss = main_loss  + aux_loss * 0.5
        return loss

    def init_weights(self):
        _ = torch.manual_seed(123)

        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_normal_(module.weight, 1)
                torch.nn.init.normal_(module.bias)

    def reset_cell_hidden_state(self):
        self.hidden_state = torch.zeros((1, self.hidden_size), dtype=torch.float32, device=self.device)
        self.cell_state = torch.zeros((1, self.hidden_size), dtype=torch.float32, device=self.device)

        torch.nn.init.xavier_normal_(self.hidden_state)
        torch.nn.init.xavier_normal_(self.cell_state)

    def save_model(self, experiment_run_path: str, model_spec: str):
        file_path = os.path.join(experiment_run_path, f'{self.file_name}_{model_spec}.pt')
        torch.save(self.state_dict(), file_path)
        print(self.file_name, 'saved in', experiment_run_path)

    def load_lstm_model(self, experiment_run_path: str, model_name: Optional[str]):
        if model_name is None:
            file_name = f'{self.file_name}.pt'
            file_path = os.path.join(experiment_run_path, f'{self.file_name}.pt')
        else:
            file_name = f'{model_name}.pt'
            file_path = os.path.join(experiment_run_path, file_name)
        self._lstm_to_lstmcell(file_path)

        print('Network', file_name, 'loaded from source file lstm')
        return None

    def load_selected_lstm_model(self, model_full_path: str):
        self._lstm_to_lstmcell(model_full_path)

        print('Network', model_full_path, 'loaded from source file lstm')
        return None

    def _lstm_to_lstmcell(self, path : str):
        """
        Take the weights of the trained LSTM and assign them to the LSTMCell. LSTMCell is used on PPO to get
        the expected return at each timestep.
        :param path: path associated with the environnement
        """
        param_lstm = torch.load(path, map_location=torch.device(self.device))
        param_lstmcell = self.state_dict()

        state_dict = {}
        for w1, w2 in zip(param_lstm, param_lstmcell):
            shape_w1 = param_lstm[w1].shape
            shape_w2 = param_lstmcell[w2].shape

            assert shape_w1 == shape_w2, f'Lstm a une dimension de {shape_w1} alors que LSTMCell a une dimension de {shape_w2}'

            state_dict[w2] = param_lstm[w1]

        self.load_state_dict(state_dict)

        return None