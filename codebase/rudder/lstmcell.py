from .utils import get_env_path
import os

import torch

class LstmCellRudder(torch.nn.Module):
    '''
    Class which represents a different implementation of PyTorch LSTM class
    '''

    def __init__(self, n_positions, n_actions, hidden_size, n_lstm_layers=1, device='cpu', init_weights=False):
        super(LstmCellRudder, self).__init__()

        self.hidden_size = hidden_size
        self.input_dim = n_positions + n_actions
        self.device = device
        self.file_name = 'lstmcell'

        self.lstm = torch.nn.LSTMCell(input_size=self.input_dim, hidden_size=self.hidden_size).to(device)
        self.fc_out = torch.nn.Linear(self.hidden_size, 1)

        if init_weights:
            self.init_weights()

    def forward(self, observation, action, length = None, hs = None):
        x_t = torch.cat([observation, action], dim=-1)

        batch_size = len(x_t)
        self.reset_cell_hidden_state(batch_size)

        output = []

        for x_t in torch.chunk(x_t, x_t.shape[1], dim=1):
            self.hidden_state, self.cell_state = self.lstm(x_t.squeeze(1), (self.hidden_state, self.cell_state))
            output.append(self.fc_out(self.hidden_state))

        net_out = torch.stack(output, 1)

        return net_out

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

    def reset_cell_hidden_state(self, batch_size):
        self.hidden_state = torch.zeros((batch_size, self.hidden_size), dtype=torch.float32, device=self.device)
        self.cell_state = torch.zeros((batch_size, self.hidden_size), dtype=torch.float32, device=self.device)

        torch.nn.init.xavier_normal_(self.hidden_state)
        torch.nn.init.xavier_normal_(self.cell_state)

    def save_model(self, env):
        """
        :param env: Gym environnment
        """
        env_path = get_env_path(env)
        file_path = os.path.join(env_path, f'{self.file_name}.pt')
        torch.save(self.state_dict(), file_path)
        print(self.file_name, 'saved in', env_path)

    def load_lstm_model(self, env):
        """
         :param env: Gym environnment
         """
        env_path = get_env_path(env)
        file_path = os.path.join(env_path, 'lstm.pt')
        self._lstm_to_lstmcell(file_path)

        print('Network', self.file_name, 'loaded from source file lstm')

        return None

    def _lstm_to_lstmcell(self, path : str):
        """
        Take the weights of the trained LSTM and assign them to the LSTMCell. LSTMCell is used on PPO to get
        the expected return at each timestep.
        :param path: path associated with the environnement
        """
        param_lstm = torch.load(path)
        param_lstmcell = self.state_dict()

        state_dict = {}
        for w1, w2 in zip(param_lstm, param_lstmcell):
            shape_w1 = param_lstm[w1].shape
            shape_w2 = param_lstmcell[w2].shape

            assert shape_w1 == shape_w2, f'Lstm a une dimension de {shape_w1} alors que LSTMCell a une dimension de {shape_w2}'

            state_dict[w2] = param_lstm[w1]

        self.load_state_dict(state_dict)

        return None