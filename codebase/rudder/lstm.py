from .utils import get_cherypicked_env_path
import os
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch


class LstmRudder(torch.nn.Module):

    def __init__(self, n_states, n_actions, hidden_size, n_lstm_layers=1, device='cpu'):
        super(LstmRudder, self).__init__()

        self.hidden_size = hidden_size
        self.input_dim = n_states + n_actions
        self.device = device
        self.file_name = 'lstm'

        self.lstm = torch.nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_size, num_layers=n_lstm_layers,
                                  batch_first=True)

        self.fc_out = torch.nn.Linear(self.hidden_size, 1)

        self.init_weights()

    def forward(self, observations, actions, length, hs=None):

        trajectory_length = length.cpu().numpy()

        x = torch.cat([observations, actions], dim=-1)
        o_a_pack = pack_padded_sequence(x, trajectory_length, batch_first=True, enforce_sorted=False)

        lstm_out, hs = self.lstm(o_a_pack, hs)

        out = pad_packed_sequence(lstm_out, batch_first=True, padding_value=0)[0]

        net_out = self.fc_out(out)

        return net_out

    def compute_loss(self, r_predicted, r_expected):
        """
        Loss original : https://github.com/widmi/rudder-a-practical-tutorial/blob/master/tutorial.ipynb
        :param r_predicted: Expected return at the end of trajectory
        :param r_expected: Predicted during the trajectory
        :return:
        """
        # Retourne le dernier output du LSTM qui représente le reward final de la trajectoire.
        main_loss = torch.mean(r_predicted[:, -1] - r_expected) ** 2
        aux_loss = torch.mean(r_predicted[:, :] - r_expected[..., None]) ** 2

        # Combine losses
        loss = main_loss + aux_loss*0.3
        return loss

    def init_weights(self):
        _ = torch.manual_seed(123)

        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_normal_(module.weight, 1)
                torch.nn.init.normal_(module.bias)

    def save_model(self, experiment_run_path: str, model_spec: str):
        file_path = os.path.join(experiment_run_path, f'{self.file_name}_{model_spec}.pt')
        torch.save(self.state_dict(), file_path)
        print(self.file_name, 'saved in', experiment_run_path)

    def load_model(self, experiment_run_path: str, model_spec, ):
        lstm_dict = torch.load(os.path.join(experiment_run_path, f'{self.file_name}_{model_spec}.pt'))
        self.load_state_dict(lstm_dict)
        print('Network', self.file_name, 'loaded')
