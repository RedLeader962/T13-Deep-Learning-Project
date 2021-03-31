import torch
from widis_lstm_tools.nn import LSTMLayer

class Net(torch.nn.Module):
    def __init__(self, n_positions, n_actions, n_lstm):
        super(Net, self).__init__()

        # This will create an LSTM layer where we will feed the concatenate
        self.lstm1 = LSTMLayer(
            in_features=n_positions + n_actions, out_features=n_lstm, inputformat='NLC',
            # cell input: initialize weights to forward inputs with xavier, disable connections to recurrent inputs
            w_ci=(torch.nn.init.xavier_normal_, False),
            # input gate: disable connections to forward inputs, initialize weights to recurrent inputs with xavier
            w_ig=(False, torch.nn.init.xavier_normal_),
            # output gate: disable all connection (=no forget gate) and disable bias
            w_og=False, b_og=False,
            # forget gate: disable all connection (=no forget gate) and disable bias
            w_fg=False, b_fg=False,
            # LSTM output activation is set to identity function
            a_out=lambda x: x
        )

        # After the LSTM layer, we add a fully connected output layer
        self.fc_out = torch.nn.Linear(n_lstm, 1)

    def forward(self, observations, actions):
        # Process input sequence by LSTM
        lstm_out, *_ = self.lstm1(torch.cat([observations, actions], dim=-1),
                                  return_all_seq_pos=True  # return predictions for all sequence positions
                                  )
        net_out = self.fc_out(lstm_out)
        return net_out