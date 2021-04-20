import torch

class LstmCellRudder(torch.nn.Module):

    def __init__(self, n_positions, n_actions, hidden_size, n_lstm_layers, device):
        super(LstmCellRudder, self).__init__()

        self.hidden_size = hidden_size
        self.input_dim = n_positions + n_actions
        self.device = device

        self.lstm = torch.nn.LSTMCell(input_size=self.input_dim, hidden_size=self.hidden_size).to(device)
        self.fc_out = torch.nn.Linear(self.hidden_size, 1)

        self.init_weights()

    def forward(self, observation, action, ht, ct):
        x_t = torch.cat([observation, action])

        # sera passé dans le forward lors de la phase de test
        '''ht = torch.zeros((x.shape[0], self.hidden_size), dtype=torch.float32, device=self.device)
        ct = torch.zeros((x.shape[0], self.hidden_size), dtype=torch.float32, device=self.device)

        state_cell = (ht, ct)

        output = []

        for x_t in torch.chunk(x, x.shape[1], dim=1):
            ht, ct = self.lstm(x_t.squeeze(1), state_cell)
            output.append(self.fc_out(ht))

        net_out = torch.stack(output, 1)'''

        ht, ct = self.lstm(x_t, (ht, ct))
        return self.fc_out(ht)

