import torch

class LstmCellRudder(torch.nn.Module):

    def __init__(self, n_positions, n_actions, hidden_size, n_lstm_layers=1, device='cpu'):
        super(LstmCellRudder, self).__init__()

        self.hidden_size = hidden_size
        self.input_dim = n_positions + n_actions
        self.device = device

        self.lstm = torch.nn.LSTMCell(input_size=self.input_dim, hidden_size=self.hidden_size).to(device)
        self.fc_out = torch.nn.Linear(self.hidden_size, 1)

        self.init_weights()

    def forward(self, observation, action, hs = None):
        x_t = torch.cat([observation, action], dim=-1)

        # sera passé dans le forward lors de la phase de test
        ht = torch.zeros((x_t.shape[0], self.hidden_size), dtype=torch.float32, device=self.device)
        ct = torch.zeros((x_t.shape[0], self.hidden_size), dtype=torch.float32, device=self.device)

        output = []

        for x_t in torch.chunk(x_t, x_t.shape[1], dim=1):
            ht, ct = self.lstm(x_t.squeeze(1), (ht, ct))
            output.append(self.fc_out(ht))

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