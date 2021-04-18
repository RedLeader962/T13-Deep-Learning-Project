import torch


class LstmRudder(torch.nn.Module):

    def __init__(self, n_positions, n_actions, hidden_size, n_lstm_layers):
        super(LstmRudder, self).__init__()

        self.hidden_size = hidden_size
        input_dim = n_positions + n_actions

        self.lstm1 = torch.nn.LSTM(input_size=input_dim, hidden_size=self.hidden_size, num_layers=n_lstm_layers,
                                   batch_first=True)

        # Output size est tjrs à 1 car on veut juste avoir le reward associé à l'état seulement !
        self.fc_out = torch.nn.Linear(self.hidden_size, 1)

        self.init_weights()

    def forward(self, observations, actions, hs):
        x = torch.cat([observations, actions], dim=-1)

        # h_s représente la mémoire court terme du LSTM
        lstm_out, hs = self.lstm1(x, hs)
        net_out = self.fc_out(lstm_out)

        return net_out, hs

    def compute_loss(self, r_predicted, r_expected):

        # Retourne le dernier output du LSTM qui représente le reward final de la trajectoire.
        main_loss = torch.mean(r_predicted[:, -1] - r_expected) ** 2

        # Auxiliary task: predicting final return at every timestep ([..., None] is for correct broadcasting)
        # Prediction détient une dimension de plus alors il ajoute une dimensions avec returns[..., None]
        # Ça revient à faire returns[:, None] en une dimension

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
