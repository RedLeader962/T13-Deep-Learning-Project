import torch

class LSTM_Rudder(torch.nn.Module):
    def __init__(self, n_positions, n_actions, hidden_size, n_lstm_layers):
        super(LSTM_Rudder, self).__init__()

        self.hidden_size = hidden_size
        input_dim = n_positions + n_actions

        self.lstm1 = torch.nn.LSTM(input_size=input_dim, hidden_size=self.hidden_size, num_layers=n_lstm_layers, batch_first=True)

        # After the LSTM layer, we add a fully connected output layer
        # Output size est tjrs à 1 car on veut juste avoir le reward associé à l'état seulement !
        self.fc_out = torch.nn.Linear(self.hidden_size, 1)

        self.init_weights()

        '''# This will create an LSTM layer where we will feed the concatenate
        self.lstm1 = LSTMLayer(
            in_features=input_dim, out_features= n_lstm_layers, inputformat='NLC',
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
        self.fc_out = torch.nn.Linear(n_lstm_layers, 1)'''

    def forward(self, observations, actions, hs):

        x = torch.cat([observations, actions], dim=-1)

        # h_n représente ce que le LSTM a vu dans le passé
        lstm_out, hs = self.lstm1(x, hs)
        net_out = self.fc_out(lstm_out)
        #net_out = net_out.squeeze(-1)
        #lstm_out, _ = self.lstm1(x, return_all_seq_pos=True)  # return predictions for all sequence positions

        return net_out, hs

    def compute_loss(self, r_predicted, r_expected):

        # Main task: predicting return at last timestep.
        # Essentiellement c'est le calcul de MSE

        # Retourne le dernier output du LSTM qui représente le reward final de la trajectoire.
        main_loss = torch.mean(r_predicted[:, -1] - r_expected) ** 2

        # Auxiliary task: predicting final return at every timestep ([..., None] is for correct broadcasting)
        # Prediction détient une dimension de plus alors il ajoute une dimensions avec returns[..., None]
        # Ça revient à faire returns[:, None] en une dimension

        aux_loss = torch.mean(r_predicted[:, :] - r_expected[..., None]) ** 2

        # Combine losses
        # Ça permet essentiellement de réduire le problème du vanishing gradient. C'est la même idée que dans ResNet
        # À ne pas utiliser si on a de courte trajectoire !
        loss = main_loss #+ aux_loss * 0.5
        return loss





    def init_weights(self):

        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_normal_(module.weight, 1)
                torch.nn.init.normal_(module.bias)