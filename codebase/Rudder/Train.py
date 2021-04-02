from .utils import *

def train_rudder(network_lstm, optimizer, epoches, data_loader, show_gap=5, device='cpu'):

    # Hidden state
    hs = None

    for epoch in range(epoches):
        track_loss = 0
        epoch += 1

        # Data contient 3 choses : Le tenseur d'états, le tenseur d'action et le tenseur de rewards
        for data in data_loader:

            # Essentiellement
            #   data contient 3 élément:
            #      Observation : qui est l'état du système. Dans ce cas-ci c'est la position du joueur
            #           (Batch Size X Longueur de la trajectoire X Position du joueur)
            #           ex. 10 batch size, trajectoire de 50 actions possible, et 13 positions possibles du joueur!

            #      Action : qui représente le onehot vector des actions jouées
            #           (Batch size X longueur de trajectoire X 2 actions possible). Je me demande l'avantage d'encoder l'action en one-hot ?

            #      Rewards     : un vecteur de la même longueur
            #           (10 Batch X 50 rewards possible sur la trajectoire). Ici, les rewards sont obtenus à la fin

            # Get samples
            observations, actions, rewards = data
            observations, actions, rewards = observations.to(device), actions.to(device), rewards.to(device)
            r_expected = rewards.sum(dim=1)

            # Reset gradients
            optimizer.zero_grad()

            # Get predicted reward form network
            r_predicted, hs = network_lstm(observations, actions, hs)
            hs = tuple([h.data for h in hs])

            # Compute loss, backpropagate gradient and update
            loss = network_lstm.compute_loss(r_predicted[...,0], r_expected)
            loss.backward()
            optimizer.step()

            # Track loss metric
            track_loss += loss

        if epoch % show_gap == 0:
            plot_reward(r_predicted, r_expected, epoch)

        # print(f'Data shape : State {data[0].shape}, Actions {data[1].shape}, Rewards {data[2].shape}')
        print(f"Epoch : {epoch}, loss mean: {track_loss / len(data_loader):8.4f}")
