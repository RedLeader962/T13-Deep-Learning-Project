import Rudder as rd
import torch
import numpy as np


# Prepare some random generators for later
rnd_gen = np.random.RandomState(seed=123)
_ = torch.manual_seed(123)

# Create environment
n_positions = 13
env = rd.Environment(n_samples=3000, max_timestep=50, n_positions=13, rnd_gen=rnd_gen)

# Load data
batch_size = 10 # Il y a donc 10 trajectoire dans une batch. Il y a 100 batch de 10 trajectoire chaque
env_loader = torch.utils.data.DataLoader(env, batch_size=batch_size)

# Create Network
device = "cuda:0" if torch.cuda.is_available() else "cpu"

n_lstm_layers = 2 # Il faut parfois jouer avec le nb de layer et le nb de hidden size
hidden_size = 40

net = rd.LSTM_Rudder(n_positions=n_positions, n_actions=2, hidden_size=hidden_size, n_lstm_layers=n_lstm_layers)
_ = net.to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)

update = 0
n_updates = 50000

hs = None
while update < n_updates:

    epoch = update / len(env_loader) + 1
    track_loss = 0

    # Data contient 3 choses : Le tenseur d'états, le tenseur d'action et le tenseur de rewards
    for data in env_loader:

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

        # Reset gradients
        optimizer.zero_grad()

        # Get outputs for network

        outputs, hs = net(observations, actions, hs)
        hs = tuple([h.data for h in hs])

        # Calculate loss, do backward pass, and update
        loss = rd.lossfunction(outputs[..., 0], rewards)
        track_loss += loss

        loss.backward()
        optimizer.step()
        update += 1

    predicted = np.round(outputs[:, -1].squeeze(-1).detach().cpu().numpy(), decimals=1)
    expected  = np.round(rewards.sum(dim=1).detach().cpu().numpy(), decimals=0)

    string = ''
    for o, r in zip(predicted, expected, ):
        string += str(o) + ' VS ' + str(int(r)) + ', '

    if update/len(env_loader) % 5 == 0:
        print('Example prediction VS actual reward: ', string)
        rd.plot_reward(predicted, expected, epoch)

    #print(f'Data shape : State {data[0].shape}, Actions {data[1].shape}, Rewards {data[2].shape}')

    print(f"Epoch : {epoch}, Epoch loss mean: {track_loss/len(env_loader):8.4f}")
