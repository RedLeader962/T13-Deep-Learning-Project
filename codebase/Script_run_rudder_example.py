from codebase import Rudder as rd
import torch
import numpy as np

# Prepare some random generators for later
rnd_gen = np.random.RandomState(seed=123)
_ = torch.manual_seed(123)

# Create environment
n_positions = 13
env = rd.Environment(n_samples=1000, max_timestep=50, n_positions=13, rnd_gen=rnd_gen)

# Load data
batch_size = 10
env_loader = torch.utils.data.DataLoader(env, batch_size=batch_size)

# Create Network
device = "cuda:0" if torch.cuda.is_available() else "cpu"
net = rd.Net(n_positions=n_positions, n_actions=2, n_lstm=16)
_ = net.to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)

update = 0
n_updates = 5000
running_loss = 100.

while update < n_updates:
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
        outputs = net(observations=observations, actions=actions)

        # Calculate loss, do backward pass, and update
        loss = rd.lossfunction(outputs[..., 0], rewards)
        loss.backward()
        running_loss = running_loss * 0.99 + loss * 0.01
        optimizer.step()
        update += 1

    print(f'Data shape : State {data[0].shape}, Actions {data[1].shape}, Rewards {data[2].shape}')

    print(f"Epoch : {update/len(env_loader)}, Loss: {loss:8.4f}")
