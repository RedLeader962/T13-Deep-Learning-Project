import torch
from torch.utils.data import Dataset
import numpy as np


class Environment(Dataset):

    def __init__(self, batch_size, max_timestep: int, n_positions: int, rnd_gen: np.random.RandomState):
        """Our simple 1D environment as PyTorch Dataset"""
        super(Environment, self).__init__()
        n_actions = 2
        self.batch_size = batch_size

        # C'est juste les règles du jeu et rien d'autre. Zéro position c'est la position de départ du joueur
        zero_position = int(np.ceil(n_positions / 2.))
        coin_position = zero_position + 2

        # Generate random action sequences. Génère une séquence d'action aléatoire de soit 0 ou 1.
        # Une ligne représente une séquence complète d'actions
        actions = np.asarray(rnd_gen.randint(low=0, high=2, size=(batch_size, max_timestep)), dtype=np.int)

        # Si l'agent a joué l'action 0 alors on prend l'identité, soit [1,0]
        # Si l'agent a joué l'action 1 alors on prend l'identité, soit [0,1]
        actions_onehot = np.identity(n_actions, dtype=np.float32)[actions]

        # Generate observations from action sequences
        # Ça permet de convertir une action 0 en -1 et l'action 1 reste là même. On va avoir une matrice de -1 et 1.
        # Ça permet de faire déplacer l'agent.
        actions[:] = (actions * 2) - 1

        # On génère 1000 samples dont la trajectoire est de 50 pas. Donc la simulation est de 50 steps au maximum
        observations = np.full(fill_value=zero_position, shape=(batch_size, max_timestep), dtype=np.int)

        #print(observations.shape)

        # On prend toutes les actions au temps 1
        for t in range(max_timestep - 1):
            action = actions[:, t]

            # Essentiellement ce que ça fait c'est que ça additionne les actions au observations et ça clip
            # les observations entre 0 et la position n-1, c'est juste pour la simulation et pas pour notre problème.
            observations[:, t + 1] = np.clip(observations[:, t] + action, 0, n_positions - 1)

        observations_onehot = np.identity(n_positions, dtype=np.float32)[observations]

        # On veut donc avoir ici la position du joueur
        #print(np.identity(n_positions, dtype=np.float32))
        #print(observations_onehot.shape)
        #print(observations.shape)

        # Calculate rewards (sum over coin position for all timesteps)
        rewards = np.zeros(shape=(batch_size, max_timestep), dtype=np.float32)
        rewards[:, -1] = observations_onehot[:, :, coin_position].sum(axis=1)

        self.actions = actions_onehot
        self.observations = observations_onehot
        self.rewards = rewards

    def __len__(self):
        return self.rewards.shape[0]

    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx], self.rewards[idx]
