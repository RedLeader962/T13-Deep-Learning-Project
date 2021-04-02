from not_code_reviewed import Rudder as rd
import torch
import numpy as np

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Prepare some random generators for later
rnd_gen = np.random.RandomState(seed=123)
_ = torch.manual_seed(123)

# Create environment
n_positions = 13
env = rd.Environment(batch_size=2000, max_timestep=50, n_positions=13, rnd_gen=rnd_gen)

# Load data
batch_size = 10
env_loader = torch.utils.data.DataLoader(env, batch_size=batch_size)

# Create Network
n_lstm_layers = 1
hidden_size = 40
network = rd.LSTM_Rudder(n_positions=n_positions, n_actions=2, hidden_size=hidden_size, n_lstm_layers=n_lstm_layers).to(device)
optimizer = torch.optim.Adam(network.parameters(), lr=1e-3, weight_decay=1e-4)

# Train LSTM
epoches = 200
rd.train_rudder(network, optimizer, epoches=epoches, data_loader=env_loader, show_gap=10, device=device)
