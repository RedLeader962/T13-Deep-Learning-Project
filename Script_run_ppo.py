import PPO as ppo
import gym
import matplotlib.pyplot as plt
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

environment = gym.make("CartPole-v1") # CartPole-v1, MountainCar-v0, LunarLander-v2

steps_by_epoch = 500
n_epoches = 20
hidden_dim = 18
n_hidden_layers = 2

agent, info_logger = ppo.PPO(environment,
                     steps_by_epoch=steps_by_epoch,
                     n_epoches=n_epoches,
                     n_hidden_layers=n_hidden_layers,
                     hidden_dim=hidden_dim,
                     lr=0.001,
                     device=device)

dir_name = environment.unwrapped.spec.id
dim_NN = environment.observation_space.shape[0], hidden_dim, environment.action_space.n

data = info_logger.load_data(dir_name, dim_NN)

plt.title(f"PPO - Number of epoches : {n_epoches} and steps by epoch : {steps_by_epoch}")
plt.plot(data['Rewards'], label='Rewards')
plt.legend()
plt.xlabel("Epoches")
plt.show()

ppo.run_NN(environment, agent, device)

