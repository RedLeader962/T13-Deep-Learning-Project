import gym
import matplotlib.pyplot as plt
import PPO

device = "cpu" # keep gpu ! Quicker for PPO !

environment = gym.make("CartPole-v1") # CartPole-v1, MountainCar-v0, LunarLander-v2

steps_by_epoch = 1000
n_epoches = 40
hidden_dim = 6
n_hidden_layers = 2

agent, info_logger = PPO.PPO(environment,
                             steps_by_epoch=steps_by_epoch,
                             n_epoches=n_epoches,
                             n_hidden_layers=n_hidden_layers,
                             hidden_dim=hidden_dim,
                             lr=0.01,
                             save_gap=1,
                             device=device)

dir_name = environment.unwrapped.spec.id
dim_NN = environment.observation_space.shape[0], hidden_dim, environment.action_space.n

data = info_logger.load_data(dir_name, dim_NN)

plt.title(f"PPO - Number of epoches : {n_epoches} and steps by epoch : {steps_by_epoch}")
plt.plot(data['Rewards'], label='Rewards')
plt.legend()
plt.xlabel("Epoches")
plt.show()

PPO.run_NN(environment, agent, device)

