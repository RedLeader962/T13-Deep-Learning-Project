import matplotlib.pyplot as plt
import numpy as np


def plot_reward(predicted, expected, epoch):
    predicted = np.round(predicted[:, -1, -1].detach().cpu().numpy(), decimals=1)
    expected = np.round(expected.detach().cpu().numpy(), decimals=0)

    plt.plot(predicted, label='Predicted', c='blue')
    plt.plot(expected, label='Expected', c='red')
    plt.title(f'Epoch {int(epoch)} : Predicted VS Expected Rewards')
    plt.xlabel('Trajectory')
    plt.ylabel('Rewards')
    plt.legend()
    plt.show()
