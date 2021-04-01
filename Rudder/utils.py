import matplotlib.pyplot as plt

def plot_reward(predicted, expected, epoch):

    plt.plot(predicted, label='Predicted', c='blue')
    plt.plot(expected, label ='Expected', c='red')
    plt.title(f'Epoch {int(epoch)} : Predicted VS Expected Rewards')
    plt.xlabel('Trajectory')
    plt.ylabel('Rewards')
    plt.legend()
    plt.show()