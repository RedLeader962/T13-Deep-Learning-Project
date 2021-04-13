import torch
import numpy as np


class NnModel(torch.nn.Module):
    """Return predicted action"""

    def __init__(self, in_dim, out_dim, n_hidden_layers=1, hidden_dim=16, lr=3e-4):
        """
        in_dim is the number of element in input
        out_dim is the number of element in output
        Builds a PyTorch Neural Network with n_hidden_layers, all of hidden_dim neurons.

        The activation function is always ReLU for intermediate layers and the final
        layer does not have any activation function.

        By default, this NN only has one hidden layer of 16 neurons.
        """
        super().__init__()

        # Relu is an activation function
        layers = [torch.nn.Linear(in_dim, hidden_dim), torch.nn.ReLU()]

        #Add hidden layers ReLu in between
        for _ in range(n_hidden_layers):
            layers.extend([torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU()])

        layers.append(torch.nn.Linear(hidden_dim, out_dim))

        # Sequential is a container to hold the layers on the order they were created
        print(layers)
        print(*layers)

        # * is for unpacking the list in the sequential function like below
        self.fa = torch.nn.Sequential(*layers)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        """
        x represent a state
        This is the function that is called when we want to get the output f(x)
        for our NN f.
        """
        return self.fa(x)


x = NnModel(2, 3, 1)


class NnActorCritic(torch.nn.Module):
    """Return predicted action"""

    def __init__(self, in_dim, out_dim, n_hidden_layers=1, hidden_dim=16):
        super().__init__()

        # Create network layers
        layers = [torch.nn.Linear(in_dim, hidden_dim), torch.nn.Tanh()]

        for _ in range(n_hidden_layers):
            layers.extend([torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.Tanh()])

        actor_layer = layers.copy()
        actor_layer.append(torch.nn.Softmax(dim=1))

        layers.append(torch.nn.Linear(hidden_dim, out_dim))

        self.actor = torch.nn.Sequential(*actor_layer)
        self.critic = torch.nn.Sequential(*layers)

    def forward(self, state):
        """ Return predicted Q_value, policy, action """

        policy = min(self.actor(state), 1 - 1e-16)
        q_val = self.critic(state)
        opt_act = torch.argmax(torch.nn.functional.softmax(policy))
        value = torch.sum(q_val * policy, dim=0, keepdim=True)

        return policy, q_val, opt_act, value


x = NnActorCritic(3, 3)
print(x)
