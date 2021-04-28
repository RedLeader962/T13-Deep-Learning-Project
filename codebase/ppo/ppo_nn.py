from .utils import file_end_epoch, file_name
import torch


def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_normal_(m.weight, 1.0)
        torch.nn.init.constant_(m.bias, 0.0)


class NnActor(torch.nn.Module):

    def __init__(self, in_dim, out_dim, n_hidden_layers=1, hidden_dim=16, lr=0.001):
        super().__init__()

        # Create network layers
        layers = [torch.nn.Linear(in_dim, hidden_dim), torch.nn.Tanh()]

        for _ in range(n_hidden_layers - 1):
            layers.extend([torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.Tanh()])

        layers.extend([torch.nn.Linear(hidden_dim, out_dim)])  #torch.nn.Softmax(dim=1))

        self.actor = torch.nn.Sequential(*layers)
        self.optim = torch.optim.Adam(self.parameters(), lr=lr)

        self.apply(init_weights)

    def forward(self, state, action=None):
        """ Return distribution pi and log prob of an action """
        logits = self.actor(state)
        pi = torch.distributions.Categorical(logits=logits)

        log_prob_a = None

        # Log probability of a given action
        if action is not None:
            log_prob_a = pi.log_prob(action)

        return pi, log_prob_a

    def distribution(self, state):
        """ Return the log probability under a categorical distribution """
        logits = self.actor(state)
        pi = torch.distributions.Categorical(logits=logits)
        return pi


class NnCritic(torch.nn.Module):

    def __init__(self, in_dim, n_hidden_layers=1, hidden_dim=16, lr=0.001):
        super().__init__()

        # Create network layers
        layers = [torch.nn.Linear(in_dim, hidden_dim), torch.nn.Tanh()]

        for _ in range(n_hidden_layers):
            layers.extend([torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.Tanh()])

        layers.append(torch.nn.Linear(hidden_dim, 1))

        self.critic = torch.nn.Sequential(*layers)
        self.optim = torch.optim.Adam(self.parameters(), lr=lr)

        self.apply(init_weights)

    def forward(self, state):
        """ Return predicted V_value """
        return torch.squeeze(self.critic(state), -1)


class NnActorCritic(torch.nn.Module):

    def __init__(self, state_dim, action_dim, n_hidden_layers=1, hidden_dim=16, lr=0.001, target_kl=0.015,
                 max_train_pi_iters=80, device='cpu'):
        super().__init__()

        self.pi = NnActor(state_dim, action_dim, n_hidden_layers=n_hidden_layers, hidden_dim=hidden_dim, lr=lr
                          ).to(device)
        self.v = NnCritic(state_dim, n_hidden_layers=n_hidden_layers, hidden_dim=hidden_dim, lr=lr).to(device)

        self.target_kl = target_kl
        self.max_train_pi_iters = max_train_pi_iters

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hid_dim = hidden_dim

        self.load = False
        self.device = device

    def step(self, state):
        with torch.no_grad():
            # Policy PI for discret action space from actor
            pi = self.pi.distribution(state)

            # Action sampled from policy pi
            action = pi.sample()

            # Log prob evaluated for a given action
            log_prob_a = pi.log_prob(action)

            # Compute V value
            v = torch.tensor([self.v(state)], device=self.device)

        return action.cpu().numpy(), v, log_prob_a

    """ Minimize state value loss"""

    def update_critic(self, data):

        # Train policy with multiple steps of gradient descent
        for i in range(self.max_train_pi_iters):

            # Clear gradient for next train
            self.v.optim.zero_grad()

            # Compute loss
            loss_v = self.compute_loss_v(data)

            # Backpropagate the loss into the network
            loss_v.backward()

            # Apply gradient thru the network
            self.v.optim.step()

        return loss_v

    """ Maximize policy performance"""

    def update_actor(self, data, target_kl):

        # Train policy with multiple steps of gradient descent
        for i in range(self.max_train_pi_iters):

            # Clear gradient for next train
            self.pi.optim.zero_grad()

            # Compute loss and approximation of KL
            loss, approx_kl = self.compute_performance_pi(data)

            # If the policies diverge too much on average, then break
            if approx_kl > target_kl:
                #print("  Policy diverge too much : break")
                break

            # Backpropagate the loss into the network
            loss.backward()

            # Apply gradient thru the network
            self.pi.optim.step()

        return loss, approx_kl

    """ Return policy loss and approximative KL"""

    def compute_performance_pi(self, data):
        s, a, discouted_cumsum, adv, logp_old = data
        clip_ratio = 0.2

        pi, logp_a = self.pi(s, a)

        # Compute exp(logp_new(a|s) - logp_old(a|s)) = exp(log(p_new(a|s)/p_old(a|s))) = p_new(a|s)/p_old(a|s)
        # Importance sampling ratio
        ratio = torch.exp(logp_a - logp_old)

        # Clip the ratio
        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv

        # Maximize the performance by modifing gradient by as little as possible
        performance_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Compute KL divergence mean between policies
        approx_kl = (logp_old - logp_a).mean()

        return performance_pi, approx_kl

    """ Return state value loss """

    def compute_loss_v(self, data):
        states, a, discouted_cumsum, adv, logp_a = data

        predicted_vals = self.v(states)
        return ((predicted_vals - discouted_cumsum) ** 2).mean()

    def save_model_data(self, info_logger):

        current_epoch = info_logger.current_epoch - 1
        n_epoches = info_logger.n_epoches - 1

        # If not save time or last epoch
        if not (current_epoch % info_logger.save_gap == 0 or current_epoch == n_epoches):
            return

        # Compute epoch of file
        end_epoch = file_end_epoch(info_logger, self)
        self.network_file_name = self.network_file_name.replace(f"epochRun_{self.start_epoch}_",
                                                                f"epochRun_{end_epoch}_")

        # Save model
        torch.save(self.state_dict(), self.network_file_name)

        # Save data
        info_logger.save_data(self.dir_name, self.dim_NN, self.start_epoch, end_epoch)

        self.start_epoch = end_epoch

    def load_model(self, env, network_file_name=None):
        self.dir_name = env.unwrapped.spec.id
        self.dim_NN = self.state_dim, self.hid_dim, self.action_dim

        try:
            self.network_file_name, self.start_epoch = file_name(self.dir_name, self.dim_NN, network_file_name)

            self.load_state_dict(torch.load(self.network_file_name))

            print(f"Loading most recent network : epoch {self.start_epoch}")
            self.load = True

        except FileNotFoundError:
            print("No file exists under this name. Training new network.")
