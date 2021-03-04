from .utils import *
import torch

class PPO_Buffer:
    """
    Replay buffer object that stores elements up until a certain maximum size.
    """

    def __init__(self, buffer_size, obs_dim, device):
        """
        Init the buffer and store buffer_size property.
        """
        self.buffer_size = buffer_size
        self.s      = torch.zeros((buffer_size, obs_dim), dtype=torch.float, device=device)
        self.a      = torch.zeros(buffer_size, dtype=torch.float, device=device)
        self.r      = torch.zeros(buffer_size, dtype=torch.float,device=device)
        self.s_next = torch.zeros((buffer_size, obs_dim), dtype=torch.float, device=device)
        self.done   = torch.zeros(buffer_size, dtype=torch.bool, device=device)
        self.v_vals = torch.zeros(buffer_size, dtype=torch.float, device=device)
        self.logp_a = torch.zeros(buffer_size, dtype=torch.float, device=device)

        self.adv = torch.zeros(buffer_size, dtype=torch.float,device=device)
        self.ret = torch.zeros(buffer_size, dtype=torch.float,device=device)

        self.buffer_current_size = 0
        self.start_trajectory    = 0

        self.device = device

    def store(self, s, a, r, s_next, done, v_val, logp_a):
        """
        Stores an element into the circular table
        If the buffer is already full, pop the oldest element inside.
        """
        current_position = self.buffer_current_size % self.buffer_size

        self.s[current_position]      = s
        self.a[current_position]      = torch.tensor(a, device=self.device)
        self.r[current_position]      = r
        self.s_next[current_position] = s_next
        self.done[current_position]   = done
        self.v_vals[current_position] = v_val
        self.logp_a[current_position] = logp_a

        self.buffer_current_size += 1


    def get_trajectories(self):
        """
        Return the data needed for update
        """
        self.buffer_current_size, self.start_trajectory = 0, 0

        # Normalize advantage trick
        adv_mean, adv_std = torch.mean(self.adv), torch.std(self.adv)
        self.adv = (self.adv - adv_mean) / adv_std

        s = torch.as_tensor(self.s, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(self.a, dtype=torch.float32, device=self.device)
        ret = torch.as_tensor(self.ret, dtype=torch.float32, device=self.device)
        adv = torch.as_tensor(self.adv, dtype=torch.float32, device=self.device)
        logp_a = torch.as_tensor(self.logp_a, dtype=torch.float32, device=self.device)

        return s, a, ret, adv, logp_a

    def epoch_ended(self, last_v, gamma, lam):

        trajectory_slice = slice(self.start_trajectory, self.buffer_current_size)

        # Last reward
        rewards = torch.cat((self.r[trajectory_slice], last_v))
        v_vals  = torch.cat((self.v_vals[trajectory_slice], last_v))

        # Compute TD error
        delta_error = rewards[:-1] + gamma * v_vals[1:] - v_vals[:-1]
        
        # Generalized advantage estimation (GAE)-Lambda
        self.adv[trajectory_slice] = cumul_discounted_rewards(delta_error, gamma * lam, self.device)

        # Targets for the value function
        self.ret[trajectory_slice] = cumul_discounted_rewards(rewards, gamma, self.device)[:-1]

        self.start_trajectory = self.buffer_current_size

# Test

"""b = PPO_Buffer(5,4)

s,a, r, s_next, done = np.array([1,2,3,4]), 0, 1, np.array([2,3,4,5]), True

print(r)

b.store(s,a,r,s_next,done)
b.store(s,a,r,s_next,done)
b.store(s,1,r,s_next,done)
b.store(s,a,r,s_next,done)
b.store(s,a,r,s_next,done)
b.store(s,5,r,s_next,done)

print(b.s)"""





