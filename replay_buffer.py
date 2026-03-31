import numpy as np

class ExperienceReplay:
    def __init__(self, state_dim, action_dim, capacity=1000000):
        self.capacity = capacity
        self.mem_ptr = 0
        
        self.states = np.zeros((self.capacity, state_dim))
        self.next_states = np.zeros((self.capacity, state_dim))
        self.actions = np.zeros((self.capacity, action_dim))
        self.rewards = np.zeros(self.capacity)
        self.dones = np.zeros(self.capacity, dtype=bool)

    def append(self, state, action, reward, next_state, done):
        idx = self.mem_ptr % self.capacity
        self.states[idx] = state
        self.next_states[idx] = next_state
        self.rewards[idx] = reward
        self.actions[idx] = action
        self.dones[idx] = done
        self.mem_ptr += 1

    def sample_batch(self, batch_size):
        max_mem = min(self.mem_ptr, self.capacity)
        indices = np.random.choice(max_mem, batch_size, replace=False)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )

class OUNoise:
    """Ornstein-Uhlenbeck process for exploration noise in continuous action spaces."""
    def __init__(self, mean, std_dev=0.15, theta=0.2, dt=0.01, initial_value=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_dev
        self.dt = dt
        self.initial_value = initial_value
        self.reset()

    def __call__(self):
        # Calculate the noise based on previous state
        noise = (self.prev_val 
                 + self.theta * (self.mean - self.prev_val) * self.dt 
                 + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape))
        self.prev_val = noise
        return noise

    def reset(self):
        self.prev_val = self.initial_value if self.initial_value is not None else np.zeros_like(self.mean)
