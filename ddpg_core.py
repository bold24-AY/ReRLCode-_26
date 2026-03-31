import torch
import numpy as np
import os
from models import PolicyNet, QValueNet
from replay_buffer import ExperienceReplay, OUNoise

class LunarLanderDDPG:
    """Refactored core logic for DDPG applied to continuous control tasks."""
    def __init__(self, state_dim, action_dim, min_action, max_action, chkpt_dir="weights", env_name="LunarLanderContinuous-v3", gamma=0.99, tau=0.001, batch_size=64, buffer_size=1000000):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.chkpt_dir = chkpt_dir
        
        if not os.path.exists(self.chkpt_dir):
            os.makedirs(self.chkpt_dir)

        # Replay memory
        self.memory = ExperienceReplay(state_dim, action_dim, capacity=buffer_size)
        
        # Action Exploration Noise
        self.noise = OUNoise(np.zeros(action_dim))

        # Main Networks
        self.policy = PolicyNet(state_dim, action_dim, max_action, min_action, chkpt_dir=self.chkpt_dir, env_name=env_name)
        self.q_value = QValueNet(state_dim, action_dim, chkpt_dir=self.chkpt_dir, env_name=env_name)

        # Target Networks
        self.target_policy = PolicyNet(state_dim, action_dim, max_action, min_action, chkpt_dir=self.chkpt_dir, env_name=env_name + "_target")
        self.target_q_value = QValueNet(state_dim, action_dim, chkpt_dir=self.chkpt_dir, env_name=env_name + "_target")

        self.soft_update_targets(tau=1)

    def select_action(self, state, add_noise=True):
        self.policy.eval()
        
        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.policy.device)
        action_mean = self.policy(state_tensor).to(self.policy.device)

        if add_noise:
            exploration_noise = torch.tensor(self.noise(), dtype=torch.float32).to(self.policy.device)
            action_mean += exploration_noise
            
        self.policy.train()
        
        # Ensure it stays within bounds, clamp it to action space if necessary
        # The environment boundary scaling handles most of it
        return action_mean.cpu().detach().numpy()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append(state, action, reward, next_state, done)

    def learn(self):
        if self.memory.mem_ptr < self.batch_size:
            return None, None

        states, actions, rewards, next_states, dones = self.memory.sample_batch(self.batch_size)
        
        states = torch.tensor(states, dtype=torch.float32).to(self.policy.device)
        actions = torch.tensor(actions, dtype=torch.float32).to(self.policy.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.policy.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.policy.device)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.policy.device)
        
        # Optimize QValueNet (Critic)
        target_actions = self.target_policy(next_states)
        target_q = self.target_q_value(next_states, target_actions)
        target_q[dones] = 0.0 # Terminal states have 0 expected future return
        target_q = target_q.view(-1)
        
        expected_q = rewards + self.gamma * target_q
        expected_q = expected_q.view(self.batch_size, 1)
        
        q_predicted = self.q_value(states, actions)
        
        self.q_value.optimizer.zero_grad()
        q_loss = torch.nn.functional.mse_loss(expected_q, q_predicted)
        q_loss.backward()
        self.q_value.optimizer.step()

        # Optimize PolicyNet (Actor)
        self.policy.optimizer.zero_grad()
        p_loss = -self.q_value(states, self.policy(states))
        p_loss = torch.mean(p_loss)
        p_loss.backward()
        self.policy.optimizer.step()

        self.soft_update_targets()

        return p_loss.item(), q_loss.item()

    def soft_update_targets(self, tau=None):
        if tau is None:
            tau = self.tau

        def update_params(net, target_net):
            for target_param, param in zip(target_net.parameters(), net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        update_params(self.policy, self.target_policy)
        update_params(self.q_value, self.target_q_value)

    def save_weights(self):
        self.policy.save_model()
        self.target_policy.save_model()
        self.q_value.save_model()
        self.target_q_value.save_model()

    def load_weights(self):
        self.policy.load_model()
        self.target_policy.load_model()
        self.q_value.load_model()
        self.target_q_value.load_model()
