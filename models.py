import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, min_action, hidden1=400, hidden2=300, learning_rate=1e-4, chkpt_dir="weights", env_name="LunarLanderContinuous-v3"):
        super(PolicyNet, self).__init__()
        self.chkpt_file = f"{chkpt_dir}/{env_name}_policy.pth"
        
        self.fc1 = nn.Linear(state_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        
        self.ln1 = nn.LayerNorm(hidden1)
        self.ln2 = nn.LayerNorm(hidden2)
        
        self.mu = nn.Linear(hidden2, action_dim)
        
        self._initialize_weights()
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Action space limits
        self.max_action = torch.tensor(max_action, device=self.device, dtype=torch.float32)
        self.min_action = torch.tensor(min_action, device=self.device, dtype=torch.float32)
        self.action_range = self.max_action - self.min_action
        
        self.to(self.device)

    def _initialize_weights(self):
        f1 = 1.0 / np.sqrt(self.fc1.weight.data.size()[0])
        nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        nn.init.uniform_(self.fc1.bias.data, -f1, f1)

        f2 = 1.0 / np.sqrt(self.fc2.weight.data.size()[0])
        nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        nn.init.uniform_(self.fc2.bias.data, -f2, f2)

        fout = 3e-3
        nn.init.uniform_(self.mu.weight.data, -fout, fout)
        nn.init.uniform_(self.mu.bias.data, -fout, fout)

    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(self.ln1(x))
        x = self.fc2(x)
        x = F.relu(self.ln2(x))
        x = torch.tanh(self.mu(x))
        # Scale to match environment boundaries
        return self.min_action + (x + 1.0) * 0.5 * self.action_range

    def save_model(self):
        torch.save(self.state_dict(), self.chkpt_file)

    def load_model(self):
        self.load_state_dict(torch.load(self.chkpt_file))


class QValueNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden1=400, hidden2=300, learning_rate=1e-3, weight_decay=1e-2, chkpt_dir="weights", env_name="LunarLanderContinuous-v3"):
        super(QValueNet, self).__init__()
        self.chkpt_file = f"{chkpt_dir}/{env_name}_q_value.pth"

        self.fc1 = nn.Linear(state_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        
        self.ln1 = nn.LayerNorm(hidden1)
        self.ln2 = nn.LayerNorm(hidden2)
        
        self.action_value = nn.Linear(action_dim, hidden2)
        self.q = nn.Linear(hidden2, 1)

        self._initialize_weights()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def _initialize_weights(self):
        f1 = 1.0 / np.sqrt(self.fc1.weight.data.size()[0])
        nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        nn.init.uniform_(self.fc1.bias.data, -f1, f1)

        f2 = 1.0 / np.sqrt(self.fc2.weight.data.size()[0])
        nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        nn.init.uniform_(self.fc2.bias.data, -f2, f2)

        fact = 1.0 / np.sqrt(self.action_value.weight.data.size()[0])
        nn.init.uniform_(self.action_value.weight.data, -fact, fact)
        nn.init.uniform_(self.action_value.bias.data, -fact, fact)

        fout = 3e-3
        nn.init.uniform_(self.q.weight.data, -fout, fout)
        nn.init.uniform_(self.q.bias.data, -fout, fout)

    def forward(self, state, action):
        state_x = self.fc1(state)
        state_x = F.relu(self.ln1(state_x))
        state_x = self.fc2(state_x)
        state_x = self.ln2(state_x)

        action_x = self.action_value(action)

        # Merge state and action pathways
        q_val = F.relu(torch.add(state_x, action_x))
        return self.q(q_val)
        
    def save_model(self):
        torch.save(self.state_dict(), self.chkpt_file)

    def load_model(self):
        self.load_state_dict(torch.load(self.chkpt_file))
