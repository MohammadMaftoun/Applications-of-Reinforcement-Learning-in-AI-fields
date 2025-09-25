import gym
from gym.wrappers import PixelObservationWrapper, ResizeObservation, GrayScaleObservation
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from collections import deque
import random

# Define MLP Q-Network
class MLPQNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPQNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

# Define CNN Q-Network (similar to Atari DQN)
class CNNQNetwork(nn.Module):
    def __init__(self, input_channels, output_dim):
        super(CNNQNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Replay buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.stack(state), np.array(action), np.array(reward), np.stack(next_state), np.array(done)
    
    def __len__(self):
        return len(self.buffer)

# DQN Agent
class DQNAgent:
    def __init__(self, env, model_type, hidden_dim, input_dim, input_channels, learning_rate, discount_factor, buffer_capacity):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.is_pixel = (model_type == "cnn")
        if model_type == "mlp":
            self.model = MLPQNetwork(input_dim, hidden_dim, env.action_space.n).to(self.device)
            self.target_model = MLPQNetwork(input_dim, hidden_dim, env.action_space.n).to(self.device)
        else:
            self.model = CNNQNetwork(input_channels, env.action_space.n).to(self.device)
            self.target_model = CNNQNetwork(input_channels, env.action_space.n).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.discount_factor = discount_factor
        self.buffer = ReplayBuffer(buffer_capacity)
    
    def act(self, state, epsilon):
        if random.random() > epsilon:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            if self.is_pixel:
                state_t = state_t.permute(0, 2, 0, 1) / 255.0  # (batch, channels, height, width)
            q_values = self.model(state_t)
            action = q_values.max(1)[1].item()
        else:
            action = self.env.action_space.sample()
        return action
    
    def update(self, batch_size):
        if len(self.buffer) < batch_size:
            return 0
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        
        states_t = torch.FloatTensor(states).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        if self.is_pixel:
            states_t = states_t.permute(0, 3, 1, 2) / 255.0  # Note: assuming shape (batch, height, width, channels)
            next_states_t = next_states_t.permute(0, 3, 1, 2) / 255.0
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        q_values = self.model(states_t).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_model(next_states_t).max(1)[0]
        target = rewards + (1 - dones) * self.discount_factor * next_q_values
        loss = nn.MSELoss()(q_values, target.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

# Objective function for Optuna
def objective(trial):
    model_type = trial.suggest_categorical("model_type", ["mlp", "cnn"])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    discount_factor = trial.suggest_float("discount_factor", 0.9, 0.999)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    buffer_capacity = trial.suggest_categorical("buffer_capacity", [1000, 10000, 100000])
    epsilon_start = trial.suggest_float("epsilon_start", 0.5, 1.0)
    epsilon_decay = trial.suggest_float("epsilon_decay", 0.99, 0.999)
    
    hidden_dim = None
    input_dim = None
    input_channels = None
    
    if model_type == "mlp":
        hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
        env = gym.make("CartPole-v1")
        input_dim = env.observation_space.shape[0]
    else:  # cnn
        env = gym.make("CartPole-v1", render_mode="rgb_array")
        env = PixelObservationWrapper(env)
        env = ResizeObservation(env, 84)
        env = GrayScaleObservation(env, keep_dim=True)  # shape (84, 84, 1)
        input_channels = env.observation_space.shape[2]
    
    agent = DQNAgent(env, model_type, hidden_dim, input_dim, input_channels, learning_rate, discount_factor, buffer_capacity)
    
    epsilon = epsilon_start
    total_rewards = []
    episodes = 50
    
    for episode in range(episodes):
        state = env.reset()
        if agent.is_pixel:
            state = state['pixels']
        episode_reward = 0
        done = False
        while not done:
            action = agent.act(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            if agent.is_pixel:
                next_state = next_state['pixels']
            agent.buffer.push(state, action, reward, next_state, done)
            agent.update(batch_size)
            state = next_state
            episode_reward += reward
            if done:
                total_rewards.append(episode_reward)
                epsilon *= epsilon_decay
                break
    
    env.close()
    return np.mean(total_rewards)

# Run optimization
def optimize_hyperparameters(n_trials=50):
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    
    print("Best hyperparameters: ", study.best_params)
    print("Best average reward: ", study.best_value)
    
    return study

if __name__ == "__main__":
    study = optimize_hyperparameters(n_trials=50)
    
    # Optional visualizations
    import optuna.visualization as vis
    vis.plot_optimization_history(study).show()
    vis.plot_param_importances(study).show()