"""
Reinforcement Learning Trading Strategies
Includes DQN, PPO, A3C, SAC, and other RL algorithms
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path
from collections import deque
import random
import warnings
warnings.filterwarnings('ignore')

# Import RL libraries
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

try:
    from stable_baselines3 import PPO, A2C, SAC, TD3, DQN
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
    from stable_baselines3.common.monitor import Monitor
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False


# ========== Custom DQN Implementation ==========

class DQNNetwork(nn.Module):
    """Deep Q-Network architecture"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 256]):
        super().__init__()

        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, action_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, state):
        return self.network(state)


class ReplayBuffer:
    """Experience replay buffer for DQN"""

    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """Deep Q-Network agent"""

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 buffer_size: int = 100000,
                 batch_size: int = 64,
                 target_update_freq: int = 1000,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = torch.device(device)

        # Networks
        self.q_network = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_network = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Training stats
        self.steps = 0
        self.episode_rewards = []

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()

    def train_step(self) -> Dict[str, float]:
        """Perform one training step"""
        if len(self.replay_buffer) < self.batch_size:
            return {}

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Compute current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q values
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # Compute loss
        loss = F.mse_loss(current_q, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return {'loss': loss.item(), 'epsilon': self.epsilon}

    def save(self, path: str):
        """Save agent"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, path)

    def load(self, path: str):
        """Load agent"""
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']


# ========== Actor-Critic Network ==========

class ActorNetwork(nn.Module):
    """Actor network for policy-based methods"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 256]):
        super().__init__()

        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)
        self.action_head = nn.Linear(prev_dim, action_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        x = self.backbone(state)
        action_probs = self.softmax(self.action_head(x))
        return action_probs


class CriticNetwork(nn.Module):
    """Critic network for value estimation"""

    def __init__(self, state_dim: int, hidden_dims: List[int] = [256, 256]):
        super().__init__()

        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, state):
        return self.network(state)


class A3CAgent:
    """A3C (Asynchronous Advantage Actor-Critic) agent"""

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 entropy_coef: float = 0.01,
                 value_coef: float = 0.5,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.device = torch.device(device)

        # Networks
        self.actor = ActorNetwork(state_dim, action_dim).to(self.device)
        self.critic = CriticNetwork(state_dim).to(self.device)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

        # Training stats
        self.episode_rewards = []

    def select_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        """Select action from policy"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_probs = self.actor(state_tensor)
            value = self.critic(state_tensor)

            # Sample action
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()

            return action.item(), action_probs[0, action].item(), value.item()

    def train_step(self,
                   states: List[np.ndarray],
                   actions: List[int],
                   rewards: List[float],
                   next_states: List[np.ndarray],
                   dones: List[bool]) -> Dict[str, float]:
        """Train on a trajectory"""
        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)

        # Compute returns
        returns = []
        G = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            G = reward + self.gamma * G * (1 - done)
            returns.insert(0, G)
        returns_tensor = torch.FloatTensor(returns).to(self.device)

        # Compute values and advantages
        values = self.critic(states_tensor).squeeze()
        advantages = returns_tensor - values.detach()

        # Actor loss
        action_probs = self.actor(states_tensor)
        log_probs = torch.log(action_probs.gather(1, actions_tensor.unsqueeze(1))).squeeze()
        entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=1).mean()

        actor_loss = -(log_probs * advantages).mean() - self.entropy_coef * entropy

        # Critic loss
        critic_loss = F.mse_loss(values, returns_tensor)

        # Update networks
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': entropy.item()
        }

    def save(self, path: str):
        """Save agent"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict()
        }, path)

    def load(self, path: str):
        """Load agent"""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])


# ========== Stable Baselines3 Wrappers ==========

class TradingCallback(BaseCallback):
    """Custom callback for tracking training progress"""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        if len(self.model.ep_info_buffer) > 0:
            for info in self.model.ep_info_buffer:
                self.episode_rewards.append(info['r'])
                self.episode_lengths.append(info['l'])
        return True


class StableBaselinesRLStrategy:
    """
    Wrapper for Stable Baselines3 RL algorithms
    Supports PPO, A2C, SAC, TD3, DQN
    """

    def __init__(self,
                 algorithm: str = 'PPO',
                 env_maker: Callable = None,
                 policy: str = 'MlpPolicy',
                 learning_rate: float = 0.0003,
                 n_steps: int = 2048,
                 batch_size: int = 64,
                 n_epochs: int = 10,
                 gamma: float = 0.99,
                 device: str = 'auto',
                 **kwargs):

        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 is required for this strategy")

        self.algorithm = algorithm
        self.env_maker = env_maker
        self.policy = policy
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.device = device
        self.kwargs = kwargs

        self.model = None
        self.env = None

    def build_model(self, env):
        """Build RL model"""
        self.env = env

        # Select algorithm
        algo_class = {
            'PPO': PPO,
            'A2C': A2C,
            'SAC': SAC,
            'TD3': TD3,
            'DQN': DQN
        }.get(self.algorithm)

        if algo_class is None:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        # Build model
        model_kwargs = {
            'policy': self.policy,
            'env': env,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'device': self.device,
            'verbose': 1
        }

        # Add algorithm-specific parameters
        if self.algorithm == 'PPO':
            model_kwargs.update({
                'n_steps': self.n_steps,
                'batch_size': self.batch_size,
                'n_epochs': self.n_epochs
            })
        elif self.algorithm == 'DQN':
            model_kwargs.update({
                'batch_size': self.batch_size,
                'learning_starts': 1000,
                'target_update_interval': 1000
            })

        model_kwargs.update(self.kwargs)

        self.model = algo_class(**model_kwargs)

    def train(self, total_timesteps: int = 100000, callback=None):
        """Train the RL agent"""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        if callback is None:
            callback = TradingCallback()

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback
        )

        return callback

    def predict(self, obs, deterministic: bool = True):
        """Make prediction"""
        if self.model is None:
            raise ValueError("Model not trained")

        action, _states = self.model.predict(obs, deterministic=deterministic)
        return action

    def save(self, path: str):
        """Save model"""
        if self.model is not None:
            self.model.save(path)

    def load(self, path: str, env=None):
        """Load model"""
        if env is not None:
            self.env = env

        algo_class = {
            'PPO': PPO,
            'A2C': A2C,
            'SAC': SAC,
            'TD3': TD3,
            'DQN': DQN
        }.get(self.algorithm)

        self.model = algo_class.load(path, env=self.env)

    def evaluate(self, env, n_eval_episodes: int = 10):
        """Evaluate the agent"""
        from stable_baselines3.common.evaluation import evaluate_policy

        mean_reward, std_reward = evaluate_policy(
            self.model,
            env,
            n_eval_episodes=n_eval_episodes,
            deterministic=True
        )

        return {
            'mean_reward': mean_reward,
            'std_reward': std_reward
        }


# ========== RL Strategy for Trading Environment ==========

class RLTradingStrategy:
    """
    Complete RL trading strategy with training and inference
    """

    def __init__(self,
                 agent_type: str = 'DQN',  # 'DQN', 'A3C', 'PPO', 'A2C', 'SAC'
                 state_dim: Optional[int] = None,
                 action_dim: Optional[int] = None,
                 use_stable_baselines: bool = True,
                 **kwargs):

        self.agent_type = agent_type
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_stable_baselines = use_stable_baselines
        self.kwargs = kwargs

        if use_stable_baselines and SB3_AVAILABLE:
            self.agent = StableBaselinesRLStrategy(algorithm=agent_type, **kwargs)
        elif agent_type == 'DQN' and not use_stable_baselines:
            if state_dim is None or action_dim is None:
                raise ValueError("state_dim and action_dim required for custom DQN")
            self.agent = DQNAgent(state_dim, action_dim, **kwargs)
        elif agent_type == 'A3C' and not use_stable_baselines:
            if state_dim is None or action_dim is None:
                raise ValueError("state_dim and action_dim required for custom A3C")
            self.agent = A3CAgent(state_dim, action_dim, **kwargs)
        else:
            raise ValueError(f"Unsupported agent type: {agent_type}")

    def train_on_environment(self, env, n_episodes: int = 1000, max_steps: int = 1000):
        """Train agent on trading environment"""
        if self.use_stable_baselines:
            # Use SB3 training
            self.agent.build_model(env)
            callback = self.agent.train(total_timesteps=n_episodes * max_steps)
            return callback.episode_rewards

        # Custom training loop
        episode_rewards = []

        for episode in range(n_episodes):
            obs, info = env.reset()
            episode_reward = 0
            episode_data = []

            for step in range(max_steps):
                # Select action
                if isinstance(self.agent, DQNAgent):
                    action = self.agent.select_action(obs.flatten())
                else:  # A3C
                    action, _, _ = self.agent.select_action(obs.flatten())

                # Take step
                next_obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward

                # Store experience
                if isinstance(self.agent, DQNAgent):
                    self.agent.replay_buffer.push(
                        obs.flatten(), action, reward, next_obs.flatten(), done or truncated
                    )
                    # Train
                    self.agent.train_step()
                else:  # A3C
                    episode_data.append((obs.flatten(), action, reward, next_obs.flatten(), done or truncated))

                obs = next_obs

                if done or truncated:
                    break

            # Train A3C on episode
            if isinstance(self.agent, A3CAgent) and episode_data:
                states, actions, rewards, next_states, dones = zip(*episode_data)
                self.agent.train_step(list(states), list(actions), list(rewards), list(next_states), list(dones))

            episode_rewards.append(episode_reward)

            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                print(f"Episode {episode + 1}/{n_episodes}, Avg Reward: {avg_reward:.2f}")

        return episode_rewards

    def predict(self, obs, deterministic: bool = True):
        """Make prediction"""
        if self.use_stable_baselines:
            return self.agent.predict(obs, deterministic)

        if isinstance(self.agent, DQNAgent):
            return self.agent.select_action(obs.flatten(), training=False)
        else:  # A3C
            action, _, _ = self.agent.select_action(obs.flatten())
            return action

    def save(self, path: str):
        """Save strategy"""
        self.agent.save(path)

    def load(self, path: str, env=None):
        """Load strategy"""
        if self.use_stable_baselines:
            self.agent.load(path, env)
        else:
            self.agent.load(path)
