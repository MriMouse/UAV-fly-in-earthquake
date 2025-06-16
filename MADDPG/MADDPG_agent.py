import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque


# Actor Network (Parameter Sharing)
# Same as DDPG
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, max_action=1.0):
        super(Actor, self).__init__()
        self.max_action = max_action
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.tanh = nn.Tanh()

    def forward(self, state):
        action = self.net(state)
        action = self.tanh(action) * self.max_action  # Scale action
        return action


# Centralized Critic Network (Parameter Sharing)
# Takes all agents' states and actions as input
class CentralizedCritic(nn.Module):
    def __init__(self, num_agents, state_dim, action_dim, hidden_dim=256):
        super(CentralizedCritic, self).__init__()
        centralized_state_dim = num_agents * state_dim
        centralized_action_dim = num_agents * action_dim

        # Define network layers
        self.layer1_state = nn.Linear(centralized_state_dim, hidden_dim)
        self.layer2_action = nn.Linear(centralized_action_dim, hidden_dim)
        self.layer3_combined = nn.Linear(hidden_dim + hidden_dim, hidden_dim)
        self.layer4_output = nn.Linear(hidden_dim, 1)  # Output a single Q-value

    def forward(self, centralized_states, centralized_actions):
        # Args:
        #     centralized_states (Tensor): Shape (batch_size, num_agents * state_dim)
        #     centralized_actions (Tensor): Shape (batch_size, num_agents * action_dim)
        # Returns:
        #     q_value (Tensor): Shape (batch_size, 1)
        state_out = F.relu(self.layer1_state(centralized_states))
        action_out = F.relu(self.layer2_action(centralized_actions))
        combined = torch.cat([state_out, action_out], dim=1)  # Concatenate
        q_out = F.relu(self.layer3_combined(combined))
        q_value = self.layer4_output(q_out)
        return q_value


# Replay Buffer for MADDPG
class MADDPGReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=int(capacity))  # Use deque

    def add(self, states_list, actions_list, reward, next_states_list, done):
        # Store one timestep of multi-agent experience
        # Ensure internal storage uses numpy arrays
        states_np = [np.array(s, dtype=np.float32) for s in states_list]
        actions_np = [np.array(a, dtype=np.float32) for a in actions_list]
        next_states_np = [np.array(ns, dtype=np.float32) for ns in next_states_list]
        reward_np = np.array([reward], dtype=np.float32)  # Shared reward
        done_np = np.array([done], dtype=np.float32)  # Shared done flag, use float32

        experience = (states_np, actions_np, reward_np, next_states_np, done_np)
        self.buffer.append(experience)

    def sample(self, batch_size, num_agents, state_dim, action_dim):
        # Sample a batch and format it for MADDPG
        batch = random.sample(self.buffer, batch_size)
        states_b, actions_b, rewards_b, next_states_b, dones_b = zip(*batch)

        # Organize lists of lists into batch tensors
        # states: (batch_size, num_agents, state_dim)
        states_tensor = torch.tensor(np.array(states_b), dtype=torch.float32)
        # actions: (batch_size, num_agents, action_dim)
        actions_tensor = torch.tensor(np.array(actions_b), dtype=torch.float32)
        # rewards: (batch_size, 1)
        rewards_tensor = torch.tensor(np.array(rewards_b), dtype=torch.float32)
        # next_states: (batch_size, num_agents, state_dim)
        next_states_tensor = torch.tensor(np.array(next_states_b), dtype=torch.float32)
        # dones: (batch_size, 1)
        dones_tensor = torch.tensor(np.array(dones_b), dtype=torch.float32)

        return states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor

    def __len__(self):
        return len(self.buffer)


# MADDPG Agent Class
class MADDPGAgent:
    def __init__(
        self,
        num_agents,
        state_dim,
        action_dim,
        hidden_dim,
        lr_actor,
        lr_critic,
        gamma,
        tau,
        device,
        buffer_capacity,
        # Noise Parameters
        initial_noise_scale=0.1,
        final_noise_scale=0.01,
        noise_decay_steps=500000,
        # Other Parameters
        max_action=1.0,
        min_lr=1e-6,
    ):

        self.device = device
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma  # Discount factor
        self.tau = tau  # Soft update rate
        self.max_action = max_action
        self.min_lr = min_lr  # Minimum learning rate

        # Noise Parameters
        self.initial_noise_scale = initial_noise_scale
        self.final_noise_scale = final_noise_scale
        self.noise_decay_steps = noise_decay_steps
        self.current_noise_scale = initial_noise_scale

        # Initialize Networks (Parameter Sharing)
        # Actor Network and Target Actor Network
        self.actor = Actor(state_dim, action_dim, hidden_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())  # Initialize target

        # Centralized Critic Network and Target Critic Network
        self.critic = CentralizedCritic(num_agents, state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = CentralizedCritic(num_agents, state_dim, action_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())  # Initialize target

        # Optimizers
        self.initial_lr_actor = lr_actor
        self.initial_lr_critic = lr_critic
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.initial_lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.initial_lr_critic)

        # Replay Buffer
        self.replay_buffer = MADDPGReplayBuffer(buffer_capacity)

        # Loss Function
        self.mse_loss = nn.MSELoss()  # MSE for critic loss

    def get_action(self, states_list, total_steps_elapsed, evaluate=False):
        # Generate actions list for all agents (decentralized execution)
        actions_list = []
        self.actor.eval()  # Set actor to evaluation mode

        # Noise Decay
        if not evaluate:  # Only apply noise during training
            decay_frac = min(1.0, total_steps_elapsed / self.noise_decay_steps) if self.noise_decay_steps > 0 else 1.0
            self.current_noise_scale = (
                self.initial_noise_scale + (self.final_noise_scale - self.initial_noise_scale) * decay_frac
            )
            self.current_noise_scale = max(
                self.final_noise_scale, self.current_noise_scale
            )  # Ensure noise doesn't go below final

        with torch.no_grad():  # No gradient calculation
            for agent_id in range(self.num_agents):
                state = states_list[agent_id]
                if not isinstance(state, np.ndarray):  # Ensure state is numpy array
                    state = np.array(state, dtype=np.float32)
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # Convert to tensor

                # Use the shared Actor network
                action = self.actor(state_tensor).cpu().numpy()[0]  # Get action

                if not evaluate:
                    # Add independent Gaussian noise for exploration
                    noise = np.random.normal(0, self.current_noise_scale, size=self.action_dim)
                    action = action + noise
                    action = np.clip(action, -self.max_action, self.max_action)  # Clip action

                actions_list.append(action)

        self.actor.train()  # Restore actor to training mode
        return actions_list

    def store_transition(self, states_list, actions_list, reward, next_states_list, done):
        self.replay_buffer.add(states_list, actions_list, reward, next_states_list, done)

    def update(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            # If buffer has insufficient samples, do not update
            current_lr_actor = self.actor_optimizer.param_groups[0]["lr"]
            current_lr_critic = self.critic_optimizer.param_groups[0]["lr"]
            return current_lr_actor, current_lr_critic, self.current_noise_scale

        # Sample from Replay Buffer
        states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch = self.replay_buffer.sample(
            batch_size, self.num_agents, self.state_dim, self.action_dim
        )

        # Move data to the specified device
        states_batch = states_batch.to(self.device)
        actions_batch = actions_batch.to(self.device)
        rewards_batch = rewards_batch.to(self.device)
        next_states_batch = next_states_batch.to(self.device)
        dones_batch = dones_batch.to(self.device)

        # Critic Update
        with torch.no_grad():  # Target network calculations don't require gradients
            # Calculate next actions for all agents (using target actor)
            next_states_flat = next_states_batch.view(-1, self.state_dim)
            next_actions_flat = self.actor_target(next_states_flat)
            next_actions_batch = next_actions_flat.view(batch_size, self.num_agents, self.action_dim)

            # Prepare centralized next states and next actions
            centralized_next_states = next_states_batch.view(batch_size, -1)
            centralized_next_actions = next_actions_batch.view(batch_size, -1)

            # Use target critic to calculate target Q-values
            target_q_values = self.critic_target(centralized_next_states, centralized_next_actions)

            # Calculate final target Q-value (Bellman equation)
            target_q = rewards_batch + (self.gamma * target_q_values * (1.0 - dones_batch))

        # Prepare centralized current states and current actions
        centralized_states = states_batch.view(batch_size, -1)
        centralized_actions = actions_batch.view(batch_size, -1)

        # Use main critic to calculate current Q-values
        current_q_values = self.critic(centralized_states, centralized_actions)

        # Calculate Critic Loss
        critic_loss = self.mse_loss(current_q_values, target_q)

        # Optimize Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)  # Optional: gradient clipping
        self.critic_optimizer.step()

        # Actor Update
        # Calculate actions for all agents in current states (using main actor)
        states_flat = states_batch.view(-1, self.state_dim)
        current_actions_flat = self.actor(states_flat)
        current_actions_batch = current_actions_flat.view(batch_size, self.num_agents, self.action_dim)

        # Prepare centralized current states and *new* current actions
        centralized_current_actions = current_actions_batch.view(batch_size, -1)

        # Calculate Actor Loss (maximize Q-value from critic for actor's output actions)
        actor_loss = -self.critic(centralized_states, centralized_current_actions).mean()

        # Optimize Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)  # Optional: gradient clipping
        self.actor_optimizer.step()

        # Soft update Target Networks
        self._soft_update(self.actor_target, self.actor, self.tau)
        self._soft_update(self.critic_target, self.critic, self.tau)

        # Return current LR and noise scale for logging
        current_lr_actor = self.actor_optimizer.param_groups[0]["lr"]
        current_lr_critic = self.critic_optimizer.param_groups[0]["lr"]
        return current_lr_actor, current_lr_critic, self.current_noise_scale

    def _soft_update(self, target_net, source_net, tau):
        # Soft update target network parameters
        for target_param, param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def save_model(self, actor_path, critic_path):
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        print(f"Model saved: Actor '{actor_path}', Critic '{critic_path}'")

    def load_model(self, actor_path, critic_path):
        self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
        self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
        # Synchronize Target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        # Set to evaluation mode
        self.actor.eval()
        self.critic.eval()
        self.actor_target.eval()
        self.critic_target.eval()
        print(f"Model loaded: Actor '{actor_path}', Critic '{critic_path}'")
