import torch
from torch import nn
from torch.distributions import Normal
import numpy as np
import torch.optim as optim


# Actor, Critic Networks
# Actor network remains the same as it processes individual agent states
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, action_scale=1.0):
        super(Actor, self).__init__()
        self.action_scale = action_scale
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.fc_std = nn.Linear(hidden_dim, action_dim)
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, x):
        hidden_out = self.net(x)
        # Constrain mean output between [-action_scale, action_scale]
        mean = self.tanh(self.fc_mean(hidden_out)) * self.action_scale
        # Constrain std to a reasonable range, prevent too small or too large
        std = self.softplus(self.fc_std(hidden_out)) + 1e-5  # Ensure min value 1e-5
        std = torch.clamp(std, min=1e-5, max=1.0)  # Constrain max value to 1.0
        return mean, std

    def get_distribution(self, state):
        mean, std = self.forward(state)
        return Normal(mean, std)

    # This method might not be directly called in main loop for MAPPO, but needed in update
    def get_log_prob(self, state, action):
        dist = self.get_distribution(state)
        # Sum log_prob for each action dimension
        return dist.log_prob(action).sum(dim=-1, keepdim=True)


# Critic network modified to accept centralized state
class CentralizedCritic(nn.Module):
    def __init__(self, num_agents, state_dim, hidden_dim=256):
        super(CentralizedCritic, self).__init__()
        # Input dimension is sum of all agent state dimensions
        centralized_state_dim = num_agents * state_dim
        self.net = nn.Sequential(
            nn.Linear(centralized_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # Output single value estimate
        )

    def forward(self, centralized_states):
        # centralized_states shape should be (batch_size, num_agents * state_dim)
        return self.net(centralized_states)


# Replay Buffer for MAPPO
class MAPPOReplayMemory:
    def __init__(self):
        self.clear()

    def add_memo(self, states_list, actions_list, reward, value, done):
        # Store data for one timestep
        # Args:
        #     states_list (list): List of num_agents state arrays
        #     actions_list (list): List of num_agents action arrays
        #     reward (float): Shared reward for this timestep
        #     value (float): Centralized Critic's value estimate for joint state
        #     done (bool): Shared done flag for this timestep
        if not isinstance(states_list, list) or not isinstance(actions_list, list):
            raise TypeError("states_list and actions_list must be lists.")
        self.states.append(states_list)
        self.actions.append(actions_list)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def get_tensors(self, device, num_agents, state_dim, action_dim):
        # Convert stored data to PyTorch tensors and adjust shape for MAPPO training
        buffer_size = len(self.rewards)

        # States: list of lists of arrays -> (buffer_size, num_agents, state_dim)
        states_np = np.array(self.states, dtype=np.float32).reshape(buffer_size, num_agents, state_dim)
        # Actions: list of lists of arrays -> (buffer_size, num_agents, action_dim)
        actions_np = np.array(self.actions, dtype=np.float32).reshape(buffer_size, num_agents, action_dim)

        # Rewards, values, dones: list -> (buffer_size, 1)
        rewards_np = np.array(self.rewards, dtype=np.float32).reshape(-1, 1)
        values_np = np.array(self.values, dtype=np.float32).reshape(-1, 1)
        dones_np = np.array(self.dones, dtype=np.float32).reshape(-1, 1)  # Use float32

        states_tensor = torch.from_numpy(states_np).to(device)
        actions_tensor = torch.from_numpy(actions_np).to(device)
        rewards_tensor = torch.from_numpy(rewards_np).to(device)
        values_tensor = torch.from_numpy(values_np).to(device)
        dones_tensor = torch.from_numpy(dones_np).to(device)  # 1.0 for done, 0.0 for not done

        # Prepare centralized states for Critic: (buffer_size, num_agents * state_dim)
        centralized_states_tensor = states_tensor.view(buffer_size, -1)

        return states_tensor, actions_tensor, centralized_states_tensor, rewards_tensor, values_tensor, dones_tensor

    def clear(self):
        self.states = []  # List of lists of states for each step
        self.actions = []  # List of lists of actions for each step
        self.rewards = []  # List of shared rewards for each step
        self.values = []  # List of centralized values for each step
        self.dones = []  # List of shared dones for each step

    def size(self):
        return len(self.rewards)  # Use rewards list length as buffer size


# MAPPO Agent Class
class MAPPOAgent:
    def __init__(
        self,
        num_agents,
        state_dim,
        action_dim,
        hidden_dim,
        lr_actor,
        lr_critic,
        gamma,
        gae_lambda,
        ppo_epsilon,
        ppo_epochs,
        minibatch_size,
        device,
        # LR Decay Parameters
        total_training_steps,
        lr_decay=True,
        final_lr_factor=0.0,
        # Entropy Decay Parameters
        initial_entropy_coef=0.01,
        final_entropy_coef=0.0,
        entropy_decay=True,
        # Other Parameters
        action_scale=1.0,
        max_grad_norm=0.5,
    ):

        self.device = device
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.epsilon = ppo_epsilon
        self.epochs = ppo_epochs
        self.minibatch_size = minibatch_size
        self.max_grad_norm = max_grad_norm
        self.action_scale = action_scale  # Ensure Actor uses correct action range

        # Store initial LR and decay parameters
        self.initial_lr_actor = lr_actor
        self.initial_lr_critic = lr_critic
        self.total_training_steps = total_training_steps
        self.lr_decay = lr_decay
        self.final_lr_factor = final_lr_factor
        self.min_lr = 1e-6  # Minimum learning rate

        # Store initial Entropy and decay parameters
        self.initial_entropy_coef = initial_entropy_coef
        self.final_entropy_coef = max(0.0, final_entropy_coef)  # Ensure non-negative
        self.entropy_decay = entropy_decay

        # Networks (Parameter Sharing)
        # Actor: processes individual agent's state and action
        self.actor = Actor(state_dim, action_dim, hidden_dim, self.action_scale).to(self.device)
        # Old Actor for PPO ratio calculation
        self.old_actor = Actor(state_dim, action_dim, hidden_dim, self.action_scale).to(self.device)
        self.old_actor.load_state_dict(self.actor.state_dict())
        self.old_actor.eval()  # Set to evaluation mode

        # Critic: processes centralized state (concatenation of all agents' states)
        self.critic = CentralizedCritic(num_agents, state_dim, hidden_dim).to(self.device)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.initial_lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.initial_lr_critic)

        # Replay Buffer
        self.replay_buffer = MAPPOReplayMemory()

        # Loss Function
        self.mse_loss = nn.MSELoss()

    def get_action(self, states_list, evaluate=False):
        # Get actions based on each agent's local state (decentralized execution).
        # Also use centralized Critic to evaluate current joint state's value.
        # Args:
        #     states_list (list): List of num_agents state arrays
        #     evaluate (bool): Whether in evaluation mode (use mean action if True)
        # Returns:
        #     actions_list (list): List of num_agents action arrays
        #     centralized_value (float): Centralized Critic's value estimate for joint state
        #     log_probs_list (list): List of num_agents action log probabilities
        if not isinstance(states_list, list):
            raise TypeError("Input 'states_list' must be a list of states.")
        if len(states_list) != self.num_agents:
            raise ValueError(f"Expected {self.num_agents} states, got {len(states_list)}")

        actions_list = []
        log_probs_list = []
        states_tensor_list = []  # For building centralized state

        with torch.no_grad():  # No gradient calculation
            for agent_id in range(self.num_agents):
                state = states_list[agent_id]
                if not isinstance(state, np.ndarray):
                    state = np.array(state, dtype=np.float32)
                # Convert to Tensor (add batch dimension)
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                states_tensor_list.append(state_tensor)

                # Actor network gets action distribution
                dist = self.actor.get_distribution(state_tensor)

                # Sample or take mean action
                if evaluate:
                    action = dist.mean
                else:
                    action = dist.sample()

                # Calculate log probability
                log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)

                # Clip action to valid range
                action_clamped = torch.clamp(action, -self.actor.action_scale, self.actor.action_scale)

                actions_list.append(action_clamped.cpu().numpy()[0])  # Remove batch dim and to numpy
                log_probs_list.append(log_prob.cpu().item())  # To scalar

            # Centralized Critic evaluation
            # Concatenate all agent state tensors to form centralized state
            centralized_state_tensor = torch.cat(states_tensor_list, dim=1)  # Shape: (1, num_agents * state_dim)
            centralized_value = self.critic(centralized_state_tensor).cpu().item()  # Get value estimate

        return actions_list, centralized_value, log_probs_list

    def _calculate_gae(self, rewards, values, dones, last_value):
        # Calculate GAE (Generalized Advantage Estimation).
        # Inputs should be (buffer_size, 1) tensors.
        # last_value is the centralized value estimate of the last state (scalar).
        num_steps = len(rewards)
        advantages = torch.zeros_like(rewards).to(self.device)
        returns = torch.zeros_like(rewards).to(self.device)
        last_gae_lam = 0

        # Convert last_value to tensor
        last_value_tensor = torch.tensor([[last_value]], dtype=torch.float32).to(self.device)
        # Concatenate values and last_value_tensor for delta calculation
        full_values = torch.cat((values, last_value_tensor), dim=0)  # Shape: (buffer_size + 1, 1)

        # Calculate GAE from back to front
        for t in reversed(range(num_steps)):
            # dones[t] is done flag (1.0 for done, 0.0 for not done)
            delta = rewards[t] + self.gamma * full_values[t + 1] * (1.0 - dones[t]) - full_values[t]
            advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * last_gae_lam

        # Calculate returns (Returns = Advantages + Values)
        returns = advantages + values
        return advantages, returns

    def update(self, last_centralized_value, current_timestep):
        # Perform MAPPO update using data from Replay Buffer, and apply LR & Entropy Coef decay

        # Calculate decay fraction
        frac = min(1.0, current_timestep / self.total_training_steps)

        # Learning rate decay
        current_lr_actor = self.initial_lr_actor
        current_lr_critic = self.initial_lr_critic
        if self.lr_decay:
            final_lr_actor = self.initial_lr_actor * self.final_lr_factor
            final_lr_critic = self.initial_lr_critic * self.final_lr_factor
            current_lr_actor = self.initial_lr_actor + (final_lr_actor - self.initial_lr_actor) * frac
            current_lr_critic = self.initial_lr_critic + (final_lr_critic - self.initial_lr_critic) * frac
            current_lr_actor = max(current_lr_actor, self.min_lr)
            current_lr_critic = max(current_lr_critic, self.min_lr)
            for param_group in self.actor_optimizer.param_groups:
                param_group["lr"] = current_lr_actor
            for param_group in self.critic_optimizer.param_groups:
                param_group["lr"] = current_lr_critic

        # Entropy coefficient decay
        current_entropy_coef = self.initial_entropy_coef
        if self.entropy_decay:
            current_entropy_coef = (
                self.initial_entropy_coef + (self.final_entropy_coef - self.initial_entropy_coef) * frac
            )
            current_entropy_coef = max(self.final_entropy_coef, current_entropy_coef)  # Ensure not below final value

        # Get data from Buffer
        states, actions, centralized_states, rewards, values, dones = self.replay_buffer.get_tensors(
            self.device, self.num_agents, self.state_dim, self.action_dim
        )

        # Calculate GAE and Returns
        # Use centralized Critic's Values and last_centralized_value
        advantages, returns = self._calculate_gae(rewards, values, dones, last_centralized_value)
        # Normalize advantages (for the whole batch)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Update loop
        self.old_actor.load_state_dict(self.actor.state_dict())  # Sync old Actor network
        buffer_size = self.replay_buffer.size()
        indices = np.arange(buffer_size)

        # Prepare Actor update data shapes
        # Reshape states and actions to (buffer_size * num_agents, dim)
        actor_states = states.view(-1, self.state_dim)
        actor_actions = actions.view(-1, self.action_dim)
        # Expand advantages to match Actor's input: (buffer_size * num_agents, 1)
        # Each agent shares the same advantage value at the same step
        actor_advantages = advantages.unsqueeze(1).repeat(1, self.num_agents, 1).view(-1, 1)

        # Calculate old policy's log probabilities
        with torch.no_grad():
            old_dist = self.old_actor.get_distribution(actor_states)
            actor_old_log_probs = old_dist.log_prob(actor_actions).sum(dim=-1, keepdim=True)

        # Prepare Critic update data shapes
        # Critic uses centralized states and original returns
        critic_centralized_states = centralized_states
        critic_returns = returns

        # Multi-epoch training
        for epoch in range(self.epochs):
            # Update using minibatches
            # Note: minibatch is based on *timestep* indices
            np.random.shuffle(indices)
            for start in range(0, buffer_size, self.minibatch_size):
                end = start + self.minibatch_size
                batch_indices = indices[start:end]
                if len(batch_indices) == 0:
                    continue

                # Extract Minibatch data
                # Actor data (needs extraction from actor_xxx)
                agent_batch_indices = []
                for idx in batch_indices:
                    agent_batch_indices.extend(range(idx * self.num_agents, (idx + 1) * self.num_agents))

                batch_actor_states = actor_states[agent_batch_indices]
                batch_actor_actions = actor_actions[agent_batch_indices]
                batch_actor_advantages = actor_advantages[agent_batch_indices]
                batch_actor_old_log_probs = actor_old_log_probs[agent_batch_indices]

                # Critic data (needs extraction from critic_xxx)
                batch_critic_centralized_states = critic_centralized_states[batch_indices]
                batch_critic_returns = critic_returns[batch_indices]

                # Calculate Actor Loss
                new_dist = self.actor.get_distribution(batch_actor_states)
                batch_actor_log_probs = new_dist.log_prob(batch_actor_actions).sum(dim=-1, keepdim=True)
                entropy = new_dist.entropy().mean()  # Average entropy

                # Calculate PPO Ratio
                ratio = torch.exp(batch_actor_log_probs - batch_actor_old_log_probs)

                # PPO Clipped Objective Function
                surr1 = ratio * batch_actor_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * batch_actor_advantages
                # Actor Loss = - (Clipped Surrogate Objective + Entropy Bonus)
                # We want to maximize the objective, so take negative
                actor_loss = -torch.min(surr1, surr2).mean() - current_entropy_coef * entropy

                # Calculate Critic Loss
                current_centralized_values = self.critic(batch_critic_centralized_states)
                critic_loss = self.mse_loss(current_centralized_values, batch_critic_returns)

                # Update Actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # Update Critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

        # Clear Replay Buffer
        self.replay_buffer.clear()

        # Return current LR and entropy coefficient for logging
        return current_lr_actor, current_lr_critic, current_entropy_coef

    def save_model(self, actor_path, critic_path):
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        print(f"Model saved: Actor '{actor_path}', Critic '{critic_path}'")  # English print

    def load_model(self, actor_path, critic_path):
        self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
        self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
        self.old_actor.load_state_dict(self.actor.state_dict())  # Sync old network
        # Set to evaluation mode
        self.actor.eval()
        self.critic.eval()
        self.old_actor.eval()
        print(f"Model loaded: Actor '{actor_path}', Critic '{critic_path}'")  # English print
        self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
        self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
        self.old_actor.load_state_dict(self.actor.state_dict())  # 同步旧网络
        # 设置为评估模式
        self.actor.eval()
        self.critic.eval()
        self.old_actor.eval()
        print(f"模型已加载: Actor '{actor_path}', Critic '{critic_path}'")
