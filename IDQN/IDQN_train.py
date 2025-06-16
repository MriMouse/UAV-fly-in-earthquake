import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
import os
import time
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from env import Env_MAPPO

# DQN参数
BUFFER_SIZE = int(1e5)
BATCH_SIZE = 128
GAMMA = 0.99
TAU = 1e-3
LR = 5e-4
UPDATE_EVERY_AGENT_STEPS = 4
TARGET_UPDATE_EVERY_LEARN_STEPS = 100

# 探索参数
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY_RATE = 0.995

# 动作离散化参数
N_ANGLE_LEVELS = 4
ANGLE_LEVELS = np.linspace(0, 2 * np.pi * (1 - 1 / N_ANGLE_LEVELS), N_ANGLE_LEVELS)

N_DIST_LEVELS = 2
DIST_LEVELS = np.array([0.0, 1.0])

N_OFFLOAD_LEVELS = 2
OFFLOAD_RATIO_LEVELS = np.array([0.0, 1.0])

# 训练设置
SEED = 42
SAVE_MODELS = True
MODEL_SAVE_DIR_BASE = "models_idqn_env_mappo_simplified_actions"

device = torch.device("cpu")


# Q网络定义
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_dims=(256, 256), seed=0):
        super(QNetwork, self).__init__()
        torch.manual_seed(seed)

        layers = []
        input_dim = state_size
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, action_size))
        self.network = nn.Sequential(*layers)

    def forward(self, state):
        return self.network(state)


# 经验回放
Experience = namedtuple("Experience", field_names=["state", "action_idx", "reward", "next_state", "done"])

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, seed):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        random.seed(seed)

    def add(self, state, action_idx, reward, next_state, done):
        self.memory.append(Experience(state, action_idx, reward, next_state, done))

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float().to(device)
        action_indices = torch.from_numpy(np.vstack([e.action_idx for e in experiences])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(device)
        return (states, action_indices, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)


# IDQN智能体类
class DQNAgent:
    def __init__(self, agent_id, state_size, num_ugv_targets_for_q_net, agent_seed):
        self.agent_id = agent_id
        self.num_ugv_targets = num_ugv_targets_for_q_net

        ugv_choices = max(1, self.num_ugv_targets)
        self.total_discrete_action_size = ugv_choices * N_ANGLE_LEVELS * N_DIST_LEVELS * N_OFFLOAD_LEVELS

        self.qnetwork_local = QNetwork(state_size, self.total_discrete_action_size, seed=agent_seed).to(device)
        self.qnetwork_target = QNetwork(state_size, self.total_discrete_action_size, seed=agent_seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, agent_seed)
        self.env_step_counter = 0
        self.learn_step_counter = 0

    def _map_discrete_action_to_env_action(self, discrete_action_idx):
        offload_level_idx = discrete_action_idx % N_OFFLOAD_LEVELS
        temp_idx = discrete_action_idx // N_OFFLOAD_LEVELS

        dist_level_idx = temp_idx % N_DIST_LEVELS
        temp_idx = temp_idx // N_DIST_LEVELS

        angle_level_idx = temp_idx % N_ANGLE_LEVELS
        ugv_choices = max(1, self.num_ugv_targets)
        target_ugv_q_idx = temp_idx // N_ANGLE_LEVELS

        if self.num_ugv_targets == 0:
            norm_target_ugv_id = 0.0
        else:
            norm_target_ugv_id = float(target_ugv_q_idx) / float(max(1, self.num_ugv_targets))
            norm_target_ugv_id = np.clip(norm_target_ugv_id, 0.0, 1.0)

        norm_fly_angle = ANGLE_LEVELS[angle_level_idx] / (2 * np.pi)
        norm_fly_angle = np.clip(norm_fly_angle, 0.0, 1.0)

        norm_fly_dist_ratio = DIST_LEVELS[dist_level_idx]
        norm_offload_ratio = OFFLOAD_RATIO_LEVELS[offload_level_idx]

        action_normalized_0_1 = np.array(
            [norm_target_ugv_id, norm_fly_angle, norm_fly_dist_ratio, norm_offload_ratio], dtype=np.float32
        )

        env_action_minus1_1 = (action_normalized_0_1 * 2.0) - 1.0
        return env_action_minus1_1.tolist()

    def act(self, state, eps=0.0):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state_tensor)
        self.qnetwork_local.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.total_discrete_action_size))

    def step_and_learn(self, state, action_idx, reward, next_state, done):
        self.memory.add(state, action_idx, reward, next_state, done)
        self.env_step_counter = (self.env_step_counter + 1) % UPDATE_EVERY_AGENT_STEPS

        if self.env_step_counter == 0 and len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self._learn_from_experiences(experiences)
            self.learn_step_counter += 1
            if self.learn_step_counter % TARGET_UPDATE_EVERY_LEARN_STEPS == 0:
                self._soft_update_target_network()

    def _learn_from_experiences(self, experiences):
        states, action_indices, rewards, next_states, dones = experiences
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (GAMMA * Q_targets_next * (1 - dones))
        Q_expected = self.qnetwork_local(states).gather(1, action_indices)
        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _soft_update_target_network(self):
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(TAU * local_param.data + (1.0 - TAU) * target_param.data)


# 训练主函数
def train_idqn_env_mappo():
    print(f"Using device: {device}")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    env = Env_MAPPO()

    num_agents = env.num_uavs
    state_size_per_agent = env.state_dimension
    num_ugv_targets_for_q_net = env.num_ugvs
    max_t_per_episode = env.max_steps
    NUM_EPISODES = 10000

    print(f"Env: {type(env).__name__}")
    print(f"Number of UAVs (Agents): {num_agents}")
    print(f"State dimension per UAV: {state_size_per_agent}")
    print(f"Number of UGV targets for Q-net: {num_ugv_targets_for_q_net}")
    print(
        f"Action discretizaton: UGV targets={max(1,num_ugv_targets_for_q_net)}, Angles={N_ANGLE_LEVELS}, Distances={N_DIST_LEVELS}, Offloads={N_OFFLOAD_LEVELS}"
    )

    temp_total_actions = max(1, num_ugv_targets_for_q_net) * N_ANGLE_LEVELS * N_DIST_LEVELS * N_OFFLOAD_LEVELS
    print(f"Total discrete actions per UAV Q-network: {temp_total_actions}")
    if temp_total_actions > 100:
        print(
            f"Warning: Number of discrete actions ({temp_total_actions}) is somewhat high, learning might be challenging."
        )

    agents = [
        DQNAgent(
            agent_id=i,
            state_size=state_size_per_agent,
            num_ugv_targets_for_q_net=num_ugv_targets_for_q_net,
            agent_seed=SEED + i * 10,
        )
        for i in range(num_agents)
    ]

    scores_deque = deque(maxlen=100)
    scores_history = []
    epsilon = EPS_START

    model_dir = None
    if SAVE_MODELS:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        scenario_name = (
            f"IDQN_{type(env).__name__}_{num_agents}UAVs_{num_ugv_targets_for_q_net}UGVs_Act{temp_total_actions}"
        )
        model_name_prefix = f"{scenario_name}_{timestamp}"
        model_dir = os.path.join(MODEL_SAVE_DIR_BASE, model_name_prefix)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
        print(f"Models and rewards will be saved in: {model_dir}")

    # 记录奖励达到阈值的时间
    reward_time_dict = {}
    last_recorded_threshold = None
    threshold_step = 50
    time_1000_episodes = None
    time_5000_episodes = None
    start_time = time.time()

    total_env_steps_elapsed = 0
    for i_episode in range(1, NUM_EPISODES + 1):
        states_list = env.reset()
        current_episode_reward = 0.0

        for t in range(max_t_per_episode):
            actions_for_env = []
            discrete_action_indices_chosen = []

            for i in range(num_agents):
                agent_state = states_list[i]
                discrete_action_idx = agents[i].act(agent_state, epsilon)
                discrete_action_indices_chosen.append(discrete_action_idx)
                env_action_minus1_1 = agents[i]._map_discrete_action_to_env_action(discrete_action_idx)
                actions_for_env.append(env_action_minus1_1)

            next_states_list, rewards_list_from_env, dones_list_from_env, _ = env.step(actions_for_env)

            shared_reward_this_step = rewards_list_from_env[0] if rewards_list_from_env else 0.0
            is_terminal_this_step = dones_list_from_env[0] if dones_list_from_env else True

            for i in range(num_agents):
                agents[i].step_and_learn(
                    states_list[i],
                    discrete_action_indices_chosen[i],
                    shared_reward_this_step,
                    next_states_list[i],
                    is_terminal_this_step,
                )

            states_list = next_states_list
            current_episode_reward += shared_reward_this_step
            total_env_steps_elapsed += 1

            if is_terminal_this_step:
                break

        scores_deque.append(current_episode_reward)
        scores_history.append(current_episode_reward)
        epsilon = max(EPS_END, EPS_DECAY_RATE * epsilon)

        # 记录奖励达到阈值的时间
        threshold = int(np.floor(current_episode_reward / threshold_step)) * threshold_step
        if (
            last_recorded_threshold is None or threshold != last_recorded_threshold
        ) and threshold not in reward_time_dict:
            elapsed_time = time.time() - start_time
            reward_time_dict[threshold] = (i_episode, elapsed_time)
            last_recorded_threshold = threshold

        # 记录1000回合用时
        if i_episode == 1000 and time_1000_episodes is None:
            time_1000_episodes = time.time() - start_time

        if i_episode == 5000 and time_5000_episodes is None:
            time_5000_episodes = time.time() - start_time

        avg_score_last_100 = np.mean(scores_deque)
        print(
            f"\rEp {i_episode}/{NUM_EPISODES}\tAvg Reward(100): {avg_score_last_100:.2f}\tReward: {current_episode_reward:.2f}\tEps: {epsilon:.3f}",
            end="",
        )
        if i_episode % 100 == 0:
            print(
                f"\rEp {i_episode}/{NUM_EPISODES}\tAvg Reward(100): {avg_score_last_100:.2f}\tReward: {current_episode_reward:.2f}\tEps: {epsilon:.3f}\tSteps: {total_env_steps_elapsed}"
            )

    if SAVE_MODELS and model_dir:
        for i_agent, agent in enumerate(agents):
            torch.save(agent.qnetwork_local.state_dict(), os.path.join(model_dir, f"qnetwork_agent{i_agent}_final.pth"))

        # 保存奖励历史
        reward_txt_filename_consistent = "IDQNreward.txt"
        reward_txt_path_for_plotting = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), reward_txt_filename_consistent
        )

        try:
            with open(reward_txt_path_for_plotting, "w") as f:
                for r_val in scores_history:
                    f.write(f"{r_val}\n")
            print(f"\nIDQN (Env_MAPPO) reward history for plotting saved to: {reward_txt_path_for_plotting}")
        except Exception as e:
            print(f"Error saving IDQN (Env_MAPPO) reward history: {e}")

        # 保存时间成本文件
        time_cost_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "time_cost.txt")
        try:
            with open(time_cost_path, "w") as f:
                f.write("# 达到各奖励阈值(50为步长)的回合数和用时（秒）\n")
                for th in sorted(reward_time_dict.keys()):
                    ep, t = reward_time_dict[th]
                    f.write(f"reward>={th}: episode={ep}, time={t:.2f}s\n")
                if time_1000_episodes is not None:
                    f.write(f"\n训练1000回合用时: {time_1000_episodes:.2f}s\n")
                if time_5000_episodes is not None:
                    f.write(f"训练5000回合用时: {time_5000_episodes:.2f}s\n")
            print(f"time_cost.txt 已保存到: {time_cost_path}")
        except Exception as e:
            print(f"保存 time_cost.txt 时出错: {e}")

    print("\nIDQN (Env_MAPPO) training finished.")


# 主程序入口
if __name__ == "__main__":
    train_idqn_env_mappo()
